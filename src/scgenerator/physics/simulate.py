import multiprocessing
import multiprocessing.connection
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Type, Union

import numpy as np
from send2trash import send2trash

from .. import env, utils
from ..logger import get_logger
from ..utils.parameter import Configuration, Parameters, format_variable_list
from . import pulse
from .fiber import create_non_linear_op, fast_dispersion_op

try:
    import ray
except ModuleNotFoundError:
    ray = None


class RK4IP:
    def __init__(
        self,
        params: Parameters,
        save_data=False,
        job_identifier="",
        task_id=0,
    ):
        """A 1D solver using 4th order Runge-Kutta in the interaction picture

                Parameters
                ----------
        Parameters
                    parameters of the simulation
                save_data : bool, optional
                    save calculated spectra to disk, by default False
                job_identifier : str, optional
                    string  identifying the parameter set, by default ""
                task_id : int, optional
                    unique identifier of the session, by default 0
        """
        self.set(params, save_data, job_identifier, task_id)

    def __iter__(self) -> Generator[tuple[int, int, np.ndarray], None, None]:
        yield from self.irun()

    def __len__(self) -> int:
        return self.len

    def set(
        self,
        params: Parameters,
        save_data=False,
        job_identifier="",
        task_id=0,
    ):

        self.job_identifier = job_identifier
        self.id = task_id
        self.save_data = save_data

        if self.save_data:
            self.data_dir = Path(params.output_path)
            os.makedirs(self.data_dir, exist_ok=True)
        else:
            self.data_dir = None

        self.logger = get_logger(self.job_identifier)
        self.resuming = False

        self.w_c = params.w_c
        self.w = params.w
        self.dw = self.w[1] - self.w[0]
        self.w0 = params.w0
        self.w_power_fact = params.w_power_fact
        self.alpha = params.alpha_arr
        self.spec_0 = np.sqrt(params.input_transmission) * params.spec_0
        self.z_targets = params.z_targets
        self.len = len(params.z_targets)
        self.z_final = params.length
        self.beta2_coefficients = (
            params.beta_func if params.beta_func is not None else params.beta2_coefficients
        )
        self.gamma = params.gamma_func if params.gamma_func is not None else params.gamma_arr
        self.C_to_A_factor = (params.A_eff_arr / params.A_eff_arr[0]) ** (1 / 4)
        self.behaviors = params.behaviors
        self.raman_type = params.raman_type
        self.hr_w = params.hr_w
        self.adapt_step_size = params.adapt_step_size
        self.error_ok = params.tolerated_error if self.adapt_step_size else params.step_size
        self.dynamic_dispersion = params.dynamic_dispersion
        self.starting_num = params.recovery_last_stored

        self._setup_functions()
        self._setup_sim_parameters()

    def _setup_functions(self):
        self.N_func = create_non_linear_op(
            self.behaviors, self.w_c, self.w0, self.gamma, self.raman_type, hr_w=self.hr_w
        )

        if self.dynamic_dispersion:
            self.disp = lambda r: fast_dispersion_op(
                self.w_c, self.beta2_coefficients(r), self.w_power_fact, alpha=self.alpha
            )
        else:
            self.disp = lambda r: fast_dispersion_op(
                self.w_c, self.beta2_coefficients, self.w_power_fact, alpha=self.alpha
            )

        # Set up which quantity is conserved for adaptive step size
        if self.adapt_step_size:
            if "raman" in self.behaviors and self.alpha is not None:
                self.logger.debug("Conserved quantity : photon number with loss")
                self.conserved_quantity_func = lambda spectrum, h: pulse.photon_number_with_loss(
                    spectrum, self.w, self.dw, self.gamma, self.alpha, h
                )
            elif "raman" in self.behaviors:
                self.logger.debug("Conserved quantity : photon number without loss")
                self.conserved_quantity_func = lambda spectrum, h: pulse.photon_number(
                    spectrum, self.w, self.dw, self.gamma
                )
            elif self.alpha is not None:
                self.logger.debug("Conserved quantity : energy with loss")
                self.conserved_quantity_func = lambda spectrum, h: pulse.pulse_energy_with_loss(
                    self.C_to_A_factor * spectrum, self.dw, self.alpha, h
                )
            else:
                self.logger.debug("Conserved quantity : energy without loss")
                self.conserved_quantity_func = lambda spectrum, h: pulse.pulse_energy(
                    self.C_to_A_factor * spectrum, self.dw
                )
        else:
            self.conserved_quantity_func = lambda spectrum, h: 0.0

    def _setup_sim_parameters(self):
        # making sure to keep only the z that we want
        self.z_stored = list(self.z_targets.copy()[0 : self.starting_num + 1])
        self.z_targets = list(self.z_targets.copy()[self.starting_num :])
        self.z_targets.sort()
        self.store_num = len(self.z_targets)

        # Initial setup of simulation parameters
        self.d_w = self.w_c[1] - self.w_c[0]  # resolution of the frequency grid
        self.z = self.z_targets.pop(0)

        # Setup initial values for every physical quantity that we want to track
        self.current_spectrum = self.spec_0.copy() / self.C_to_A_factor
        self.stored_spectra = self.starting_num * [None] + [self.current_spectrum.copy()]
        self.cons_qty = [
            self.conserved_quantity_func(self.current_spectrum, 0),
            0,
        ]
        self.size_fac = 2 ** (1 / 5)

        # Initial step size
        if self.adapt_step_size:
            self.initial_h = (self.z_targets[0] - self.z) / 2
        else:
            self.initial_h = self.error_ok

    def _save_current_spectrum(self, num: int):
        """saves the spectrum and the corresponding cons_qty array

        Parameters
        ----------
        num : int
            index of the z postition
        """
        self._save_data(self.C_to_A_factor * self.current_spectrum, f"spectrum_{num}")
        self._save_data(self.cons_qty, f"cons_qty")
        self.step_saved()

    def get_current_spectrum(self) -> tuple[int, np.ndarray]:
        """returns the current spectrum

        Returns
        -------
        np.ndarray
            spectrum
        """
        return self.C_to_A_factor * self.current_spectrum

    def _save_data(self, data: np.ndarray, name: str):
        """calls the appropriate method to save data

        Parameters
        ----------
        data : np.ndarray
            data to save
        name : str
            file name
        """
        utils.save_data(data, self.data_dir, name)

    def run(self) -> list[np.ndarray]:
        time_start = datetime.today()

        for step, num, _ in self.irun():
            if self.save_data:
                self._save_current_spectrum(num)

        self.logger.info(
            "propagation finished in {} steps ({} seconds)".format(
                step, (datetime.today() - time_start).total_seconds()
            )
        )

        if self.save_data:
            self._save_data(self.z_stored, "z.npy")

        return self.stored_spectra

    def irun(self) -> Generator[tuple[int, int, np.ndarray], None, None]:

        # Print introduction
        self.logger.debug(
            "Computing {} new spectra, first one at {}m".format(self.store_num, self.z_targets[0])
        )

        # Start of the integration
        step = 1
        h_taken = self.initial_h
        h_next_step = self.initial_h
        store = False  # store a spectrum

        yield step, len(self.stored_spectra) - 1, self.get_current_spectrum()

        while self.z < self.z_final:
            h_taken, h_next_step, self.current_spectrum = self.take_step(
                step, h_next_step, self.current_spectrum.copy()
            )

            self.z += h_taken
            step += 1
            self.cons_qty.append(0)

            # Whether the current spectrum has to be stored depends on previous step
            if store:
                self.logger.debug("{} steps, z = {:.4f}, h = {:.5g}".format(step, self.z, h_taken))

                self.stored_spectra.append(self.current_spectrum)

                yield step, len(self.stored_spectra) - 1, self.get_current_spectrum()

                self.z_stored.append(self.z)
                del self.z_targets[0]

                # reset the constant step size after a spectrum is stored
                if not self.adapt_step_size:
                    h_next_step = self.error_ok

                if len(self.z_targets) == 0:
                    break
                store = False

            # if the next step goes over a position at which we want to store
            # a spectrum, we shorten the step to reach this position exactly
            if self.z + h_next_step >= self.z_targets[0]:
                store = True
                h_next_step = self.z_targets[0] - self.z

    def take_step(
        self, step: int, h_next_step: float, current_spectrum: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """computes a new spectrum, whilst adjusting step size if required, until the error estimation
        validates the new spectrum

        Parameters
        ----------
        step : int
            index of the current
        h_next_step : float
            candidate step size
        current_spectrum : np.ndarray
            spectrum of the last step taken

        Returns
        -------
        h : float
            step sized used
        h_next_step : float
            candidate next step size
        new_spectrum : np.ndarray
            new spectrum
        """
        keep = False
        while not keep:
            h = h_next_step
            z_ratio = self.z / self.z_final

            expD = np.exp(h / 2 * self.disp(z_ratio))

            A_I = expD * current_spectrum
            k1 = expD * (h * self.N_func(current_spectrum, z_ratio))
            k2 = h * self.N_func(A_I + k1 / 2, z_ratio)
            k3 = h * self.N_func(A_I + k2 / 2, z_ratio)
            k4 = h * self.N_func(expD * (A_I + k3), z_ratio)
            new_spectrum = expD * (A_I + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6

            if self.adapt_step_size:
                self.cons_qty[step] = self.conserved_quantity_func(new_spectrum, h)
                curr_p_change = np.abs(self.cons_qty[step - 1] - self.cons_qty[step])
                cons_qty_change_ok = self.error_ok * self.cons_qty[step - 1]

                if curr_p_change > 2 * cons_qty_change_ok:
                    progress_str = f"step {step} rejected with h = {h:.4e}, doing over"
                    self.logger.debug(progress_str)
                    keep = False
                    h_next_step = h / 2
                elif cons_qty_change_ok < curr_p_change <= 2 * cons_qty_change_ok:
                    keep = True
                    h_next_step = h / self.size_fac
                elif curr_p_change < 0.1 * cons_qty_change_ok:
                    keep = True
                    h_next_step = h * self.size_fac
                else:
                    keep = True
                    h_next_step = h
            else:
                keep = True
        return h, h_next_step, new_spectrum

    def step_saved(self):
        pass


class SequentialRK4IP(RK4IP):
    def __init__(
        self,
        params: Parameters,
        pbars: utils.PBars,
        save_data=False,
        job_identifier="",
        task_id=0,
    ):
        self.pbars = pbars
        super().__init__(
            params,
            save_data=save_data,
            job_identifier=job_identifier,
            task_id=task_id,
        )

    def step_saved(self):
        self.pbars.update(1, self.z / self.z_final - self.pbars[1].n)


class MutliProcRK4IP(RK4IP):
    def __init__(
        self,
        params: Parameters,
        p_queue: multiprocessing.Queue,
        worker_id: int,
        save_data=False,
        job_identifier="",
        task_id=0,
    ):
        self.worker_id = worker_id
        self.p_queue = p_queue
        super().__init__(
            params,
            save_data=save_data,
            job_identifier=job_identifier,
            task_id=task_id,
        )

    def step_saved(self):
        self.p_queue.put((self.worker_id, self.z / self.z_final))


class RayRK4IP(RK4IP):
    def __init__(self):
        pass

    def set(
        self,
        params: Parameters,
        p_actor,
        worker_id: int,
        save_data=False,
        job_identifier="",
        task_id=0,
    ):
        self.worker_id = worker_id
        self.p_actor = p_actor
        super().set(
            params,
            save_data=save_data,
            job_identifier=job_identifier,
            task_id=task_id,
        )

    def set_and_run(self, v):
        params, p_actor, worker_id, save_data, job_identifier, task_id = v
        self.set(params, p_actor, worker_id, save_data, job_identifier, task_id)
        self.run()

    def step_saved(self):
        self.p_actor.update.remote(self.worker_id, self.z / self.z_final)
        self.p_actor.update.remote(0)


class Simulations:
    """The recommended way to run simulations.
    New Simulations child classes can be written and must implement the following
    """

    simulation_methods: list[tuple[Type["Simulations"], int]] = []
    simulation_methods_dict: dict[str, Type["Simulations"]] = dict()

    def __init_subclass__(cls, priority=0, **kwargs):
        Simulations.simulation_methods.append((cls, priority))
        Simulations.simulation_methods_dict[cls.__name__] = cls
        Simulations.simulation_methods.sort(key=lambda el: el[1], reverse=True)
        super().__init_subclass__(**kwargs)

    @classmethod
    def get_best_method(cls):
        for method, _ in Simulations.simulation_methods:
            if method.is_available():
                return method

    @classmethod
    def is_available(cls) -> bool:
        return False

    @classmethod
    def new(
        cls, configuration: Configuration, task_id, method: Union[str, Type["Simulations"]] = None
    ) -> "Simulations":
        """Prefered method to create a new simulations object

        Returns
        -------
        Simulations
            obj that uses the best available parallelization method
        """
        if method is not None:
            if isinstance(method, str):
                method = Simulations.simulation_methods_dict[method]
            return method(configuration, task_id)
        elif configuration.num_sim > 1 and configuration.parallel:
            return Simulations.get_best_method()(configuration, task_id)
        else:
            return SequencialSimulations(configuration, task_id)

    def __init__(self, configuration: Configuration, task_id=0):
        """
        Parameters
        ----------
        configuration : scgenerator.Configuration obj
            parameter sequence
        task_id : int, optional
            a unique id that identifies the simulation, by default 0
        data_folder : str, optional
            path to the folder where data is saved, by default "scgenerator/"
        """
        if not self.is_available():
            raise RuntimeError(f"{self.__class__} is currently not available")
        self.logger = get_logger(__name__)
        self.id = int(task_id)

        self.configuration = configuration

        self.name = self.configuration.final_path
        self.sim_dir = self.configuration.final_sim_dir
        self.configuration.save_parameters()

        self.sim_jobs_per_node = 1

    def finished_and_complete(self):
        for sim in self.configuration.all_configs_dict.values():
            if (
                self.configuration.sim_status(sim.output_path)[0]
                != self.configuration.State.COMPLETE
            ):
                return False
        return True

    def run(self):
        self._run_available()
        self.ensure_finised_and_complete()

    def _run_available(self):
        for variable, params in self.configuration:
            v_list_str = format_variable_list(variable, add_iden=True)
            utils.save_parameters(params.prepare_for_dump(), Path(params.output_path))

            self.new_sim(v_list_str, params)
        self.finish()

    def new_sim(self, v_list_str: str, params: Parameters):
        """responsible to launch a new simulation

        Parameters
        ----------
        v_list_str : str
            string that uniquely identifies the simulation as returned by utils.format_variable_list
        params : Parameters
            computed parameters
        """
        raise NotImplementedError()

    def finish(self):
        """called once all the simulations are launched."""
        raise NotImplementedError()

    def ensure_finised_and_complete(self):
        while not self.finished_and_complete():
            self.logger.warning(f"Something wrong happened, running again to finish simulation")
            self._run_available()

    def stop(self):
        raise NotImplementedError()


class SequencialSimulations(Simulations, priority=0):
    @classmethod
    def is_available(cls):
        return True

    def __init__(self, configuration: Configuration, task_id):
        super().__init__(configuration, task_id=task_id)
        self.pbars = utils.PBars(
            self.configuration.total_num_steps, "Simulating " + self.configuration.final_path, 1
        )
        self.configuration.skip_callback = lambda num: self.pbars.update(0, num)

    def new_sim(self, v_list_str: str, params: Parameters):
        self.logger.info(
            f"{self.configuration.final_path} : launching simulation with {v_list_str}"
        )
        SequentialRK4IP(
            params, self.pbars, save_data=True, job_identifier=v_list_str, task_id=self.id
        ).run()

    def stop(self):
        pass

    def finish(self):
        self.pbars.close()


class MultiProcSimulations(Simulations, priority=1):
    @classmethod
    def is_available(cls):
        return True

    def __init__(self, configuration: Configuration, task_id):
        super().__init__(configuration, task_id=task_id)
        if configuration.worker_num is not None:
            self.sim_jobs_per_node = configuration.worker_num
        else:
            self.sim_jobs_per_node = max(1, os.cpu_count() // 2)
        self.queue = multiprocessing.JoinableQueue(self.sim_jobs_per_node)
        self.progress_queue = multiprocessing.Queue()
        self.configuration.skip_callback = lambda num: self.progress_queue.put((0, num))
        self.workers = [
            multiprocessing.Process(
                target=MultiProcSimulations.worker,
                args=(self.id, i + 1, self.queue, self.progress_queue),
            )
            for i in range(self.sim_jobs_per_node)
        ]
        self.p_worker = multiprocessing.Process(
            target=utils.progress_worker,
            args=(
                self.configuration.final_path,
                self.sim_jobs_per_node,
                self.configuration.total_num_steps,
                self.progress_queue,
            ),
        )
        self.p_worker.start()

    def run(self):
        for worker in self.workers:
            worker.start()
        super().run()

    def new_sim(self, v_list_str: str, params: Parameters):
        self.queue.put((v_list_str, params), block=True, timeout=None)

    def finish(self):
        """0 means finished"""
        for worker in self.workers:
            self.queue.put(0)
        for worker in self.workers:
            worker.join()
        self.queue.join()
        self.progress_queue.put(0)

    def stop(self):
        self.finish()

    @staticmethod
    def worker(
        task_id,
        worker_id: int,
        queue: multiprocessing.JoinableQueue,
        p_queue: multiprocessing.Queue,
    ):
        while True:
            raw_data: tuple[list[tuple], Parameters] = queue.get()
            if raw_data == 0:
                queue.task_done()
                return
            v_list_str, params = raw_data
            MutliProcRK4IP(
                params,
                p_queue,
                worker_id,
                save_data=True,
                job_identifier=v_list_str,
                task_id=task_id,
            ).run()
            queue.task_done()


class RaySimulations(Simulations, priority=2):
    """runs simulation with the help of the ray module.
    ray must be initialized before creating an instance of RaySimulations"""

    @classmethod
    def is_available(cls):
        if ray:
            return ray.is_initialized()
        return False

    def __init__(
        self,
        configuration: Configuration,
        task_id=0,
    ):
        super().__init__(configuration, task_id)

        nodes = ray.nodes()
        self.logger.info(
            f"{len(nodes)} node{'s' if len(nodes) > 1 else ''} in the Ray cluster : "
            + str(
                [
                    (node.get("NodeManagerHostname", "unknown"), node.get("Resources", {}))
                    for node in nodes
                ]
            )
        )

        self.propagator = ray.remote(RayRK4IP)

        self.update_cluster_frequency = 3
        self.jobs = []
        self.pool = ray.util.ActorPool(self.propagator.remote() for _ in range(self.sim_jobs_total))
        self.num_submitted = 0
        self.rolling_id = 0
        self.p_actor = ray.remote(utils.ProgressBarActor).remote(
            self.configuration.final_path, self.sim_jobs_total, self.configuration.total_num_steps
        )
        self.configuration.skip_callback = lambda num: ray.get(self.p_actor.update.remote(0, num))

    def new_sim(self, v_list_str: str, params: Parameters):
        while self.num_submitted >= self.sim_jobs_total:
            self.collect_1_job()

        self.rolling_id = (self.rolling_id + 1) % self.sim_jobs_total
        self.pool.submit(
            lambda a, v: a.set_and_run.remote(v),
            (
                params,
                self.p_actor,
                self.rolling_id + 1,
                True,
                v_list_str,
                self.id,
            ),
        )
        self.num_submitted += 1

        self.logger.info(
            f"{self.configuration.final_path} : launching simulation with {v_list_str}"
        )

    def collect_1_job(self):
        ray.get(self.p_actor.update_pbars.remote())
        try:
            self.pool.get_next_unordered(self.update_cluster_frequency)
            ray.get(self.p_actor.update_pbars.remote())
            self.num_submitted -= 1
        except TimeoutError:
            return

    def finish(self):
        while self.num_submitted > 0:
            self.collect_1_job()
        ray.get(self.p_actor.close.remote())

    def stop(self):
        ray.shutdown()

    @property
    def sim_jobs_total(self):
        if self.configuration.worker_num is not None:
            return self.configuration.worker_num
        tot_cpus = ray.cluster_resources().get("CPU", 1)
        return int(min(self.configuration.num_sim, tot_cpus))


def run_simulation(
    config_file: os.PathLike,
    method: Union[str, Type[Simulations]] = None,
):
    config = Configuration(config_file)

    sim = new_simulation(config, method)
    sim.run()
    path_trees = utils.build_path_trees(config.sim_dirs[-1])

    final_name = env.get(env.OUTPUT_PATH)
    if final_name is None:
        final_name = config.final_path

    utils.merge(final_name, path_trees)
    try:
        send2trash(config.sim_dirs)
    except (PermissionError, OSError):
        get_logger(__name__).error("Could not send temporary directories to trash")


def new_simulation(
    configuration: Configuration,
    method: Union[str, Type[Simulations]] = None,
) -> Simulations:
    logger = get_logger(__name__)
    task_id = random.randint(1e9, 1e12)
    logger.info(f"running {configuration.final_path}")
    return Simulations.new(configuration, task_id, method)


def __parallel_RK4IP_worker(
    worker_id: int,
    msq_queue: multiprocessing.connection.Connection,
    data_queue: multiprocessing.Queue,
    params: Parameters,
):
    logger = get_logger(__name__)
    logger.debug(f"workder {worker_id} started")
    for out in RK4IP(params).irun():
        logger.debug(f"worker {worker_id} waiting for msg")
        msq_queue.recv()
        logger.debug(f"worker {worker_id} got msg")
        data_queue.put((worker_id, out))
        logger.debug(f"worker {worker_id} sent data")


def parallel_RK4IP(
    config: os.PathLike,
) -> Generator[
    tuple[tuple[list[tuple[str, Any]], Parameters, int, int, np.ndarray], ...], None, None
]:
    logger = get_logger(__name__)
    params = list(Configuration(config))
    n = len(params)
    z_num = params[0][1].z_num

    cpu_no = multiprocessing.cpu_count()
    if len(params) < cpu_no:
        cpu_no = len(params)

    pipes = [multiprocessing.Pipe(duplex=False) for i in range(n)]
    data_queue = multiprocessing.Queue()
    workers = [
        multiprocessing.Process(target=__parallel_RK4IP_worker, args=(i, pipe[0], data_queue, p[1]))
        for i, (pipe, p) in enumerate(zip(pipes, params))
    ]
    try:
        [w.start() for w in workers]
        logger.debug("pool started")
        for i in range(z_num):
            for q in pipes:
                q[1].send(0)
                logger.debug("msg sent")
            computed_dict: dict[int, np.ndarray] = {}
            for j in range(n):
                w_id, computed = data_queue.get()
                computed_dict[w_id] = computed
            computed_dict = list(computed_dict.items())
            computed_dict.sort()
            yield tuple((*p, *c) for p, c in zip(params, [el[1] for el in computed_dict]))
        print("finished")
    finally:
        for w, cs in zip(workers, pipes):
            w.join()
            w.close()
            cs[0].close()
            cs[1].close()
        data_queue.close()


if __name__ == "__main__":
    pass
