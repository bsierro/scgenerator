import multiprocessing
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import numpy as np

from .. import env, initialize, io, utils
from ..const import PARAM_SEPARATOR
from ..errors import IncompleteDataFolderError
from ..logger import get_logger
from . import pulse
from .fiber import create_non_linear_op, fast_dispersion_op

try:
    import ray
except ModuleNotFoundError:
    ray = None


class RK4IP:
    def __init__(
        self,
        params: initialize.Params,
        save_data=False,
        job_identifier="",
        task_id=0,
    ):
        """A 1D solver using 4th order Runge-Kutta in the interaction picture

        Parameters
        ----------
        params : Params
            parameters of the simulation
        save_data : bool, optional
            save calculated spectra to disk, by default False
        job_identifier : str, optional
            string  identifying the parameter set, by default ""
        task_id : int, optional
            unique identifier of the session, by default 0
        """
        self.set(params, save_data, job_identifier, task_id)

    def set(
        self,
        params: initialize.Params,
        save_data=False,
        job_identifier="",
        task_id=0,
    ):

        self.job_identifier = job_identifier
        self.id = task_id

        self.sim_dir = io.get_sim_dir(self.id)
        self.sim_dir.mkdir(exist_ok=True)
        self.data_dir = self.sim_dir / self.job_identifier

        self.logger = get_logger(self.job_identifier)
        self.resuming = False
        self.save_data = save_data

        self.w_c = params.w_c
        self.w = params.w
        self.dw = self.w[1] - self.w[0]
        self.w0 = params.w0
        self.w_power_fact = params.w_power_fact
        self.alpha = params.alpha
        self.spec_0 = np.sqrt(params.input_transmission) * params.spec_0
        self.z_targets = params.z_targets
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
        self.error_ok = params.tolerated_error
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
                    spectrum, self.dw, self.alpha, h
                )
            else:
                self.logger.debug("Conserved quantity : energy without loss")
                self.conserved_quantity_func = lambda spectrum, h: pulse.pulse_energy(
                    spectrum, self.dw
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

        if self.save_data:
            self._save_current_spectrum(0)

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

    def _save_data(self, data: np.ndarray, name: str):
        """calls the appropriate method to save data

        Parameters
        ----------
        data : np.ndarray
            data to save
        name : str
            file name
        """
        io.save_data(data, self.data_dir, name)

    def run(self):

        # Print introduction
        self.logger.debug(
            "Computing {} new spectra, first one at {}m".format(self.store_num, self.z_targets[0])
        )

        # Start of the integration
        step = 1
        h_taken = self.initial_h
        h_next_step = self.initial_h
        store = False  # store a spectrum
        time_start = datetime.today()

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
                if self.save_data:
                    self._save_current_spectrum(len(self.stored_spectra) - 1)

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

        self.logger.info(
            "propagation finished in {} steps ({} seconds)".format(
                step, (datetime.today() - time_start).total_seconds()
            )
        )

        if self.save_data:
            self._save_data(self.z_stored, "z.npy")

        return self.stored_spectra

    def take_step(
        self, step: int, h_next_step: float, current_spectrum: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
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
        params: initialize.Params,
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
        params: initialize.Params,
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
        params: initialize.Params,
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

    simulation_methods: List[Tuple[Type["Simulations"], int]] = []
    simulation_methods_dict: Dict[str, Type["Simulations"]] = dict()

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
        cls, param_seq: initialize.ParamSequence, task_id, method: Type["Simulations"] = None
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
            return method(param_seq, task_id)
        elif param_seq.num_sim > 1 and param_seq.config.parallel:
            return Simulations.get_best_method()(param_seq, task_id)
        else:
            return SequencialSimulations(param_seq, task_id)

    def __init__(self, param_seq: initialize.ParamSequence, task_id=0):
        """
        Parameters
        ----------
        param_seq : scgenerator.initialize.ParamSequence obj
            parameter sequence
        task_id : int, optional
            a unique id that identifies the simulation, by default 0
        data_folder : str, optional
            path to the folder where data is saved, by default "scgenerator/"
        """
        if not self.is_available():
            raise RuntimeError(f"{self.__class__} is currently not available")
        self.logger = io.get_logger(__name__)
        self.id = int(task_id)

        self.update(param_seq)

        self.name = self.param_seq.name
        self.sim_dir = io.get_sim_dir(
            self.id, path_if_new=Path(self.name + PARAM_SEPARATOR + "tmp")
        )
        io.save_parameters(self.param_seq.config, self.sim_dir, file_name="initial_config.toml")

        self.sim_jobs_per_node = 1

    @property
    def finished_and_complete(self):
        try:
            io.check_data_integrity(io.get_data_dirs(self.sim_dir), self.param_seq.config.z_num)
            return True
        except IncompleteDataFolderError:
            return False

    def update(self, param_seq: initialize.ParamSequence):
        self.param_seq = param_seq

    def run(self):
        self._run_available()
        self.ensure_finised_and_complete()

    def _run_available(self):
        for variable, params in self.param_seq:
            v_list_str = utils.format_variable_list(variable)
            io.save_parameters(params, self.sim_dir / v_list_str)

            self.new_sim(v_list_str, params)
        self.finish()

    def new_sim(self, v_list_str: str, params: initialize.Params):
        """responsible to launch a new simulation

        Parameters
        ----------
        v_list_str : str
            string that uniquely identifies the simulation as returned by utils.format_variable_list
        params : initialize.Params
            computed parameters
        """
        raise NotImplementedError()

    def finish(self):
        """called once all the simulations are launched."""
        raise NotImplementedError()

    def ensure_finised_and_complete(self):
        while not self.finished_and_complete:
            self.logger.warning(f"Something wrong happened, running again to finish simulation")
            self.update(initialize.RecoveryParamSequence(self.param_seq.config, self.id))
            self._run_available()

    def stop(self):
        raise NotImplementedError()


class SequencialSimulations(Simulations, priority=0):
    @classmethod
    def is_available(cls):
        return True

    def __init__(self, param_seq: initialize.ParamSequence, task_id):
        super().__init__(param_seq, task_id=task_id)
        self.pbars = utils.PBars(self.param_seq.num_steps, "Simulating " + self.param_seq.name, 1)

    def new_sim(self, v_list_str: str, params: initialize.Params):
        self.logger.info(f"{self.param_seq.name} : launching simulation with {v_list_str}")
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

    def __init__(self, param_seq: initialize.ParamSequence, task_id):
        super().__init__(param_seq, task_id=task_id)
        if param_seq.config.worker_num is not None:
            self.sim_jobs_per_node = param_seq.config.worker_num
        else:
            self.sim_jobs_per_node = max(1, os.cpu_count() // 2)
        self.queue = multiprocessing.JoinableQueue(self.sim_jobs_per_node)
        self.progress_queue = multiprocessing.Queue()
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
                self.param_seq.name,
                self.sim_jobs_per_node,
                self.param_seq.num_steps,
                self.progress_queue,
            ),
        )
        self.p_worker.start()

    def run(self):
        for worker in self.workers:
            worker.start()
        super().run()

    def new_sim(self, v_list_str: str, params: initialize.Params):
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
            raw_data: Tuple[List[tuple], initialize.Params] = queue.get()
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
    """runs simulation with the help of the ray module. ray must be initialized before creating an instance of RaySimulations"""

    @classmethod
    def is_available(cls):
        if ray:
            return ray.is_initialized()
        return False

    def __init__(
        self,
        param_seq: initialize.ParamSequence,
        task_id=0,
    ):
        super().__init__(param_seq, task_id)

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

        self.propagator = ray.remote(RayRK4IP).options(runtime_env=dict(env_vars=env.all_environ()))

        self.update_cluster_frequency = 3
        self.jobs = []
        self.pool = ray.util.ActorPool(self.propagator.remote() for _ in range(self.sim_jobs_total))
        self.num_submitted = 0
        self.rolling_id = 0
        self.p_actor = (
            ray.remote(utils.ProgressBarActor)
            .options(runtime_env=dict(env_vars=env.all_environ()))
            .remote(self.param_seq.name, self.sim_jobs_total, self.param_seq.num_steps)
        )

    def new_sim(self, v_list_str: str, params: initialize.Params):
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

        self.logger.info(f"{self.param_seq.name} : launching simulation with {v_list_str}")

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
        if self.param_seq.config.worker_num is not None:
            return self.param_seq.config.worker_num
        tot_cpus = ray.cluster_resources().get("CPU", 1)
        return int(min(self.param_seq.num_sim, tot_cpus))


def run_simulation_sequence(
    *config_files: os.PathLike,
    method=None,
    prev_sim_dir: os.PathLike = None,
):
    configs = io.load_config_sequence(*config_files)

    prev = prev_sim_dir
    for config in configs:
        sim = new_simulation(config, prev, method)
        sim.run()
        prev = sim.sim_dir
    path_trees = io.build_path_trees(sim.sim_dir)

    final_name = env.get(env.OUTPUT_PATH)
    if final_name is None:
        final_name = config.name

    io.merge(final_name, path_trees)


def new_simulation(
    config: utils.BareConfig,
    prev_sim_dir=None,
    method: Type[Simulations] = None,
) -> Simulations:
    logger = get_logger(__name__)

    if prev_sim_dir is not None:
        config.prev_sim_dir = str(prev_sim_dir)

    task_id = random.randint(1e9, 1e12)

    if prev_sim_dir is None:
        param_seq = initialize.ParamSequence(config)
    else:
        param_seq = initialize.ContinuationParamSequence(prev_sim_dir, config)

    logger.info(f"running {param_seq.name}")

    return Simulations.new(param_seq, task_id, method)


def resume_simulations(sim_dir: Path, method: Type[Simulations] = None) -> Simulations:

    task_id = random.randint(1e9, 1e12)
    config = io.load_toml(sim_dir / "initial_config.toml")
    io.set_data_folder(task_id, sim_dir)
    param_seq = initialize.RecoveryParamSequence(config, task_id)

    return Simulations.new(param_seq, task_id, method)


if __name__ == "__main__":
    pass
