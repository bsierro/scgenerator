import multiprocessing
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from numba import jit
from tqdm import tqdm

from .. import initialize, io, utils, const
from ..errors import IncompleteDataFolderError
from ..logger import get_logger
from . import pulse
from .fiber import create_non_linear_op, fast_dispersion_op

using_ray = False
try:
    import ray

    using_ray = True
except ModuleNotFoundError:
    pass


class RK4IP:
    def __init__(self, sim_params, save_data=False, job_identifier="", task_id=0, n_percent=10):
        """A 1D solver using 4th order Runge-Kutta in the interaction picture

        Parameters
        ----------
        sim_params : dict
        a flattened parameter dictionary containing :
            w_c : numpy.ndarray
                angular frequencies centered around 0 generated with scgenerator.initialize.wspace
            w0 : float
                central angular frequency of the pulse
            w_power_fact : numpy.ndarray
                precomputed factorial/peak_power operations on w_c (scgenerator.math.power_fact)
            spec_0 : numpy.ndarray
                initial spectral envelope as function of w_c
            z_targets : list
                target distances
            length : float
                length of the fiber
            beta : numpy.ndarray or Callable[[float], numpy.ndarray]
                beta coeficients (Taylor expansion of beta(w))
            gamma : float or Callable[[float], float]
                non-linear parameter
            t : numpy.ndarray
                time
            dt : float
                time resolution
            behaviors : list(str {'ss', 'raman', 'spm'})
                behaviors to include in the simulation given as a list of strings
            raman_type : str, optional
                type of raman modelisation if raman effect is present
            f_r, hr_w : (opt) arguments of delayed_raman_t (see there for infos)
            adapt_step_size : bool, optional
                if True (default), adapts the step size with conserved quantity methode
            error_ok : float
            tolerated relative error for the adaptive step size if adaptive
            step size is turned on, otherwise length of fixed steps in m
        save_data : bool, optional
            save calculated spectra to disk, by default False
        job_identifier : str, optional
            string  identifying the parameter set, by default ""
        task_id : int, optional
            unique identifier of the session, by default 0
        n_percent : int, optional
            print/log progress update every n_percent, by default 10
        """

        self.set_new_params(sim_params, save_data, job_identifier, task_id, n_percent)

    def set_new_params(self, sim_params, save_data, job_identifier, task_id, n_percent):
        self.job_identifier = job_identifier
        self.id = task_id
        self.n_percent = n_percent
        self.logger = get_logger(self.job_identifier)
        self.resuming = False
        self.save_data = save_data
        self._extract_params(sim_params)
        self._setup_functions()
        self.starting_num = sim_params.get("recovery_last_stored", 0)
        self._setup_sim_parameters()

    def _extract_params(self, params):
        self.w_c = params.pop("w_c")
        self.w0 = params.pop("w0")
        self.w_power_fact = params.pop("w_power_fact")
        self.spec_0 = params.pop("spec_0")
        self.z_targets = params.pop("z_targets")
        self.z_final = params.pop("length")
        self.beta = params.pop("beta_func", params.pop("beta"))
        self.gamma = params.pop("gamma_func", params.pop("gamma"))
        self.behaviors = params.pop("behaviors")
        self.raman_type = params.pop("raman_type", "stolen")
        self.f_r = params.pop("f_r", 0)
        self.hr_w = params.pop("hr_w", None)
        self.adapt_step_size = params.pop("adapt_step_size", True)
        self.error_ok = params.pop("error_ok")
        self.dynamic_dispersion = params.pop("dynamic_dispersion", False)

    def _setup_functions(self):
        self.N_func = create_non_linear_op(
            self.behaviors, self.w_c, self.w0, self.gamma, self.raman_type, self.f_r, self.hr_w
        )

        if self.dynamic_dispersion:
            self.disp = lambda r: fast_dispersion_op(self.w_c, self.beta(r), self.w_power_fact)
        else:
            self.disp = lambda r: fast_dispersion_op(self.w_c, self.beta, self.w_power_fact)

        # Set up which quantity is conserved for adaptive step size
        if self.adapt_step_size:
            if "raman" in self.behaviors:
                self.conserved_quantity_func = pulse.photon_number
            else:
                self.logger.debug("energy conserved")
                self.conserved_quantity_func = pulse.pulse_energy
        else:
            self.conserved_quantity_func = lambda a, b, c, d: 0

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
        self.current_spectrum = self.spec_0.copy()
        self.stored_spectra = self.starting_num * [None] + [self.current_spectrum.copy()]
        self.cons_qty = [
            self.conserved_quantity_func(
                self.current_spectrum,
                self.w_c + self.w0,
                self.d_w,
                self.gamma,
            ),
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
        self._save_data(self.current_spectrum, f"spectrum_{num}")
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
        io.save_data(data, name, self.id, self.job_identifier)

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
                self.cons_qty[step] = self.conserved_quantity_func(
                    new_spectrum, self.w_c + self.w0, self.d_w, self.gamma
                )
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
        return h, h_next_step, new_spectrum

    def step_saved(self):
        pass


class SequentialRK4IP(RK4IP):
    def __init__(
        self,
        sim_params,
        overall_pbar: tqdm,
        save_data=False,
        job_identifier="",
        task_id=0,
        n_percent=10,
    ):
        self.overall_pbar = overall_pbar
        self.pbar = tqdm(**const.pbar_format(1))
        super().__init__(
            sim_params,
            save_data=save_data,
            job_identifier=job_identifier,
            task_id=task_id,
            n_percent=n_percent,
        )

    def step_saved(self):
        self.overall_pbar.update()
        self.pbar.update(self.z / self.z_final - self.pbar.n)


class MutliProcRK4IP(RK4IP):
    def __init__(
        self,
        sim_params,
        p_queue: multiprocessing.Queue,
        worker_id: int,
        save_data=False,
        job_identifier="",
        task_id=0,
        n_percent=10,
    ):
        self.worker_id = worker_id
        self.p_queue = p_queue
        super().__init__(
            sim_params,
            save_data=save_data,
            job_identifier=job_identifier,
            task_id=task_id,
            n_percent=n_percent,
        )

    def step_saved(self):
        self.p_queue.put((self.worker_id, self.z / self.z_final))


class RayRK4IP(RK4IP):
    def __init__(
        self,
        sim_params,
        p_actor,
        worker_id: int,
        save_data=False,
        job_identifier="",
        task_id=0,
        n_percent=10,
    ):
        self.worker_id = worker_id
        self.p_actor = p_actor
        super().__init__(
            sim_params,
            save_data=save_data,
            job_identifier=job_identifier,
            task_id=task_id,
            n_percent=n_percent,
        )

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
        cls._available = cls.is_available()
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
        self.logger = io.get_logger(__name__)
        self.id = int(task_id)

        self.update(param_seq)

        self.name = self.param_seq.name
        self.data_folder = io.get_data_folder(self.id, name_if_new=self.name)
        io.save_toml(os.path.join(self.data_folder, "initial_config.toml"), self.param_seq.config)

        self.sim_jobs_per_node = 1
        self.max_concurrent_jobs = np.inf

    @property
    def finished_and_complete(self):
        try:
            io.check_data_integrity(
                io.get_data_subfolders(self.data_folder), self.param_seq["simulation", "z_num"]
            )
            return True
        except IncompleteDataFolderError:
            return False

    def limit_concurrent_jobs(self, max_concurrent_jobs):
        self.max_concurrent_jobs = max_concurrent_jobs

    def update(self, param_seq: initialize.ParamSequence):
        self.param_seq = param_seq

    def run(self):
        self._run_available()
        self.ensure_finised_and_complete()

    def _run_available(self):
        for variable, params in self.param_seq:
            io.save_parameters(
                params,
                io.generate_file_path("params.toml", self.id, utils.format_variable_list(variable)),
            )
            self.new_sim(variable, params)
        self.finish()

    def new_sim(self, variable_list: List[tuple], params: dict):
        """responsible to launch a new simulation

        Parameters
        ----------
        variable_list : list[tuple]
            list of tuples (name, value) where name is the name of a
            variable parameter and value is its current value
        params : dict
            a flattened parameter dictionary, as returned by scgenerator.initialize.compute_init_parameters
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
        self.overall_pbar = tqdm(
            total=self.param_seq.num_steps, desc="Simulating", unit="step", **const.pbar_format(0)
        )

    def new_sim(self, variable_list: List[tuple], params: Dict[str, Any]):
        v_list_str = utils.format_variable_list(variable_list)
        self.logger.info(f"{self.param_seq.name} : launching simulation with {v_list_str}")
        SequentialRK4IP(
            params, self.overall_pbar, save_data=True, job_identifier=v_list_str, task_id=self.id
        ).run()

    def stop(self):
        pass

    def finish(self):
        pass


class MultiProcSimulations(Simulations, priority=1):
    @classmethod
    def is_available(cls):
        return True

    def __init__(self, param_seq: initialize.ParamSequence, task_id):
        super().__init__(param_seq, task_id=task_id)
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
            args=(self.param_seq.num_steps, self.progress_queue),
        )
        self.p_worker.start()

    def run(self):
        for worker in self.workers:
            worker.start()
        super().run()

    def new_sim(self, variable_list: List[tuple], params: dict):
        self.queue.put((variable_list, params), block=True, timeout=None)

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
            raw_data: Tuple[List[tuple], Dict[str, Any]] = queue.get()
            if raw_data == 0:
                queue.task_done()
                return
            variable_list, params = raw_data
            v_list_str = utils.format_variable_list(variable_list)
            MutliProcRK4IP(
                params,
                p_queue,
                worker_id,
                save_data=True,
                job_identifier=v_list_str,
                task_id=task_id,
            ).run()
            queue.task_done()

    # @staticmethod
    # def progress_worker(num_steps: int, progress_queue: multiprocessing.Queue):
    #     pbars: Dict[int, tqdm] = {}
    #     with tqdm(total=num_steps, desc="Simulating", unit="step", position=0) as tq:
    #         while True:
    #             raw = progress_queue.get()
    #             if raw == 0:
    #                 for pbar in pbars.values():
    #                     pbar.close()
    #                 return
    #             i, rel_pos = raw
    #             if i not in pbars:
    #                 pbars[i] = tqdm(
    #                     total=1,
    #                     desc=f"Worker {i}",
    #                     position=i,
    #                     bar_format="{l_bar}{bar}"
    #                     "|[{elapsed}<{remaining}, "
    #                     "{rate_fmt}{postfix}]",
    #                 )
    #             pbars[i].update(rel_pos - pbars[i].n)
    #             tq.update()


class RaySimulations(Simulations, priority=2):
    """runs simulation with the help of the ray module. ray must be initialized before creating an instance of RaySimulations"""

    @classmethod
    def is_available(cls):
        return using_ray and ray.is_initialized()

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

        self.propagator = ray.remote(RayRK4IP).options(
            override_environment_variables=io.get_all_environ()
        )
        self.sim_jobs_per_node = min(
            self.param_seq.num_sim, self.param_seq["simulation", "parallel"]
        )
        self.update_cluster_frequency = 3
        self.jobs = []
        self.actors = {}
        self.rolling_id = 0
        self.p_actor = ray.remote(utils.ProgressBarActor).remote(self.sim_jobs_total)
        self.p_bars = utils.PBars(
            [
                tqdm(
                    total=self.param_seq.num_steps,
                    unit="step",
                    desc="Simulating",
                    smoothing=0,
                    ncols=100,
                )
            ]
        )
        for i in range(1, self.sim_jobs_total + 1):
            self.p_bars.append(
                tqdm(
                    total=1,
                    desc=f"Worker {i}",
                    position=i,
                    ncols=100,
                    bar_format="{l_bar}{bar}" "|[{elapsed}<{remaining}, " "{rate_fmt}{postfix}]",
                )
            )

    def new_sim(self, variable_list: List[tuple], params: dict):
        while len(self.jobs) >= self.sim_jobs_total:
            self._collect_1_job()

        self.rolling_id = (self.rolling_id + 1) % self.sim_jobs_total
        v_list_str = utils.format_variable_list(variable_list)

        new_actor = self.propagator.remote(
            params,
            self.p_actor,
            self.rolling_id + 1,
            save_data=True,
            job_identifier=v_list_str,
            task_id=self.id,
        )
        new_job = new_actor.run.remote()

        self.actors[new_job.task_id()] = new_actor
        self.jobs.append(new_job)

        self.logger.info(
            f"{self.param_seq.name} : launching simulation with {v_list_str}, job : {self.jobs[-1].hex()}"
        )

    def finish(self):
        while len(self.jobs) > 0:
            self._collect_1_job()
        self.p_bars.close()

    def _collect_1_job(self):
        ready, self.jobs = ray.wait(self.jobs, timeout=self.update_cluster_frequency)
        self.update_pbars()
        if len(ready) == 0:
            return
        ray.get(ready)

        del self.actors[ready[0].task_id()]

    def stop(self):
        ray.shutdown()

    @property
    def sim_jobs_total(self):
        tot_cpus = sum([node.get("Resources", {}).get("CPU", 0) for node in ray.nodes()])
        tot_cpus = min(tot_cpus, self.max_concurrent_jobs)
        return int(min(self.param_seq.num_sim, tot_cpus))

    def update_pbars(self):
        counters = ray.get(self.p_actor.wait_for_update.remote())
        for counter, pbar in zip(counters, self.p_bars):
            pbar.update(counter - pbar.n)
        self.p_bars.print()


def run_simulation_sequence(
    *config_files: os.PathLike,
    method=None,
    final_name: str = None,
    prev_data_folder: os.PathLike = None,
):
    prev = prev_data_folder
    for config_file in config_files:
        sim = new_simulation(config_file, prev, method)
        sim.run()
        prev = sim.data_folder
    io.append_and_merge(prev, final_name)


def new_simulation(
    config_file: os.PathLike,
    prev_data_folder=None,
    method: Type[Simulations] = None,
) -> Simulations:

    config = io.load_toml(config_file)
    task_id = np.random.randint(1e9, 1e12)

    if prev_data_folder is None:
        param_seq = initialize.ParamSequence(config)
    else:
        param_seq = initialize.ContinuationParamSequence(prev_data_folder, config)

    print(f"{param_seq.name=}")

    return _new_simulations(param_seq, task_id, method)


def resume_simulations(data_folder: str, method: Type[Simulations] = None) -> Simulations:

    task_id = np.random.randint(1e9, 1e12)
    config = io.load_toml(os.path.join(data_folder, "initial_config.toml"))
    io.set_data_folder(task_id, data_folder)
    param_seq = initialize.RecoveryParamSequence(config, task_id)

    return _new_simulations(param_seq, task_id, method)


def _new_simulations(
    param_seq: initialize.ParamSequence,
    task_id,
    method: Type[Simulations],
) -> Simulations:
    if method is not None:
        if isinstance(method, str):
            method = Simulations.simulation_methods_dict[method]
        return method(param_seq, task_id)
    elif param_seq.num_sim > 1 and param_seq["simulation", "parallel"] and using_ray:
        return Simulations.get_best_method()(param_seq, task_id)
    else:
        return SequencialSimulations(param_seq, task_id)


if __name__ == "__main__":
    try:
        ray.init()
    except NameError:
        pass
    config_file, *opts = sys.argv[1:]
    new_simulation(config_file, *opts)
