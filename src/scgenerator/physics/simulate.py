import multiprocessing
import multiprocessing.connection
import os
from collections import defaultdict
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Type, Union

import numpy as np

from .. import solver, utils
from ..logger import get_logger
from ..operators import CurrentState
from ..parameter import Configuration, Parameters
from ..pbar import PBars, ProgressBarActor, progress_worker

try:
    import ray
except ModuleNotFoundError:
    ray = None


class TrackedValues(defaultdict):
    def __init__(self):
        super().__init__(list)

    def append(self, d: dict[str, Any]):
        for k, v in d.items():
            self[k].append(v)


class RK4IP:
    params: Parameters
    save_data: bool
    data_dir: Optional[Path]
    logger: Logger

    dw: float
    z_targets: list[float]
    z_store: list[float]
    z: float
    store_num: int
    error_ok: float
    size_fac: float
    cons_qty: list[float]

    init_state: CurrentState
    stored_spectra: list[np.ndarray]

    def __init__(
        self,
        params: Parameters,
        save_data=False,
    ):
        """A 1D solver using 4th order Runge-Kutta in the interaction picture

        Parameters
        ----------
        params : Parameters
            parameters of the simulation
        save_data : bool, optional
            save calculated spectra to disk, by default False
        """

        self.params = params
        self.save_data = save_data

        if self.save_data:
            self.data_dir = params.output_path
            os.makedirs(self.data_dir, exist_ok=True)
        else:
            self.data_dir = None

        self.logger = get_logger(self.params.output_path.name)

        self.error_ok = (
            params.tolerated_error if self.params.adapt_step_size else self.params.step_size
        )

        # setup save targets
        self.z_targets = self.params.z_targets
        self.z_stored = list(self.z_targets.copy()[0 : self.params.recovery_last_stored + 1])
        self.z_targets = list(self.z_targets.copy()[self.params.recovery_last_stored :])
        self.z_targets.sort()
        self.store_num = len(self.z_targets)

        # Initial step size
        if self.params.adapt_step_size:
            initial_h = (self.z_targets[1] - self.z_targets[0]) / 2
        else:
            initial_h = self.error_ok
        self.init_state = CurrentState(
            length=self.params.length,
            z=self.z_targets.pop(0),
            current_step_size=initial_h,
            step=0,
            C_to_A_factor=self.params.c_to_a_factor,
            converter=self.params.ifft,
            spectrum=self.params.spec_0.copy() / self.params.c_to_a_factor,
        )
        self.stored_spectra = self.params.recovery_last_stored * [None] + [
            self.init_state.spectrum.copy()
        ]
        self.tracked_values = TrackedValues()

    def _save_current_spectrum(self, spectrum: np.ndarray, num: int):
        """saves the spectrum and the corresponding cons_qty array

        Parameters
        ----------
        num : int
            index of the z postition
        """
        self.write(spectrum, f"spectrum_{num}")
        self.write(self.tracked_values, "tracked_values")

    def write(self, data: np.ndarray, name: str):
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
        state = self.init_state
        for num, state in self.irun():
            if self.save_data:
                self._save_current_spectrum(state.actual_spectrum, num)
                self.step_saved(state)

        self.logger.info(
            "propagation finished in {} steps ({} seconds)".format(
                state.step, (datetime.today() - time_start).total_seconds()
            )
        )

        if self.save_data:
            self.write(self.z_stored, "z.npy")

        return self.stored_spectra

    def irun(self) -> Iterator[tuple[int, CurrentState]]:
        """run the simulation as a generator obj

        Yields
        -------
        int
            current number of spectra returned
        CurrentState
            current simulation state
        """

        self.logger.debug(
            "Computing {} new spectra, first one at {}m".format(self.store_num, self.z_targets[0])
        )
        store = False
        state = self.init_state.copy()
        yield len(self.stored_spectra) - 1, state

        integrator_args = [
            self.params.compute(a) for a in solver.Integrator.factory_args() if a != "init_state"
        ]
        integrator = solver.Integrator.create(
            self.params.integration_scheme, state, *integrator_args
        )

        for state in integrator:

            new_tracked_values = integrator.all_values()
            self.logger.debug(f"tracked values at z={state.z} : {new_tracked_values}")
            self.tracked_values.append(new_tracked_values)

            # Whether the current spectrum has to be stored depends on previous step
            if store:
                current_spec = state.actual_spectrum
                self.stored_spectra.append(current_spec)

                yield len(self.stored_spectra) - 1, state.copy()

                self.z_stored.append(state.z)
                del self.z_targets[0]

                # reset the constant step size after a spectrum is stored
                if not self.params.adapt_step_size:
                    integrator.state.current_step_size = self.error_ok

                if len(self.z_targets) == 0:
                    break
                store = False

            # if the next step goes over a position at which we want to store
            # a spectrum, we shorten the step to reach this position exactly
            if state.z + integrator.state.current_step_size >= self.z_targets[0]:
                store = True
                integrator.state.current_step_size = self.z_targets[0] - state.z

    def step_saved(self, state: CurrentState):
        pass

    def __iter__(self) -> Iterator[tuple[int, CurrentState]]:
        yield from self.irun()

    def __len__(self) -> int:
        return self.params.z_num


class SequentialRK4IP(RK4IP):
    def __init__(
        self,
        params: Parameters,
        pbars: PBars,
        save_data=False,
    ):
        self.pbars = pbars
        super().__init__(
            params,
            save_data=save_data,
        )

    def step_saved(self, state: CurrentState):
        self.pbars.update(1, state.z / self.params.length - self.pbars[1].n)


class MutliProcRK4IP(RK4IP):
    def __init__(
        self,
        params: Parameters,
        p_queue: multiprocessing.Queue,
        worker_id: int,
        save_data=False,
    ):
        self.worker_id = worker_id
        self.p_queue = p_queue
        super().__init__(
            params,
            save_data=save_data,
        )

    def step_saved(self, state: CurrentState):
        self.p_queue.put((self.worker_id, state.z / self.params.length))


class RayRK4IP(RK4IP):
    def __init__(self):
        pass

    def set(
        self,
        params: Parameters,
        p_actor,
        worker_id: int,
        save_data=False,
    ):
        self.worker_id = worker_id
        self.p_actor = p_actor
        super().__init__(
            params,
            save_data=save_data,
        )

    def set_and_run(self, v):
        params, p_actor, worker_id, save_data = v
        self.set(params, p_actor, worker_id, save_data)
        self.run()

    def step_saved(self, state: CurrentState):
        self.p_actor.update.remote(self.worker_id, state.z / self.params.length)
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
        cls, configuration: Configuration, method: Union[str, Type["Simulations"]] = None
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
            return method(configuration)
        elif configuration.num_sim > 1 and configuration.parallel:
            return Simulations.get_best_method()(configuration)
        else:
            return SequencialSimulations(configuration)

    def __init__(self, configuration: Configuration):
        """
        Parameters
        ----------
        configuration : scgenerator.Configuration obj
            parameter sequence
        data_folder : str, optional
            path to the folder where data is saved, by default "scgenerator/"
        """
        if not self.is_available():
            raise RuntimeError(f"{self.__class__} is currently not available")
        self.logger = get_logger(__name__)

        self.configuration = configuration

        self.name = self.configuration.name
        self.sim_dir = self.configuration.final_path
        self.configuration.save_parameters()

        self.sim_jobs_per_node = 1

    def finished_and_complete(self):
        # for sim in self.configuration.all_configs.values():
        #     if (
        #         self.configuration.sim_status(sim.output_path)[0]
        #         != self.configuration.State.COMPLETE
        #     ):
        #         return False
        return True

    def run(self):
        self._run_available()
        self.ensure_finised_and_complete()

    def _run_available(self):
        for _, params in self.configuration:
            utils.save_parameters(params.dump_dict(), params.output_path)

            self.new_sim(params)
        self.finish()

    def new_sim(self, params: Parameters):
        """responsible to launch a new simulation

        Parameters
        ----------
        params : Parameters
            computed parameters
        """
        raise NotImplementedError()

    def finish(self):
        """called once all the simulations are launched."""
        raise NotImplementedError()

    def ensure_finised_and_complete(self):
        while not self.finished_and_complete():
            self.logger.warning("Something wrong happened, running again to finish simulation")
            self._run_available()

    def stop(self):
        raise NotImplementedError()


class SequencialSimulations(Simulations, priority=0):
    @classmethod
    def is_available(cls):
        return True

    def __init__(self, configuration: Configuration):
        super().__init__(configuration)
        self.pbars = PBars(
            self.configuration.total_num_steps,
            "Simulating " + self.configuration.final_path.name,
            1,
        )
        self.configuration.skip_callback = lambda num: self.pbars.update(0, num)

    def new_sim(self, params: Parameters):
        self.logger.info(f"{self.configuration.final_path} : launching simulation")
        SequentialRK4IP(params, self.pbars, save_data=True).run()

    def stop(self):
        pass

    def finish(self):
        self.pbars.close()


class MultiProcSimulations(Simulations, priority=1):
    @classmethod
    def is_available(cls):
        return True

    def __init__(self, configuration: Configuration):
        super().__init__(configuration)
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
                args=(i + 1, self.queue, self.progress_queue),
            )
            for i in range(self.sim_jobs_per_node)
        ]
        self.p_worker = multiprocessing.Process(
            target=progress_worker,
            args=(
                Path(self.configuration.final_path).name,
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

    def new_sim(self, params: Parameters):
        self.queue.put(params.dump_dict(), block=True, timeout=None)

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
        worker_id: int,
        queue: multiprocessing.JoinableQueue,
        p_queue: multiprocessing.Queue,
    ):
        while True:
            raw_data: tuple[list[tuple], Parameters] = queue.get()
            if raw_data == 0:
                queue.task_done()
                return
            params = Parameters(**raw_data)
            MutliProcRK4IP(
                params,
                p_queue,
                worker_id,
                save_data=True,
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
    ):
        super().__init__(configuration)

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
        self.p_actor = ray.remote(ProgressBarActor).remote(
            self.configuration.final_path.name,
            self.sim_jobs_total,
            self.configuration.total_num_steps,
        )
        self.configuration.skip_callback = lambda num: ray.get(self.p_actor.update.remote(0, num))

    def new_sim(self, params: Parameters):
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
            ),
        )
        self.num_submitted += 1

        self.logger.info(f"{self.configuration.final_path} : launching simulation")

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
    config = Configuration(config_file, wait=True)

    sim = new_simulation(config, method)
    sim.run()

    for path in config.fiber_paths:
        utils.combine_simulations(path)


def new_simulation(
    configuration: Configuration,
    method: Union[str, Type[Simulations]] = None,
) -> Simulations:
    logger = get_logger(__name__)
    logger.info(f"running {configuration.final_path}")
    return Simulations.new(configuration, method)


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
