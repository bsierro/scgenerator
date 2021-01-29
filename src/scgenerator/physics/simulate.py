import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
from numpy.fft import fft, ifft

from .. import initialize, io, utils
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

        self.job_identifier = job_identifier
        self.id = task_id
        self.n_percent = n_percent
        self.logger = get_logger(self.job_identifier)

        self.resuming = False
        self.save_data = save_data
        self._extract_params(sim_params)
        self._setup_functions()
        self.starting_num = sim_params.get("recovery_last_store", 1) - 1
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
                self.logger.info("energy conserved")
                self.conserved_quantity_func = pulse.pulse_energy
        else:
            self.conserved_quantity_func = lambda a, b, c, d: 0

    def _setup_sim_parameters(self):
        # making sure to keep only the z that we want
        self.z_targets = list(self.z_targets.copy()[self.starting_num :])
        self.z_targets.sort()
        self.store_num = len(self.z_targets)

        # Initial setup of simulation parameters
        self.d_w = self.w_c[1] - self.w_c[0]  # resolution of the frequency grid
        self.z = self.z_targets.pop(0)
        self.z_stored = [self.z]  # position of each stored spectrum (for display)

        self.progress_tracker = utils.ProgressTracker(
            self.z_final, percent_incr=self.n_percent, logger=self.logger
        )

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
            _save_current_spectrum(
                self.current_spectrum, self.cons_qty, 0, self.id, self.job_identifier
            )

        # Initial step size
        if self.adapt_step_size:
            self.initial_h = (self.z_targets[0] - self.z) / 2
        else:
            self.initial_h = self.error_ok

    # def _setup_sim_parameters(self):
    #     # making sure to keep only the z that we want
    #     self.z_targets = list(self.z_targets.copy())
    #     self.z_targets.sort()
    #     self.store_num = len(self.z_targets)

    #     # Initial setup of simulation parameters
    #     self.d_w = self.w_c[1] - self.w_c[0]  # resolution of the frequency grid
    #     self.z = self.z_targets.pop(0)
    #     self.z_stored = [self.z]  # position of each stored spectrum (for display)

    #     self.progress_tracker = utils.ProgressTracker(
    #         self.z_final, percent_incr=self.n_percent, logger=self.logger
    #     )

    #     # Setup initial values for every physical quantity that we want to track
    #     self.current_spectrum = self.spec_0.copy()
    #     self.stored_spectra = [self.current_spectrum.copy()]
    #     self.cons_qty = [
    #         self.conserved_quantity_func(
    #             self.current_spectrum,
    #             self.w_c + self.w0,
    #             self.d_w,
    #             self.gamma,
    #         ),
    #         0,
    #     ]
    #     self.size_fac = 2 ** (1 / 5)

    #     if self.save_data:
    #         _save_current_spectrum(
    #             self.current_spectrum, self.cons_qty, 0, self.id, self.job_identifier
    #         )

    #     # Initial step size
    #     if self.adapt_step_size:
    #         self.initial_h = (self.z_targets[0] - self.z) / 2
    #     else:
    #         self.initial_h = self.error_ok

    def run(self):
        # Print introduction
        self.logger.info(
            "Computing {} new spectra, first one at {}m".format(self.store_num, self.z_targets[0])
        )
        self.progress_tracker.set(self.z)

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
                self.progress_tracker.suffix = " ({} steps). z = {:.4f}, h = {:.5g}".format(
                    step, self.z, h_taken
                )
                self.progress_tracker.set(self.z)

                self.stored_spectra.append(self.current_spectrum)
                if self.save_data:
                    _save_current_spectrum(
                        self.current_spectrum,
                        self.cons_qty,
                        len(self.stored_spectra) - 1,
                        self.id,
                        self.job_identifier,
                    )

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
            io.save_data(self.z_stored, "z.npy", self.id, self.job_identifier)

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
                    self.logger.info(progress_str)
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


class Simulations:
    """The recommended way to run simulations.
    New Simulations child classes can be written and must implement the following
    """

    _available_simulation_methods = []

    def __init_subclass__(cls, available: bool, priority=0, **kwargs):
        cls._available = available
        if available:
            Simulations._available_simulation_methods.append((cls, priority))
        Simulations._available_simulation_methods.sort(key=lambda el: el[1])
        super().__init_subclass__(**kwargs)

    @classmethod
    def get_best_method(cls):
        return Simulations._available_simulation_methods[-1][0]

    def __init__(self, param_seq: initialize.ParamSequence, task_id=0, data_folder="scgenerator/"):
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

        self.param_seq = param_seq
        self.name = param_seq.name
        self.data_folder = io.get_data_folder(self.id, name_if_new=self.name)
        io.save_toml(os.path.join(self.data_folder, "initial_config.toml"), self.param_seq.config)

        self.using_ray = False
        self.sim_jobs = 1

        self.propagator = RK4IP

        self.progress_tracker = utils.ProgressTracker(
            len(self.param_seq), percent_incr=1, logger=self.logger
        )

    def run(self):
        for varying_params, params in self.param_seq:
            for i in range(self.param_seq["simulation", "repeat"]):
                varying = varying_params + [("num", i)]
                io.save_parameters(
                    params,
                    io.generate_file_path(
                        "params.toml", self.id, utils.format_varying_list(varying)
                    ),
                )
                self.new_sim(varying, params.copy())

        self.finish()
        self.logger.info(f"Merging data...")

        self.merge_data()
        self.logger.info(f"Finished simulations from config {self.name} !")

    def new_sim(self, varying_list: List[tuple], params: dict):
        """responsible to launch a new simulation

        Parameters
        ----------
        varying_list : list[tuple]
            list of tuples (name, value) where name is the name of a
            varying parameter and value is its current value
        params : dict
            a flattened parameter dictionary, as returned by scgenerator.initialize.compute_init_parameters
        """
        raise NotImplementedError()

    def finish(self):
        """called once all the simulations are launched."""
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def merge_data(self):
        io.merge_same_simulations(self.data_folder)


class SequencialSimulations(Simulations, available=True, priority=0):
    def new_sim(self, varying_list: List[tuple], params: dict):
        v_list_str = utils.format_varying_list(varying_list)
        self.logger.info(f"launching simulation with {v_list_str}")
        self.propagator(
            params,
            save_data=True,
            job_identifier=v_list_str,
            task_id=self.id,
        ).run()
        self.progress_tracker.update()

    def finish(self):
        pass

    def stop(self):
        pass


class RaySimulations(Simulations, available=using_ray, priority=1):
    """runs simulation with the help of the ray module. ray must be initialized before creating an instance of RaySimulations"""

    def __init__(self, param_seq: initialize.ParamSequence, task_id=0, data_folder="scgenerator/"):
        super().__init__(param_seq, task_id, data_folder)
        self._init_ray()

    def _init_ray(self):
        nodes = ray.nodes()
        nodes_num = len(nodes)
        self.logger.info(
            f"{nodes_num} node{'s' if nodes_num > 1 else ''} in the Ray cluster : "
            + str([node.get("NodeManagerHostname", "unknown") for node in nodes])
        )

        self.sim_jobs = min(self.param_seq.num_sim, self.param_seq["simulation", "parallel"])
        self.propagator = ray.remote(self.propagator).options(
            override_environment_variables=io.get_all_environ()
        )

        self.jobs = []
        self.actors = {}

    def new_sim(self, varying_list: List[tuple], params: dict):
        while len(self.jobs) >= self.sim_jobs:
            self._collect_1_job()

        v_list_str = utils.format_varying_list(varying_list)

        new_actor = self.propagator.remote(
            params, save_data=True, job_identifier=v_list_str, task_id=self.id
        )
        new_job = new_actor.run.remote()

        self.actors[new_job.task_id()] = new_actor
        self.jobs.append(new_job)

        self.logger.info(f"launching simulation with {v_list_str}, job : {self.jobs[-1].hex()}")

    def finish(self):
        while len(self.jobs) > 0:
            self._collect_1_job()

    def _collect_1_job(self):
        ready, self.jobs = ray.wait(self.jobs)
        ray.get(ready)
        del self.actors[ready[0].task_id()]
        self.progress_tracker.update()

    def stop(self):
        ray.shutdown()


def new_simulations(config_file: str, task_id: int, data_folder="scgenerator/"):

    config = io.load_toml(config_file)
    param_seq = initialize.ParamSequence(config)

    return _new_simulations(param_seq, task_id, data_folder)


def resume_simulations(data_folder: str, task_id: int = 0):

    config = io.load_toml(os.path.join(data_folder, "initial_config.toml"))
    io.set_data_folder(task_id, data_folder)
    param_seq = initialize.RecoveryParamSequence(config, task_id)

    return _new_simulations(param_seq, task_id, data_folder)


def _new_simulations(param_seq: initialize.ParamSequence, task_id, data_folder):
    if param_seq.num_sim > 1 and param_seq["simulation", "parallel"] > 1 and using_ray:
        return Simulations.get_best_method()(param_seq, task_id, data_folder=data_folder)
    else:
        return SequencialSimulations(param_seq, task_id, data_folder=data_folder)


def RK4IP_func(sim_params, save_data=False, job_identifier="", task_id=0, n_percent=10):
    """Computes the spectrum of a pulse as it propagates through a PCF

    Parameters
    ----------
        sim_params : a dictionary containing the following :
            w_c : array
                angular frequencies centered around 0 generated with scgenerator.initialize.wspace
            w0 : float
                central angular frequency of the pulse
            t : array
                time
            dt : float
                time resolution
            spec_0 : array
                initial spectral envelope as function of w_c
            z_targets : list
                target distances
            beta : array
                beta coeficients (Taylor expansion of beta(w))
            gamma : float
                non-linear parameter
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
        save_data : bool
            False : return the spectra (recommended, save manually later if necessary)
            True : save in a temporary folder and return the folder name
                   to be used for merging later
        job_id : int
            id of this particular simulation
        param_id : int
            id corresponding to the set of paramters. Files created with the same param_id will be
            merged if an indexer is passed (this feature is mainly used for automated parallel simulations
            using the parallel_simulations function).
        task_id : int
            id of the whole program (useful when many python instances run at once). None if not running in parallel
        n_percent : int, float
            log message every n_percent of the simulation done
        pt : scgenerator.progresstracker.ProgressTracker object
        indexer : indexer object
        debug_return : bool
            if True and save_data False, will return photon number and step sizes as well as the spectra.
    Returns
    ----------
        stored_spectra : (z_num, nt) array
            spectrum aligned on w_c array
        h_stored : 1D array
            length of each valid step
        cons_qty : 1D array
            conserved quantity at each valid step

    """
    # DEBUG
    debug = False

    w_c = sim_params.pop("w_c")
    w0 = sim_params.pop("w0")
    w_power_fact = sim_params.pop("w_power_fact")
    spec_0 = sim_params.pop("spec_0")
    z_targets = sim_params.pop("z_targets")
    z_final = sim_params.pop("length")
    beta = sim_params.pop("beta_func", sim_params.pop("beta"))
    gamma = sim_params.pop("gamma_func", sim_params.pop("gamma"))
    behaviors = sim_params.pop("behaviors")
    raman_type = sim_params.pop("raman_type", "stolen")
    f_r = sim_params.pop("f_r", 0)
    hr_w = sim_params.pop("hr_w", None)
    adapt_step_size = sim_params.pop("adapt_step_size", True)
    error_ok = sim_params.pop("error_ok")
    dynamic_dispersion = sim_params.pop("dynamic_dispersion", False)
    del sim_params

    logger = get_logger(job_identifier)

    # Initial setup of both non linear and linear operators
    N_func = create_non_linear_op(behaviors, w_c, w0, gamma, raman_type, f_r, hr_w)
    if dynamic_dispersion:
        disp = lambda r: fast_dispersion_op(w_c, beta(r), w_power_fact)
    else:
        disp = lambda r: fast_dispersion_op(w_c, beta, w_power_fact)

    # Set up which quantity is conserved for adaptive step size
    if adapt_step_size:
        if "raman" in behaviors:
            conserved_quantity_func = pulse.photon_number
        else:
            print("energy conserved")
            conserved_quantity_func = pulse.pulse_energy
    else:
        conserved_quantity_func = lambda a, b, c, d: 0

    # making sure to keep only the z that we want
    z_targets = list(z_targets.copy())
    z_targets.sort()
    store_num = len(z_targets)

    # Initial setup of simulation parameters
    d_w = w_c[1] - w_c[0]  # resolution of the frequency grid
    z = z_targets.pop(0)
    z_stored = [z]  # position of each stored spectrum (for display)

    pt = utils.ProgressTracker(z_final, percent_incr=n_percent, logger=logger)

    # Setup initial values for every physical quantity that we want to track
    current_spectrum = spec_0.copy()
    stored_spectra = [current_spectrum.copy()]
    stored_field = [ifft(current_spectrum.copy())]
    cons_qty = [conserved_quantity_func(current_spectrum, w_c + w0, d_w, gamma), 0]
    size_fac = 2 ** (1 / 5)

    if save_data:
        _save_current_spectrum(current_spectrum, cons_qty, 0, task_id, job_identifier)

    # Initial step size
    if adapt_step_size:
        h = (z_targets[0] - z) / 2
    else:
        h = error_ok
    newh = h

    # Print introduction
    logger.info("Computing {} new spectra, first one at {}m".format(store_num, z_targets[0]))
    pt.set(z)

    # Start of the integration
    step = 1
    keep = True  # keep a step
    store = False  # store a spectrum
    time_start = datetime.today()

    while z < z_final:
        h = newh
        z_ratio = z / z_final

        # Store Exp(h/2 * disp) to be used several times
        expD = np.exp(h / 2 * disp(z_ratio))

        # RK4 algorithm
        A_I = expD * current_spectrum
        k1 = expD * (h * N_func(current_spectrum, z_ratio))
        k2 = h * N_func(A_I + k1 / 2, z_ratio)
        k3 = h * N_func(A_I + k2 / 2, z_ratio)
        k4 = h * N_func(expD * (A_I + k3), z_ratio)

        end_spectrum = expD * (A_I + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6

        # Check relative error and adjust next step size
        if adapt_step_size:
            cons_qty[step] = conserved_quantity_func(end_spectrum, w_c + w0, d_w, gamma)
            curr_p_change = np.abs(cons_qty[step - 1] - cons_qty[step])
            cons_qty_change_ok = error_ok * cons_qty[step - 1]

            if curr_p_change > 2 * cons_qty_change_ok:
                keep = False
                newh = h / 2
            elif cons_qty_change_ok < curr_p_change <= 2 * cons_qty_change_ok:
                keep = True
                newh = h / size_fac
            elif curr_p_change < 0.1 * cons_qty_change_ok:
                keep = True
                newh = h * size_fac
            else:
                keep = True
                newh = h

        # consider storing anythin only if the step was valid
        if keep:

            # If step is accepted, z becomes the current position
            z += h
            step += 1
            cons_qty.append(0)

            current_spectrum = end_spectrum.copy()

            # Whether the current spectrum has to be stored depends on previous step
            if store:
                pt.suffix = " ({} steps). z = {:.4f}, h = {:.5g}".format(step, z, h)
                pt.set(z)

                stored_spectra.append(end_spectrum)
                stored_field.append(ifft(end_spectrum))
                if save_data:
                    _save_current_spectrum(
                        end_spectrum, cons_qty, len(stored_spectra) - 1, task_id, job_identifier
                    )

                z_stored.append(z)
                del z_targets[0]

                # No more spectrum to store
                if len(z_targets) == 0:
                    break
                store = False

                # reset the constant step size after a spectrum is stored
                if not adapt_step_size:
                    newh = error_ok

            # if the next step goes over a position at which we want to store
            # a spectrum, we shorten the step to reach this position exactly
            if z + newh >= z_targets[0]:
                store = True
                newh = z_targets[0] - z
        else:
            progress_str = f"step {step} rejected with h = {h:.4e}, doing over"
            logger.info(progress_str)

    logger.info(
        "propagation finished in {} steps ({} seconds)".format(
            step, (datetime.today() - time_start).total_seconds()
        )
    )

    if save_data:
        io.save_data(z_stored, "z.npy", task_id, job_identifier)

    return stored_spectra


def _save_current_spectrum(
    spectrum: np.ndarray, cons_qty: np.ndarray, num: int, task_id: int, job_identifier: str
):
    """saves the spectrum and the corresponding cons_qty array

    Parameters
    ----------
    spectrum : np.ndarray
        spectrum as function of w
    cons_qty : np.ndarray
        cons_qty array
    num : int
        index of the z postition
    task_id : int
        unique number identifyin the session
    job_identifier : str
        to differentiate this particular run from the others in the session
    """
    io.save_data(spectrum, f"spectrum_{num}", task_id, job_identifier)
    io.save_data(cons_qty, f"cons_qty", task_id, job_identifier)
