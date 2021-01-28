import json
import os
from datetime import datetime
from typing import List

import numpy as np
from numpy.fft import fft, ifft


from .. import initialize
from .. import io, state
from .. import utilities
from ..io import generate_file_path, get_logger
from ..math import abs2
from ..utilities import ProgressTracker, format_varying_list
from . import pulse, units
from .fiber import create_non_linear_op, fast_dispersion_op

using_ray = False
try:
    import ray

    using_ray = True
except ModuleNotFoundError:
    pass


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

        self.propagation_func = lambda params, varying_list: RK4IP(
            params,
            save_data=True,
            job_identifier=utilities.format_varying_list(varying_list),
            task_id=self.id,
        )

        self.progress_tracker = utilities.ProgressTracker(
            max=len(self.param_seq),
            auto_print=True,
            percent_incr=1,
            callback=lambda s, logger: logger.info(s),
        )

    def run(self):
        for varying_params, params in self.param_seq:
            for i in range(self.param_seq["simulation", "repeat"]):
                varying = varying_params + [("num", i)]
                io.save_parameters(
                    params,
                    io.generate_file_path(
                        "params.toml", self.id, utilities.format_varying_list(varying)
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
        io.merge_data(self.data_folder)


class SequencialSimulations(Simulations, available=True, priority=0):
    def new_sim(self, varying_list: List[tuple], params: dict):
        self.logger.info(f"launching simulation with {varying_list}")
        self.propagation_func(params, varying_list)
        self.progress_tracker.update(1, [self.logger])

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
        self.propagation_func = ray.remote(self.propagation_func).options(
            override_environment_variables=io.get_all_environ()
        )
        self.jobs = []

    def new_sim(self, varying_list: List[tuple], params: dict):
        if len(self.jobs) >= self.sim_jobs:

            # wait for a slot to free before starting a new job
            _, self.jobs = ray.wait(self.jobs)
            ray.get(_)
            self.progress_tracker.update(1, [self.logger])

        self.jobs.append(self.propagation_func.remote(params, varying_list))

        self.logger.info(f"launching simulation with {varying_list}, job : {self.jobs[-1].hex()}")

    def finish(self):
        for job in self.jobs:
            ray.get(job)
            self.progress_tracker.update(1, [self.logger])

    def stop(self):
        ray.shutdown()


def new_simulations(config_file: str, task_id: int, data_folder="scgenerator/"):

    config = io.load_toml(config_file)
    param_seq = initialize.ParamSequence(config)

    if param_seq.num_sim > 1 and param_seq["simulation", "parallel"] > 1 and using_ray:
        return Simulations.get_best_method()(param_seq, task_id, data_folder=data_folder)
    else:
        return SequencialSimulations(param_seq, task_id, data_folder=data_folder)


def RK4IP(sim_params, save_data=False, job_identifier="", task_id=0, n_percent=10):
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
            field_0 : array
                initial field envelope as function of w_c
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
        stored_spectra : (store_num, nt) array
            spectrum aligned on w_c array
        h_stored : 1D array
            length of each valid step
        cons_qty : 1D array
            conserved quantity at each valid step
        cons_qty_change : 1D array
            conserved quantity change at each valid step

    """
    # DEBUG
    debug = False

    w_c = sim_params.pop("w_c")
    w0 = sim_params.pop("w0")
    w_power_fact = sim_params.pop("w_power_fact")
    field_0 = sim_params.pop("field_0")
    z_targets = sim_params.pop("z_targets")
    z_final = sim_params.pop("length")
    beta = sim_params.pop("beta_func", sim_params.pop("beta"))
    gamma = sim_params.pop("gamma_func", sim_params.pop("gamma"))
    behaviors = sim_params.pop("behaviors")
    raman_type = sim_params.pop("raman_type", "stolen")
    f_r = sim_params.pop("f_r", 0)
    hr_w = sim_params.pop("hr_w", None)
    adapt_step_size = sim_params.pop("adapt_step_size", True)
    error_ok = sim_params.pop("error_ok", 1e-10)
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
    z_targets = list(set(value for value in z_targets if value > 0))
    z_targets.sort()
    store_num = len(z_targets)

    # Initial setup of simulation parameters
    d_w = w_c[1] - w_c[0]  # resolution of the frequency grid
    z_stored, z = [0], 0  # position of each stored spectrum (for display)

    pt = utilities.ProgressTracker(
        z_final,
        auto_print=True,
        percent_incr=n_percent,
        callback=_gen_RK4IP_progress_callback(),
    )

    # Setup initial values for every physical quantity that we want to track
    current_spectrum = fft(field_0)
    stored_spectra = [current_spectrum.copy()]
    stored_field = [ifft(current_spectrum.copy())]
    cons_qty = [conserved_quantity_func(current_spectrum, w_c + w0, d_w, gamma), 0]
    cons_qty_change = [0, 0]
    size_fac = 2 ** (1 / 5)

    if save_data:
        _save_current_spectrum(current_spectrum, 0, task_id, job_identifier)

    # Initial step size
    if adapt_step_size:
        h = z_targets[0] / 2
    else:
        h = error_ok
    newh = h

    # Print introduction
    logger.info("Storing {} new spectra, first one at {}m".format(store_num, z_targets[0]))

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
            cons_qty_change[step] = cons_qty_change[step - 1] + curr_p_change
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
            cons_qty_change.append(0)

            current_spectrum = end_spectrum.copy()

            # Whether the current spectrum has to be stored depends on previous step
            if store:
                pt.set(z, [logger, step, z, h])

                stored_spectra.append(end_spectrum)
                stored_field.append(ifft(end_spectrum))
                if save_data:
                    _save_current_spectrum(
                        end_spectrum, len(stored_spectra) - 1, task_id, job_identifier
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


def _save_current_spectrum(spectrum: np.ndarray, num: int, task_id: int, job_identifier: str):
    base_name = f"spectrum_{num}.npy"
    io.save_data(spectrum, base_name, task_id, job_identifier)


def _gen_RK4IP_progress_callback():
    def callback(s, logger, step, z, h):
        progress_str = " ({} steps). z = {:.4f}, h = {:.5g}".format(step, z, h)
        logger.info(s + progress_str)

    return callback


def _RK4IP_extract_params(sim_params):
    """extracts the right parameters from the the flattened params dict

    Parameters
    ----------
    sim_params : dict
        flattened parameters dictionary

    Returns
    -------
    tuple
        all the necessary parameters
    """
    w_c = sim_params.pop("w_c")
    w0 = sim_params.pop("w0")
    w_power_fact = sim_params.pop("w_power_fact")
    field_0 = sim_params.pop("field_0")
    z_targets = sim_params.pop("z_targets")
    beta = sim_params.pop("beta_func", sim_params.pop("beta"))
    gamma = sim_params.pop("gamma_func", sim_params.pop("gamma"))
    behaviors = sim_params.pop("behaviors")
    raman_type = sim_params.pop("raman_type", "stolen")
    f_r = sim_params.pop("f_r", 0)
    hr_w = sim_params.pop("hr_w", None)
    adapt_step_size = sim_params.pop("adapt_step_size", True)
    error_ok = sim_params.pop("error_ok", 1e-10)
    dynamic_dispersion = sim_params.pop("dynamic_dispersion", False)
    del sim_params
    return (
        behaviors,
        w_c,
        w0,
        gamma,
        raman_type,
        f_r,
        hr_w,
        dynamic_dispersion,
        beta,
        w_power_fact,
        adapt_step_size,
        z_targets,
        field_0,
        error_ok,
    )


def _prepare_grid(z_targets, w_c):
    """prepares some derived values for the propagation

    Parameters
    ----------
    z_targets : array
        array of target z positions
    w_c : array
        angular frequency array (centered on 0)

    Returns
    -------
    d_w : float
        angular frequency grid size
    z_targets : list
        list of target z positions
    store_num : int
        number of spectra to store
    z_final : float
        final z position
    z_sored : list
        where the spectra are already stored

    """
    # making sure to keep only the z that we want
    z_targets = list(set(value for value in z_targets if value > 0))
    z_targets.sort()
    z_final = z_targets[-1]
    store_num = len(z_targets)

    # Initial setup of simulation parameters
    d_w = w_c[1] - w_c[0]  # resolution of the frequency grid
    z_stored = [0]  # position of each stored spectrum (for display)
    return d_w, z_targets, store_num, z_final, z_stored


def parallel_simulations(config_file, num_cpu_per_task=1, task_id=0):
    """runs simulations in parallel thanks to Ray
    Parameters
    ----------
        config_file : str
            name of the config file
            should be a json containing all necessary parameters for the simulation. Varying parameters should be placed in a subdictionary
            called "varying" (see scgenerator.utilities.dictionary_iterator for details)
        num_cpu_per_task : int
            number of concurrent job per node
        task_id : give an id for book keeping purposes (must be set if multiple ray instances run at once so their files do not overlap)

    Returns
    ----------
        name of the folder where the data is stored
    """
    logger = ray.remote(io.Logger).remote()
    state.CurrentLogger.focus_logger(logger)

    print("Nodes in the Ray cluster:", len(ray.nodes()))
    for node in ray.nodes():
        print("    " + node.get("NodeManagerHostname", "unknown"))

    config_name, config_dict, store_num, n, m = _sim_preps(config_file)

    # Override number of simultaneous jobs if provided by config file
    sim_jobs = config_dict.pop("sim_jobs", len(ray.nodes()) * num_cpu_per_task)
    print(f"number of simultaneous jobs : {sim_jobs}")

    if n * m < sim_jobs:
        sim_jobs = n * m

    # Initiate helper workers (a logger, a progress tracker to give estimates of
    # completion time and an indexer to keep track of the individual files
    # created after each simulation. The indexer can then automatically merge them)
    pt = ray.remote(ProgressTracker).remote(max=n * m * store_num, auto_print=True, percent_incr=1)
    indexer = ray.remote(io.tmp_index_manager).remote(
        config_name=config_name, task_id=task_id, varying_keys=config_dict.get("varying", None)
    )
    ray.get(
        logger.log.remote(f"CRITICAL FILE at {ray.get(indexer.get_path.remote())}, do not touch it")
    )
    RK4IP_parallel = ray.remote(RK4IP)

    jobs = []

    # we treat loops over different parameters differently
    for k, dico in enumerate(utilities.dictionary_iterator(config_dict, varying_dict="varying")):
        # loop over same parameter set
        for i in range(n):
            # because of random processes, initial conditions are recalculated every time
            params = initialize.compute_init_parameters(dictionary=config_dict, replace=dico)

            # make sure initial conditions are saved
            params["init_P0"] = dico.get("P0", config_dict.get("P0", 0))
            params["init_T0_FWHM"] = dico.get("T0_FWHM", config_dict.get("T0_FWHM", 0))
            params["param_id"] = k
            params_file_name = io.generate_file_path("param", i, k, task_id, "")
            io.save_parameters(params, params_file_name)
            ray.get(indexer.append_to_index.remote(k, params_file_name=params_file_name))

            if len(jobs) >= sim_jobs:
                # update the number of jobs if new nodes connect
                sim_jobs = min(n * (m - k) - i, len(ray.nodes()) * num_cpu_per_task)

                # print(f"Nodes in the Ray cluster: {len(ray.nodes())}, {sim_jobs} simultaneous jobs")
                # for node in ray.nodes():
                #     print("    " + node.get("NodeManagerHostname", "unknown"))

                # wait for a slot to free before starting a new job
                _, jobs = ray.wait(jobs)
                ray.get(_)

            # start a new simulation
            ray.get(
                logger.log.remote(
                    f"Launching propagation of a {params.get('t0', 0) * 1e15:.2f}fs pulse with {np.max(abs2(params['field_0'])):.0f}W peak power over {np.max(params['z_targets'])}m"
                )
            )
            jobs.append(
                RK4IP_parallel.remote(
                    params,
                    save_data=True,
                    job_id=i,
                    param_id=k,
                    task_id=task_id,
                    pt=pt,
                    indexer=indexer,
                    logger=logger,
                    n_percent=1,
                )
            )

            ray.get(logger.log.remote("number of running jobs : {}".format(len(jobs))))
            ray.get(logger.log.remote(ray.get(pt.get_eta.remote())))

    # wait for the last jobs to finish
    ray.get(jobs)

    # merge the data properly
    folder_0 = ray.get(indexer.convert_sim_data.remote())

    print(f"{config_name} successfully finished ! data saved in {folder_0}")

    return folder_0


def simulate(config_file, task_id=0, n_percent=1):
    """runs simulations one after another
    Parameters
    ----------
        config_file : str
            name of the config file
            should be a json containing all necessary parameters for the simulation. Varying parameters should be placed in a subdictionary
            called "varying" (see scgenerator.utilities.dictionary_iterator for details)
        task_id : any formatable (int, string, float, ...)
            give an id for book keeping purposes (must be set if multiple ray instances run at once so their files do not overlap)
        n_percent : int or float
            each individual simulation reports its progress every n_percent percent.

    Returns
    ----------
        name of the folder where the data is stored
    """
    logger = io.Logger()
    state.CurrentLogger.focus_logger(logger)

    config_name, config_dict, store_num, n, m = _sim_preps(config_file)

    # Initiate helper workers (a logger, a progress tracker to give estimates of
    # completion time and an indexer to keep track of the individual files
    # created after each simulation. The indexer can then automatically merge them)
    pt = ProgressTracker(max=n * m * store_num, auto_print=True, percent_incr=1)
    indexer = io.tmp_index_manager(
        config_name=config_name, task_id=task_id, varying_keys=config_dict.get("varying", None)
    )
    logger.log(f"CRITICAL FILE at {indexer.get_path()}, do not touch it")

    # we treat loops over different parameters differently
    for k, dico in enumerate(utilities.dictionary_iterator(config_dict, varying_dict="varying")):
        # loop over same parameter set
        for i in range(n):
            # because of random processes, initial conditions are recalculated every time
            params = initialize.compute_init_parameters(dictionary=config_dict, replace=dico)

            # make sure initial conditions are saved
            params["init_P0"] = dico.get("P0", config_dict.get("P0", 0))
            params["init_T0_FWHM"] = dico.get("T0_FWHM", config_dict.get("T0_FWHM", 0))
            params["param_id"] = k
            params_file_name = io.generate_file_path("param", i, k, task_id, "")
            io.save_parameters(params, params_file_name)
            indexer.append_to_index(k, params_file_name=params_file_name)

            # start a new simulation
            logger.log(
                f"Launching propagation of a {params.get('t0', 0) * 1e15:.2f}fs pulse with {np.max(abs2(params['field_0'])):.0f}W peak power over {np.max(params['z_targets'])}m"
            )
            RK4IP(
                params,
                save_data=True,
                job_id=i,
                param_id=k,
                task_id=task_id,
                pt=pt,
                indexer=indexer,
                logger=logger,
                n_percent=n_percent,
            )

            logger.log(pt.get_eta())

    # merge the data properly
    folder_0 = indexer.convert_sim_data()

    print(f"{config_name} successfully finished ! data saved in {folder_0}")

    return folder_0


def _sim_preps(config_file):
    # Load the config file
    try:
        with open(config_file, "r") as file:
            config_dict = json.loads(file.read())
    except FileNotFoundError:
        print("No config file named {} found".format(config_file))
        raise

    # Store a master dictionary of parameters to generate file names and such
    config_name = config_dict.pop("name", os.path.split(config_file)[-1][:-5])

    # make sure we store spectra every time at the exact same place
    if "z_targets" not in config_dict:
        config_dict["z_targets"] = np.linspace(0, 1, 128)
    config_dict["z_targets"] = initialize.sanitize_z_targets(config_dict["z_targets"])
    config_dict = units.standardize_dictionary(config_dict)
    store_num = len(config_dict["z_targets"])

    # How many total simulations
    n = int(config_dict.pop("n", 1))
    m = np.prod([len(np.atleast_1d(ls)) for _, ls in config_dict.get("varying", {1: 1}).items()])

    return config_name, config_dict, store_num, n, m
