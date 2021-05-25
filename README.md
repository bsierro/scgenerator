It is recommended to import scgenerator in the following manner :
`import scgenerator as sc`

# How to run a set of simulations
create a config file

run `sc.parallel_simulations(config_file)` or `sc.simulate(config_file)`

# How to analyse a simulation

load data with the load_sim_data method
spectra, params = load_sim_data("varyTechNoise100kW_sim_data")
    to plot
        plot_results_2D(spectra[0], (600, 1450, nm), params)


# Configuration

You can load parameters by simpling passing the path to a toml file to the appropriate simulation function. Each possible key of this dictionary is described below. Every value must be given in standard SI units (m, s, W, J, ...)
The configuration file can have a ```name``` parameter at the root and must otherwise contain the following sections with the specified parameters

note : internally, another structure with a flattened dictionary is used



## Fiber parameters
If you already know the Taylor coefficients corresponding to the expansion of the beta2 profile, you can specify them and skip to "Other fiber parameters":

beta: list-like
    list of Taylor coefficients for the beta_2 function


else, you can choose a mathematical fiber model

model: str {"pcf", "marcatili", "marcatili_adjusted", "hasan"}

**PCF** : solid core silica photonic crystal fiber, as modeled in Saitoh, Kunimasa, and Masanori Koshiba. "Empirical relations for simple design of photonic crystal fibers." Optics express 13.1 (2005): 267-274.

**marcatili** : Marcatili model of a capillary fiber : Marcatili, Enrique AJ, and R. A. Schmeltzer. "Hollow metallic and dielectric waveguides for long distance optical transmission and lasers." Bell System Technical Journal 43.4 (1964): 1783-1809.

**marcatili_adjusted** : Marcatili model of a capillary fiber with adjusted effective radius in the longer wavelength : Köttig, F., et al. "Novel mid-infrared dispersive wave generation in gas-filled PCF by transient ionization-driven changes in dispersion." arXiv preprint arXiv:1701.04843 (2017).

**hasan** : Hasan model of hollow core anti-resonance fibers : Hasan, Md Imran, Nail Akhmediev, and Wonkeun Chang. "Empirical formulae for dispersion and effective mode area in hollow-core antiresonant fibers." Journal of Lightwave Technology 36.18 (2018): 4060-4065.

and specify the parameters it needs

pcf : 

pitch: float
    distance between air holes in m
pitch_ratio: float 0.2 < pitch_ratio < 0.8
    ratio hole diameter/pich

marcatili, marcatili_adjusted, hasan :

core_radius: float
    radius of the hollow core in m


marcatili, marcatili_adjusted :

he_mode: list, shape (2, ), optional
    mode of propagation. default is (1, 1), which is the fundamental mode

marcatili_adjusted :

fit_parameters: list, shape (2, ), optional
    parameters for the effective radius correction. Defaults are (s, h) = (0.08, 200e-9) as in the referenced paper.

hasan :

capillary_num : int
    number of capillaries

capillary_outer_d : float, optional if g is specified
    outer diameter of the capillaries

capillary_thickness : float
    thickness of the capillary walls

capillary_spacing : float, optional if d is specified
    spacing between the capillary

capillray_resonance_strengths : list, optional
    list of resonance strengths. Default is []

capillary_nested : int, optional
    how many nested capillaries. Default is 0

## Other fiber parameters :
   

gamma: float, optional unless beta is directly provided
    nonlinear parameter in m^2 / W. Will overwrite any computed gamma parameter.

length: float, optional
    length of the fiber in m. default : 1

fiber_id : int
    in case multiple fibers are chained together, indicates the index of this particular fiber, default : 0

input_transmission : float
    number between 0 and 1 indicating how much light enters the fiber, default : 1


## Gas parameters
this section is completely optional and ignored if the fiber model is "pcf"

gas_name: str
    name of the gas. default : "vacuum"

pressure: float
    pressure of the gas in the fiber. default : 1e5

temperature: float
    temperature of the gas in the fiber. default : 300

plasma_density: float
    constant plasma density (in m^-3). default : 0

## Pulse parameters:
### Mandatory

wavelength: float
    pump wavelength in m

To specify the initial pulse shape, either use one of 2 in (power, energy) together with one of 2 in (width, t0), or use soliton_num together with one of 4 in (power, energy, width, t0)

power: float
    peak power in W

energy: float
    total pulse energy in J

width: float
    full width half maximum of the pulse in s. Will be converted to appropriate t0 depending on pulse shape

t0: float
    pulse width parameter

solition_num: float
    soliton number

### optional

quantum_noise: bool
    whether or not one-photon-per-mode quantum noise is activated. default : False

intensity_noise: float
    relative intensity noise

shape: str {"gaussian", "sech"}
    shape of the pulse. default : gaussian



## Simulation parameters
### 2 of 3

dt: float
    resolution of the temporal grid in s
    
t_num: int
    number of temporal grid points

time_window: float
    total length of the temporal grid in s

### optional
behaviors: list of str {"spm", "raman", "ss"}
    spm is self-phase modulation
    raman is raman effect
    ss is self-steepening
    default : ["spm", "ss"]

raman_type: str {"measured", "stolen", "agrawal"}
    type of Raman effect. Default is "agrawal".

ideal_gas: bool
    if True, use the ideal gas law. Otherwise, use van der Waals equation. default : False

z_num : int
    number of spatial grid points along the fiber. default : 128

frep: float
    repetition rate in Hz. Only useful to convert units. default : 80e6

tolerated_error: float
    relative tolerated step-to-step error. default : 1e-11

step_size: float
    if given, sets a constant step size rather than adapting it.

parallel: bool
    whether to run simulations in parallel with the available ressources. default : false

repeat: int
    how many simulations to run per parameter set. default : 1

lower_wavelength_interp_limit: float
    dispersion coefficients are computed over a certain wavelength range. This parameter
    sets the lowest end of this range. If the set value is lower than the lower end of the
    wavelength window, it is raised up to that point. default : 0

upper_wavelength_interp_limit: float
    dispersion coefficients are computed over a certain wavelength range. This parameter
    sets the lowest end of this range. If the set value is higher than the higher end of the
    wavelength window, it is lowered down to that point. default : 1900e-9

