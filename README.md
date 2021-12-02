It is recommended to import scgenerator in the following manner :
`import scgenerator as sc`

# How to run a set of simulations
create a config file

run `sc.run_simulation(config_file)`

# How to analyse a simulation

Load completed simulations with the SimulationSeries class. The path argument should be that of the last fiber in the series (of the form `xx fiber A...` where `xx` is an integer)

The SimulationSeries class has basic plotting functions available. See the class documentation for more info

```
series = sc.SimulationSeries(path_to_data_folder)
fig, ax = plt.subplots()
series.plot_2D(800, 1600, "nm", ax)
plt.show()
```

# Environment variables

`SCGENERATOR_PBAR_POLICY` : "none", "file", "print", "both", optional
    whether progress should be printed to a file ("file"), to the standard output ("print") or both, default : print

`SCGENERATOR_LOG_FILE_LEVEL` : "debug", "info", "warning", "error", "critical", optional
    level of logging printed in $PWD/scgenerator.log

`SCGENERATOR_LOG_PRINT_LEVEL` : "debug", "info", "warning", "error", "critical", optional
    level of logging printed in the cli.

# Configuration

You can load parameters by simply passing the path to a toml file to the appropriate simulation function. Each possible key of this dictionary is described below. Every value must be given in standard SI units (m, s, W, J, ...)
the root of the file has information concerning the whole simulation : name, grid information, input pulse, ...
Then, there must be a `[[Fiber]]` array with at least one fiber with fiber-specific parameters.

Parameters can be variable (either in the root or in one fiber). if at most one single `[variable]` dict is specified by section, all the possible combinations of those variable parameters are considered. Another possibility is to specify a `[[variable]]` array where the length of each set of parameter is the same so they're coupled to each other. 

Examples : 

```
n2 = 2.2e-20
```
a single simulation is ran with this value
```
[variable]
n2 = [2.1e-20, 2.4e-20, 2.6e-20]
```
3 simulations are ran, one for each value
```
n2 = 2.2e-20
[variable]
n2 = [2.1e-20, 2.4e-20, 2.6e-20]
```
NOT ALLOWED. You cannot specified the same parameter twice.

Here is an example of a configuration file

```
# these be applied to the whole simulation fiber PM1550_1 only
# fiber parameters specified here would apply to the whole simulation as well
# unless overridden in one of the individual fiber
name = "Test/Compound 1"

field_file = "Toptica/init_field.npz"
repetition_rate = 40e6
wavelength = 1535.6e-9

dt = 1e-15
t_num = 16384
tolerated_error = 1e-6
quantum_noise = true
z_num = 32
mean_power = 200e-3
repeat = 3


[[variable]]
spm = [true, false]
raman_type = ["agrawal", "stolen"]

[[Fiber]]
name = "PM1550_1"
n2 = 2.2e-20
dispersion_file = "PM1550/Dispersion/PM1550XP extrapolated 1.npz"
length = 0.01
effective_mode_diameter = 10.1e-6

[[Fiber]]
name = "PM2000D_2"
length = 0.01
n2 = 3.4e-20
A_eff_file = "PM2000D/PM2000D_A_eff_marcuse.npz"
dispersion_file = "PM2000D/Dispersion/PM2000D_1 extrapolated 0 4.npz"

[Fiber.variable] # this variable parameter will be applied to PM2000D_2
input_transmission = [0.9, 0.95]
```

this means that only `(spm=true, raman_type="agrawal")` and `(spm=false, raman_type="stolen")` are considered and not `(spm=false, raman_type="agrawal")` for example. In the end, 12 simulations are ran with this configuration.



## Fiber parameters
If you already know the Taylor coefficients corresponding to the expansion of the beta2 profile, you can specify them and skip to "Other fiber parameters":

beta2_coefficients: list-like
    list of Taylor coefficients for the beta_2 function

If you already have a dispersion curve, you can convert it to a npz file with the wavelength (key : 'wavelength') in m and the D parameter (key : 'dispersion') in s/m/m. You the refer to this file as

dispersion_file : str
    path to the npz dispersion file


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

capillary_radius : float, optional if g is specified
    outer radius of the capillaries

capillary_thickness : float
    thickness of the capillary walls

capillary_spacing : float, optional if d is specified
    spacing between the capillary

capillary_resonance_strengths : list, optional
    list of resonance strengths. Default is []

capillary_resonance_max_order : int, optional
    max order of resonance strengths to be deduced

capillary_nested : int, optional
    how many nested capillaries. Default is 0

## Other fiber parameters :
   

gamma: float, optional unless beta is directly provided
    nonlinear parameter in m^-1*W^-1. Will overwrite any computed gamma parameter.

effective_mode_diameter : float, optional
    effective mode field diameter in m

n2 : float, optional
    non linear refractive index in m^2/W

A_eff : float, optional
    effective mode field area

A_eff_file : str, optional
    file containing an A_eff array (in m^2) as function of a wavelength array (in m)

length: float, optional
    length of the fiber in m. default : 1

input_transmission : float, optional
    number between 0 and 1 indicating how much light enters the fiber, useful when chaining many fibers together, default : 1

zero_dispersion_wavelength : float, optional
    target zero dispersion wavelength for hollow capillaries (Marcatili only)


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

To specify the initial pulse properties, either use one of 3 in (peak_power, energy, mean_power) together with one of 2 in (width, t0), or use soliton_num together with one of 5 in (peak_power, mean_power, energy, width, t0)

peak_power : float
    peak power in W

mean_power : float
    mean power of the pulse train in W. if specified, repetition_rate must also be specified

repetition_rate : float
    repetition rate of the pulse train in Hz

energy: float
    total pulse energy in J

width: float
    full width half maximum of the pulse in s. Will be converted to appropriate t0 depending on pulse shape

t0: float
    pulse width parameter

soliton_num: float
    soliton number

### optional

field_file : str
    if you have an initial field to use, convert it to a npz file with time (key : 'time') in s and electric field (key : 'field') in sqrt(W) (can be complex). You the use it with this config key. You can then scale it by settings any 1 of mean_power, energy and peak_power (priority is in this order)

quantum_noise : bool
    whether or not one-photon-per-mode quantum noise is activated. default : False

intensity_noise : float
    relative intensity noise

noise_correlation : float
    correlation between intensity noise and pulse width noise. a negative value means anti-correlation

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
raman_type: str {"measured", "stolen", "agrawal"}, optional
    type of Raman effect. Specifying this parameter has the effect of turning on Raman effect

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
    whether to run simulations in parallel with the available resources. default : false

repeat: int
    how many simulations to run per parameter set. default : 1

interpolation_range : tuple[float, float]
    range over which dispersion is computed and interpolated in m. ex: (500e-9, 2000e-9)

interpolation_degree: int
    max degree of the Taylor polynomial fitting the dispersion data
