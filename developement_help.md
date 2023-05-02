## add parameter
- add it to ```utils.parameters```
- add it to README.md
- add the necessary Rules in ```utils.parameters```
- optional : add a default value
- optional : add to mandatory_parameters

## complicated Rule conditions
- add the desired parameters to the init of the operator
- raise OperatorError if the conditions are not met

## operators
There are 3 kinds of operators
- `SpecOperator(spec: np.ndarray, z: float) -> np.ndarray` : operate on the spectrum
    - Envelope nonlinear operator used in the solver
    - Full field nonlinear operator used in the solver
- `FieldOperator(field: np.ndarray, z: float) -> np.ndarray` : operate on the field
    - SPM
    - Raman
    - Ionization
- `VariableQuantity(z: float) -> float | np.ndarray` : return the value of a certain quantity along the fiber depending on z
    - dispersion
    - refractive index
    - full field nonlinear prefactor
    - nonlinear parameter (chi3, n2, gamma)
