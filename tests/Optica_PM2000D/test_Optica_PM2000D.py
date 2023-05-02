import matplotlib.pyplot as plt

import scgenerator as sc
import scgenerator.solver as sol
import scgenerator.math as math


def main():
    params = sc.Parameters(**sc.open_single_config("Optica_PM2000D.toml"))
    print(params.nonlinear_operator)
    print(params.compute("dispersion_op"))
    print(params.linear_operator)
    print(params.spec_0)
    print(params.compute("gamma_op"))

    plt.plot(params.w, params.linear_operator(0).imag)
    plt.show()

    res = sol.integrate(
        params.spec_0, params.length, params.linear_operator, params.nonlinear_operator
    )
    plt.plot(res.spectra[0].real)
    plt.show()


if __name__ == "__main__":
    main()
