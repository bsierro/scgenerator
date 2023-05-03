import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import scgenerator as sc
import scgenerator.math as math
import scgenerator.physics.units as units
import scgenerator.plotting as plot
import scgenerator.solver as sol


def main():
    params = sc.Parameters(**sc.open_single_config("./tests/Optica_PM2000D/Optica_PM2000D.toml"))
    # print(params.nonlinear_operator)
    # print(params.compute("dispersion_op"))
    # print(params.linear_operator)
    # print(params.spec_0)
    # print(params.compute("gamma_op"))
    #
    # plt.plot(params.w, params.linear_operator(0).imag)
    # plt.show()
    

    breakpoint()

    res = sol.integrate(params.spec_0, params.length, params.linear_operator, params.nonlinear_operator)

    new_z = np.linspace(0, params.length, 256)

    specs2 = math.abs2(res.spectra)
    specs2 = units.to_WL(specs2, params.l)
    x = params.l
    # x = units.THz.inv(w)
    # new_x = np.linspace(100, 2200, 1024)
    new_x = np.linspace(800e-9, 2000e-9, 1024)
    solution = interp1d(res.z, specs2, axis=0)(new_z)
    solution = interp1d(x, solution)(new_x)
    solution = units.to_log2D(solution)

    plt.imshow(
        solution,
        origin="lower",
        aspect="auto",
        extent=plot.get_extent(1e9 * new_x, new_z * 1e2),
        vmin=-30,
    )
    plt.show()

    fields = np.fft.irfft(res.spectra)
    solution = math.abs2(fields)
    solution = interp1d(res.z, solution, axis=0)(new_z)
    solution.T[:] /= solution.max(axis=1)
    plt.imshow(
        solution,
        origin="lower",
        aspect="auto",
        extent=plot.get_extent(params.t * 1e15, new_z * 1e2),
    )
    plt.show()


if __name__ == "__main__":
    main()
