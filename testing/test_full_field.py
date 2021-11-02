import warnings

import matplotlib.pyplot as plt
import numpy as np
import scgenerator as sc
from customfunc.app import PlotApp
from scgenerator.physics.simulate import RK4IP
from customfunc import pprint

warnings.filterwarnings("error")


def main():
    params = sc.Parameters.load("testing/configs/Chang2011Fig2.toml")
    x = params.l * 1e9
    o = np.argsort(x)
    x = x[o]

    plt.plot(x, sc.abs2(params.spec_0[o]))
    state = sc.operators.CurrentState(
        params.length, 0, params.step_size, 1.0, params.ifft, params.spec_0
    )
    # expD = np.exp(state.h / 2 * params.linear_operator(state))
    # plt.plot(x, expD.imag[o], x, expD.real[o])
    plt.plot(x, sc.abs2(params.nonlinear_operator(state))[o])
    plt.yscale("log")
    plt.xlim(100, 2000)
    plt.show()

    # for *_, spec in RK4IP(params).irun():
    #     plt.plot(w[2:-2], sc.abs2(spec[ind]))
    #     plt.show()
    #     plt.close()


if __name__ == "__main__":
    main()
