import numpy as np
import scgenerator as sc
import matplotlib.pyplot as plt


def main():
    params = sc.Configuration(
        "/Users/benoitsierro/tests/test_sc/Travers2019 Fig2 dev/initial_config.toml"
    ).first
    evalu = sc.Evaluator.default()
    evalu.set(**params.prepare_for_dump())
    n_eff = evalu.compute("n_eff")
    wl = evalu.compute("wl_for_disp")
    w = evalu.compute("w_for_disp")
    print(w.max(), w.min())
    disp_inf = evalu.compute("dispersion_ind")
    # quit()

    params.interpolation_degree = 6
    params.compute()
    current_disp = params.linear_operator.dispersion_op
    beta = n_eff * w / 3e8
    beta1 = np.gradient(beta, w)
    ind = sc.argclosest(w, params.w0)
    disp = -1j * (beta - beta1[ind] * (w - params.w0) - beta[ind])
    disp2 = sc.fiber.fast_direct_dispersion(w, params.w0, n_eff, ind)

    plt.plot(params.l * 1e9, current_disp(None).imag)
    plt.plot(wl * 1e9, disp.imag)
    plt.plot(wl * 1e9, disp2.imag)
    plt.xlim(100, 3000)
    plt.ylim(-1000, 4000)
    plt.show()


if __name__ == "__main__":
    main()
