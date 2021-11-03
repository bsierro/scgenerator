import warnings
import numpy as np
import rediscache
import scgenerator as sc
from customfunc.app import PlotApp
from scipy.interpolate import interp1d
from tqdm import tqdm

# warnings.filterwarnings("error")


@rediscache.rcache
def get_specs(params: dict):
    p = sc.Parameters(**params)
    sim = sc.RK4IP(p)
    return [s[-1] for s in tqdm(sim.irun(), total=p.z_num)], p.dump_dict()


def main():

    params = sc.Parameters.load("testing/configs/Chang2011Fig2.toml")
    specs, params = get_specs(params.dump_dict(add_metadata=False))
    params = sc.Parameters(**params)
    rs = sc.PlotRange(100, 1500, "nm")
    rt = sc.PlotRange(-500, 500, "fs")
    x, o, ext = rs.sort_axis(params.w)
    vmin = -50
    with PlotApp(i=(int, 0, params.z_num - 1)) as app:
        spec_ax = app[0]
        spec_ax.set_xlabel(rs.unit.label)
        field_ax = app[1]
        field_ax.set_xlabel(rt.unit.label)

        @app.update
        def draw(i):
            spec, *fields = compute(i)
            spec_ax.set_line_data("spec", *spec, label=f"z = {params.z_targets[i]*1e2:.0f}cm")
            for label, x, y in fields:
                field_ax.set_line_data(label, x, y)

        print(params)

        @app.cache
        def compute(i):
            xt, field = sc.transform_1D_values(params.ifft(specs[i]), rt, params)
            x, spec = sc.transform_1D_values(sc.abs2(specs[i]), rs, params, log=True)
            # spec = np.where(spec > vmin, spec, vmin)
            field2 = sc.abs2(field)
            bot, top = sc.math.envelope_ind(field2)
            return (x, spec), ("field^2", xt, field2), ("envelope", xt[top], field2[top])

            # bot, top = sc.math.envelope_ind(field)
            # bot = interp1d(xt[bot], field[bot], "cubic", bounds_error=False, fill_value=0)(xt)
            # top = interp1d(xt[top], field[top], "cubic", bounds_error=False, fill_value=0)(xt)

            # return ((x, spec), ("upper", xt, top), ("field", xt, field), ("lower", xt, bot))


if __name__ == "__main__":
    main()
