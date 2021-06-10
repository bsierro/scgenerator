from numba.core import config
from scgenerator.initialize import Config, Params, BareParams
from scgenerator.utils import variable_iterator, override_config
from scgenerator.io import load_toml
from pprint import pprint
from dataclasses import asdict

dico = load_toml("testing/configs/ensure_consistency/good2.toml")
out = dict(variable=dict())
for k, v in dico.items():
    if isinstance(v, dict):
        for kk, vv in v.items():
            if kk == "variable":
                for kkk, vvv in vv.items():
                    out["variable"][kkk] = vvv
            else:
                out[kk] = vv

pprint(out)
p = Config(**out)
print(p)

for l, c in variable_iterator(p):
    print(l, c.width, c.intensity_noise)
    print()

config2 = override_config(dict(width=1.2e-13, variable=dict(peak_power=[1e5, 2e5])), p)
print(
    f"{config2.variable=}",
    f"{config2.intensity_noise=}",
    f"{config2.width=}",
    f"{config2.peak_power=}",
)

par = BareParams()

print(all(v is None for v in vars(par).values()))
