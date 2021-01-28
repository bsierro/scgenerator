from scgenerator.initialize import ParamSequence
from logging import StreamHandler
from scgenerator import io
from scgenerator import utilities

# dispatcher = ParamSequence(io.load_toml("testing/configs/ensure_consistency/good4"))
dispatcher = ParamSequence(io.load_toml("testing/configs/compute_init_parameters/good"))
print(dispatcher)

for only, params in dispatcher:
    print(only, params["width"])
print(len(dispatcher))
print(dispatcher["fiber", "length"])

print(utilities.varying_list_from_path("/a_5_b_asdf"))