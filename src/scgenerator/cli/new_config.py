from .. import const


def list_input():
    answer = ""
    while answer == "":
        answer = input("Please enter a list of values (one per line)\n")

    out = [process_input(answer)]

    while answer != "":
        answer = input()
        out.append(process_input(answer))

    return out[:-1]


def process_input(s):
    try:
        return int(s)
    except ValueError:
        pass

    try:
        return float(s)
    except ValueError:
        pass

    return s


def accept(question, default=True):
    question += " ([y]/n)" if default else " (y/[n])"
    question += "\n"
    inp = input(question)

    yes_str = ["y", "yes"]
    if default:
        yes_str.append("")

    return inp.lower() in yes_str


def get(section, param_name):
    question = f"Please enter a value for the parameter '{param_name}'\n"
    valid = const.valid_param_types[section][param_name]

    is_valid = False
    value = None

    while not is_valid:
        answer = input(question)
        if answer == "\\variable" and param_name in const.valid_variable[section]:
            value = list_input()
            print(value)
            is_valid = all(valid(v) for v in value)
        else:
            value = process_input(answer)
            is_valid = valid(value)

    return value