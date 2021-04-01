from .. import const
import toml

valid_commands = ["finish", "next"]


class Configurator:
    def __init__(self, name):
        self.config = dict(name=name, fiber=dict(), gas=dict(), pulse=dict(), simulation=dict())

    def list_input(self):
        answer = ""
        while answer == "":
            answer = input("Please enter a list of values (one per line)\n")

        out = [self.process_input(answer)]

        while answer != "":
            answer = input()
            out.append(self.process_input(answer))

        return out[:-1]

    def process_input(self, s):
        try:
            return int(s)
        except ValueError:
            pass

        try:
            return float(s)
        except ValueError:
            pass

        return s

    def accept(self, question, default=True):
        question += " ([y]/n)" if default else " (y/[n])"
        question += "\n"
        inp = input(question)

        yes_str = ["y", "yes"]
        if default:
            yes_str.append("")

        return inp.lower() in yes_str

    def print_current(self, config: dict):
        print(toml.dumps(config))

    def get(self, section, param_name):
        question = f"Please enter a value for the parameter '{param_name}'\n"
        valid = const.valid_param_types[section][param_name]

        is_valid = False
        value = None

        while not is_valid:
            answer = input(question)
            if answer == "variable" and param_name in const.valid_variable[section]:
                value = self.list_input()
                print(value)
                is_valid = all(valid(v) for v in value)
            else:
                value = self.process_input(answer)
                is_valid = valid(value)

        return value

    def ask_next_command(self):
        s = ""
        raw_input = input(s).split(" ")
        return raw_input[0], raw_input[1:]

    def main(self):
        editing = True
        while editing:
            command, args = self.ask_next_command()
