from enum import Enum, auto


class Test:
    class State(Enum):
        complete = auto()
        partial = auto()
        absent = auto()

    def state(self):
        return self.State.complete


a = Test()
print(a.state() == Test.State.complete)
