from string import printable as str_printable


def to_62(i: int) -> str:
    arr = []
    if i == 0:
        return "0"
    i = abs(i)
    while i:
        i, value = divmod(i, 62)
        arr.append(str_printable[value])
    return "".join(reversed(arr))
