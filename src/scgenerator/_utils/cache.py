from collections import namedtuple
from copy import copy
from functools import wraps
import numpy as np

CacheInfo = namedtuple("CacheInfo", "hits misses size")


def np_cache(func):
    def new_cached_func():
        cache = {}
        hits = misses = 0

        @wraps(func)
        def wrapped(*args, **kwargs):
            nonlocal cache, hits, misses
            hashable_args = tuple(
                tuple(arg) if isinstance(arg, (np.ndarray, list)) else arg for arg in args
            )
            hashable_kwargs = tuple(
                {
                    k: tuple(kwarg) if isinstance(kwarg, (np.ndarray, list)) else kwarg
                    for k, kwarg in kwargs.items()
                }.items()
            )
            key = hash((hashable_args, hashable_kwargs))
            if key not in cache:
                misses += 1
                cache[key] = func(*args, **kwargs)
            else:
                hits += 1
            return copy(cache[key])

        def reset():
            nonlocal cache, hits, misses
            cache = {}
            hits = misses = 0

        wrapped.cache_info = lambda: CacheInfo(hits, misses, len(cache))
        wrapped.reset = reset

        return wrapped

    return new_cached_func()


if __name__ == "__main__":
    import random
    import time

    @np_cache
    def lol(a):
        time.sleep(random.random() * 4)
        return a / 2

    @np_cache
    def ggg(b):
        time.sleep(random.random() * 4)
        return b * 2

    x = np.arange(6)
    for i in range(5):
        print(lol.cache_info())
        print(lol(x))

    print(f"{ggg.cache_info()=}")
    print(f"{lol.cache_info()=}")
    lol.reset()

    print(ggg(np.arange(3)))
    print(ggg(np.arange(8)))
    print(ggg(np.arange(3)))

    print(f"{ggg.cache_info()=}")
    print(f"{lol.cache_info()=}")
