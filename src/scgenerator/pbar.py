import multiprocessing
import os
import random
import threading
import typing
from collections import abc
from io import StringIO
from pathlib import Path
from typing import Iterable, Union

from tqdm import tqdm

from scgenerator.env import pbar_policy

T_ = typing.TypeVar("T_")


class PBars:
    def __init__(
        self,
        task: Union[int, Iterable[T_]],
        desc: str,
        num_sub_bars: int = 0,
        head_kwargs=None,
        worker_kwargs=None,
    ) -> "PBars":
        """creates a PBars obj

        Parameters
        ----------
        task : int | Iterable
            if int : total length of the main task
            if Iterable : behaves like tqdm
        desc : str
            description of the main task
        num_sub_bars : int
            number of sub-tasks

        """
        self.id = random.randint(100000, 999999)
        try:
            self.width = os.get_terminal_size().columns
        except OSError:
            self.width = 120
        if isinstance(task, abc.Iterable):
            self.iterator: Iterable[T_] = iter(task)
            self.num_tot: int = len(task)
        else:
            self.num_tot: int = task
            self.iterator = None

        self.policy = pbar_policy()
        if head_kwargs is None:
            head_kwargs = dict()
        if worker_kwargs is None:
            worker_kwargs = dict(
                total=1,
                desc="Worker {worker_id}",
                bar_format="{l_bar}{bar}" "|[{elapsed}<{remaining}, " "{rate_fmt}{postfix}]",
            )
        if "print" not in pbar_policy():
            head_kwargs["file"] = worker_kwargs["file"] = StringIO()
            self.width = 80
        head_kwargs["desc"] = desc
        self.pbars = [tqdm(total=self.num_tot, ncols=self.width, ascii=False, **head_kwargs)]
        for i in range(1, num_sub_bars + 1):
            kwargs = {k: v for k, v in worker_kwargs.items()}
            if "desc" in kwargs:
                kwargs["desc"] = kwargs["desc"].format(worker_id=i)
            self.append(tqdm(position=i, ncols=self.width, ascii=False, **kwargs))
        self.print_path = Path(
            f"progress {self.pbars[0].desc.replace('/', '')} {self.id}"
        ).resolve()
        self.close_ev = threading.Event()
        if "file" in self.policy:
            self.thread = threading.Thread(target=self.print_worker, daemon=True)
            self.thread.start()

    def print(self):
        if "file" not in self.policy:
            return
        s = []
        for pbar in self.pbars:
            s.append(str(pbar))
        self.print_path.write_text("\n".join(s))

    def print_worker(self):
        while True:
            if self.close_ev.wait(2.0):
                return
            self.print()

    def __iter__(self):
        with self as pb:
            for thing in self.iterator:
                yield thing
                pb.update()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, key):
        return self.pbars[key]

    def update(self, i=None, value=1):
        if i is None:
            for pbar in self.pbars[1:]:
                pbar.update(value)
        elif i > 0:
            self.pbars[i].update(value)
        self.pbars[0].update()

    def append(self, pbar: tqdm):
        self.pbars.append(pbar)

    def reset(self, i):
        self.pbars[i].update(-self.pbars[i].n)
        self.print()

    def close(self):
        self.print()
        self.close_ev.set()
        if "file" in self.policy:
            self.thread.join()
        for pbar in self.pbars:
            pbar.close()


class ProgressBarActor:
    def __init__(self, name: str, num_workers: int, num_steps: int) -> None:
        self.counters = [0 for _ in range(num_workers + 1)]
        self.p_bars = PBars(
            num_steps, "Simulating " + name, num_workers, head_kwargs=dict(unit="step")
        )

    def update(self, worker_id: int, rel_pos: float = None) -> None:
        """update a counter

        Parameters
        ----------
        worker_id : int
            id of the worker. 0 is the overall progress
        rel_pos : float, optional
            if None, increase the counter by one, if set, will set
            the counter to the specified value (instead of incrementing it), by default None
        """
        if rel_pos is None:
            self.counters[worker_id] += 1
        else:
            self.counters[worker_id] = rel_pos

    def update_pbars(self):
        for counter, pbar in zip(self.counters, self.p_bars.pbars):
            pbar.update(counter - pbar.n)

    def close(self):
        self.p_bars.close()


def progress_worker(
    name: str, num_workers: int, num_steps: int, progress_queue: multiprocessing.Queue
):
    """keeps track of progress on a separate thread

    Parameters
    ----------
    num_steps : int
        total number of steps, used for the main progress bar (position 0)
    progress_queue : multiprocessing.Queue
        values are either
            Literal[0] : stop the worker and close the progress bars
            tuple[int, float] : worker id and relative progress between 0 and 1
    """
    with PBars(
        num_steps, "Simulating " + name, num_workers, head_kwargs=dict(unit="step")
    ) as pbars:
        while True:
            raw = progress_queue.get()
            if raw == 0:
                return
            i, rel_pos = raw
            if i > 0:
                pbars[i].update(rel_pos - pbars[i].n)
                pbars[0].update()
            elif i == 0:
                pbars[0].update(rel_pos)
