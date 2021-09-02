"""
This files includes utility functions designed more or less to be used specifically with the
scgenerator module but some function may be used in any python program

"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import random
import re
import shutil
import threading
from collections import abc
from copy import deepcopy
from io import StringIO
from pathlib import Path
from string import printable as str_printable
from typing import Any, Callable, Generator, Iterable, MutableMapping, Sequence, TypeVar, Union

import numpy as np
import pkg_resources as pkg
import toml
from tqdm import tqdm

from ..const import PARAM_FN, PARAM_SEPARATOR, SPEC1_FN, SPECN_FN, Z_FN, __version__
from ..env import TMP_FOLDER_KEY_BASE, data_folder, pbar_policy
from ..errors import IncompleteDataFolderError
from ..logger import get_logger

T_ = TypeVar("T_")

PathTree = list[tuple[Path, ...]]


class Paths:
    _data_files = [
        "silica.toml",
        "gas.toml",
        "hr_t.npz",
        "submit_job_template.txt",
        "start_worker.sh",
        "start_head.sh",
    ]

    paths = {
        f.split(".")[0]: os.path.abspath(
            pkg.resource_filename("scgenerator", os.path.join("data", f))
        )
        for f in _data_files
    }

    @classmethod
    def get(cls, key):
        if key not in cls.paths:
            if os.path.exists("paths.toml"):
                with open("paths.toml") as file:
                    paths_dico = toml.load(file)
                for k, v in paths_dico.items():
                    cls.paths[k] = v
        if key not in cls.paths:
            get_logger(__name__).info(
                f"{key} was not found in path index, returning current working directory."
            )
            cls.paths[key] = os.getcwd()

        return cls.paths[key]

    @classmethod
    def gets(cls, key):
        """returned the specified file as a string"""
        with open(cls.get(key)) as file:
            return file.read()

    @classmethod
    def plot(cls, name):
        """returns the paths to the specified plot. Used to save new plot
        example
        ---------
        fig.savefig(Paths.plot("figure5.pdf"))
        """
        return os.path.join(cls.get("plots"), name)


def load_previous_spectrum(prev_data_dir: str) -> np.ndarray:
    prev_data_dir = Path(prev_data_dir)
    num = find_last_spectrum_num(prev_data_dir)
    return np.load(prev_data_dir / SPEC1_FN.format(num))


def conform_toml_path(path: os.PathLike) -> Path:
    path = Path(path)
    if not path.name.lower().endswith(".toml"):
        path = path.parent / (path.name + ".toml")
    return path


def load_toml(path: os.PathLike):
    """returns a dictionary parsed from the specified toml file"""
    path = conform_toml_path(path)
    with open(path, mode="r") as file:
        dico = toml.load(file)
    dico.setdefault("variable", {})
    for key in {"simulation", "fiber", "gas", "pulse"} & dico.keys():
        section = dico.pop(key, {})
        dico["variable"].update(section.pop("variable", {}))
        dico.update(section)
    if len(dico["variable"]) == 0:
        dico.pop("variable")
    return dico


def save_toml(path: os.PathLike, dico):
    """saves a dictionary into a toml file"""
    path = conform_toml_path(path)
    with open(path, mode="w") as file:
        toml.dump(dico, file)
    return dico


def load_config_sequence(final_config_path: os.PathLike) -> tuple[list[dict[str, Any]], str]:
    loaded_config = load_toml(final_config_path)
    final_name = loaded_config.get("name")
    fiber_list = loaded_config.pop("Fiber")
    configs = []
    if fiber_list is not None:
        for i, params in enumerate(fiber_list):
            params.setdefault("variable", loaded_config.get("variable", {}) if i == 0 else {})
            configs.append(loaded_config | params)
    else:
        configs.append(loaded_config)
        while "previous_config_file" in configs[0]:
            configs.insert(0, load_toml(configs[0]["previous_config_file"]))
        configs[0].setdefault("variable", {})
        for pre, nex in zip(configs[:-1], configs[1:]):
            variable = nex.pop("variable", {})
            nex.update({k: v for k, v in pre.items() if k not in nex})
            nex["variable"] = variable

    return configs, final_name


def save_parameters(
    params: dict[str, Any], destination_dir: Path, file_name: str = PARAM_FN
) -> Path:
    """saves a parameter dictionary. Note that is does remove some entries, particularly
    those that take a lot of space ("t", "w", ...)

    Parameters
    ----------
    params : dict[str, Any]
        dictionary to save
    destination_dir : Path
        destination directory

    Returns
    -------
    Path
        path to newly created the paramter file
    """
    file_path = destination_dir / file_name
    os.makedirs(file_path.parent, exist_ok=True)

    # save toml of the simulation
    with open(file_path, "w") as file:
        toml.dump(params, file, encoder=toml.TomlNumpyEncoder())

    return file_path


def load_material_dico(name: str) -> dict[str, Any]:
    """loads a material dictionary
    Parameters
    ----------
        name : str
            name of the material
    Returns
    ----------
        material_dico : dict
    """
    if name == "silica":
        return toml.loads(Paths.gets("silica"))
    else:
        return toml.loads(Paths.gets("gas"))[name]


def update_appended_params(source: Path, destination: Path, z: Sequence):
    z_num = len(z)
    params = load_toml(source)
    if "simulation" in params:
        params["simulation"]["z_num"] = z_num
        params["fiber"]["length"] = float(z[-1] - z[0])
    else:
        params["z_num"] = z_num
        params["length"] = float(z[-1] - z[0])
    for p_name in ["recovery_data_dir", "prev_data_dir", "output_path"]:
        if p_name in params:
            del params[p_name]
    save_toml(destination, params)


def to_62(i: int) -> str:
    arr = []
    if i == 0:
        return "0"
    i = abs(i)
    while i:
        i, value = divmod(i, 62)
        arr.append(str_printable[value])
    return "".join(reversed(arr))


def build_path_trees(sim_dir: Path) -> list[PathTree]:
    sim_dir = sim_dir.resolve()
    path_branches: list[tuple[Path, ...]] = []
    to_check = list(sim_dir.glob("*fiber*num*"))
    with PBars(len(to_check), desc="Building path trees") as pbar:
        for branch in map(build_path_branch, to_check):
            if branch is not None:
                path_branches.append(branch)
                pbar.update()
    path_trees = group_path_branches(path_branches)
    return path_trees


def build_path_branch(data_dir: Path) -> tuple[Path, ...]:
    if not data_dir.is_dir():
        return None
    path_branch = [data_dir]
    while (prev_sim_path := load_toml(path_branch[-1] / PARAM_FN).get("prev_data_dir")) is not None:
        p = Path(prev_sim_path).resolve()
        if not p.exists():
            p = Path(*p.parts[-2:]).resolve()
        path_branch.append(p)
    return tuple(reversed(path_branch))


def group_path_branches(path_branches: list[tuple[Path, ...]]) -> list[PathTree]:
    """groups path lists

    [
        ("a/id 0 wavelength 100 num 0"," b/id 0 wavelength 100 num 0"),
        ("a/id 2 wavelength 100 num 1"," b/id 2 wavelength 100 num 1"),
        ("a/id 1 wavelength 200 num 0"," b/id 1 wavelength 200 num 0"),
        ("a/id 3 wavelength 200 num 1"," b/id 3 wavelength 200 num 1")
    ]
    ->
    [
        (
            ("a/id 0 wavelength 100 num 0", "a/id 2 wavelength 100 num 1"),
            ("b/id 0 wavelength 100 num 0", "b/id 2 wavelength 100 num 1"),
        )
        (
            ("a/id 1 wavelength 200 num 0", "a/id 3 wavelength 200 num 1"),
            ("b/id 1 wavelength 200 num 0", "b/id 3 wavelength 200 num 1"),
        )
    ]


    Parameters
    ----------
    path_branches : list[tuple[Path, ...]]
        each element of the list is a path to a folder containing data of one simulation

    Returns
    -------
    list[PathTree]
        list of PathTrees to be used in merge
    """
    sort_key = lambda el: el[0]

    size = len(path_branches[0])
    out_trees_map: dict[str, dict[int, dict[int, Path]]] = {}
    for branch in path_branches:
        b_id = branch_id(branch)
        out_trees_map.setdefault(b_id, {i: {} for i in range(size)})
        for sim_part, data_dir in enumerate(branch):
            num = re.search(r"(?<=num )[0-9]+", data_dir.name)[0]
            out_trees_map[b_id][sim_part][int(num)] = data_dir

    return [
        tuple(
            tuple(w for _, w in sorted(v.items(), key=sort_key))
            for __, v in sorted(d.items(), key=sort_key)
        )
        for d in out_trees_map.values()
    ]


def merge_path_tree(
    path_tree: PathTree, destination: Path, z_callback: Callable[[int], None] = None
):
    """given a path tree, copies the file into the right location

    Parameters
    ----------
    path_tree : PathTree
        elements of the list returned by group_path_branches
    destination : Path
        dir where to save the data
    """
    z_arr: list[float] = []

    destination.mkdir(exist_ok=True)

    for i, (z, merged_spectra) in enumerate(merge_spectra(path_tree)):
        z_arr.append(z)
        spec_out_name = SPECN_FN.format(i)
        np.save(destination / spec_out_name, merged_spectra)
        if z_callback is not None:
            z_callback(i)
    d = np.diff(z_arr)
    d[d < 0] = 0
    z_arr = np.concatenate(([z_arr[0]], np.cumsum(d)))
    np.save(destination / Z_FN, z_arr)
    update_appended_params(path_tree[-1][0] / PARAM_FN, destination / PARAM_FN, z_arr)


def merge_spectra(
    path_tree: PathTree,
) -> Generator[tuple[float, np.ndarray], None, None]:
    for same_sim_paths in path_tree:
        z_arr = np.load(same_sim_paths[0] / Z_FN)
        for i, z in enumerate(z_arr):
            spectra: list[np.ndarray] = []
            for data_dir in same_sim_paths:
                spec = np.load(data_dir / SPEC1_FN.format(i))
                spectra.append(spec)
            yield z, np.atleast_2d(spectra)


def merge(destination: os.PathLike, path_trees: list[PathTree] = None):

    destination = ensure_folder(Path(destination))

    z_num = 0
    prev_z_num = 0

    for i, sim_dir in enumerate(sim_dirs(path_trees)):
        conf = sim_dir / "initial_config.toml"
        shutil.copy(
            conf,
            destination / f"initial_config_{i}.toml",
        )
        prev_z_num = load_toml(conf).get("z_num", prev_z_num)
        z_num += prev_z_num

    pbars = PBars(
        len(path_trees) * z_num, "Merging", 1, worker_kwargs=dict(total=z_num, desc="current pos")
    )
    for path_tree in path_trees:
        pbars.reset(1)
        iden_items = path_tree[-1][0].name.split()[2:]
        for i, p_name in list(enumerate(iden_items))[-2::-2]:
            if p_name == "num":
                del iden_items[i + 1]
                del iden_items[i]
        iden = PARAM_SEPARATOR.join(iden_items)
        merge_path_tree(path_tree, destination / iden, z_callback=lambda i: pbars.update(1))


def sim_dirs(path_trees: list[PathTree]) -> Generator[Path, None, None]:
    for p in path_trees[0]:
        yield p[0].parent


def save_data(data: np.ndarray, data_dir: Path, file_name: str):
    """saves numpy array to disk

    Parameters
    ----------
    data : np.ndarray
        data to save
    file_name : str
        file name
    task_id : int
        id that uniquely identifies the process
    identifier : str, optional
        identifier in the main data folder of the task, by default ""
    """
    path = data_dir / file_name
    np.save(path, data)
    get_logger(__name__).debug(f"saved data in {path}")
    return


def ensure_folder(path: Path, prevent_overwrite: bool = True, mkdir=True) -> Path:
    """ensure a folder exists and doesn't overwrite anything if required

    Parameters
    ----------
    path : Path
        desired path
    prevent_overwrite : bool, optional
        whether to create a new directory when one already exists, by default True

    Returns
    -------
    Path
        final path
    """

    path = path.resolve()

    # is path root ?
    if len(path.parts) < 2:
        return path

    # is a part of path an existing *file* ?
    parts = path.parts
    path = Path(path.root)
    for part in parts:
        if path.is_file():
            path = ensure_folder(path, mkdir=mkdir, prevent_overwrite=False)
        path /= part

    folder_name = path.name

    for i in itertools.count():
        if not path.is_file() and (not prevent_overwrite or not path.is_dir()):
            if mkdir:
                path.mkdir(exist_ok=True)
            return path
        path = path.parent / (folder_name + f"_{i}")


class PBars:
    def __init__(
        self,
        task: Union[int, Iterable[T_]],
        desc: str,
        num_sub_bars: int = 0,
        head_kwargs=None,
        worker_kwargs=None,
    ) -> "PBars":

        self.id = random.randint(100000, 999999)
        try:
            self.width = os.get_terminal_size().columns
        except OSError:
            self.width = 80
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


def branch_id(branch: tuple[Path, ...]) -> str:
    return branch[-1].name.split()[1]


def find_last_spectrum_num(data_dir: Path):
    for num in itertools.count(1):
        p_to_test = data_dir / SPEC1_FN.format(num)
        if not p_to_test.is_file() or os.path.getsize(p_to_test) == 0:
            return num - 1


def auto_crop(x: np.ndarray, y: np.ndarray, rel_thr: float = 0.01) -> np.ndarray:
    threshold = y.min() + rel_thr * (y.max() - y.min())
    above_threshold = y > threshold
    ind = np.argsort(x)
    valid_ind = [
        np.array(list(g)) for k, g in itertools.groupby(ind, key=lambda i: above_threshold[i]) if k
    ]
    ind_above = sorted(valid_ind, key=lambda el: len(el), reverse=True)[0]
    width = len(ind_above)
    return np.concatenate(
        (
            np.arange(max(ind_above[0] - width, 0), ind_above[0]),
            ind_above,
            np.arange(ind_above[-1] + 1, min(len(y), ind_above[-1] + width)),
        )
    )


def translate_parameters(d: dict[str, Any]) -> dict[str, Any]:
    old_names = dict(interp_degree="interpolation_degree")
    new = {}
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            new[k] = translate_parameters(v)
        else:
            new[old_names.get(k, k)] = v
    return new
