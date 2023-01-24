import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set

ENVIRON_KEY_BASE = "SCGENERATOR_"
TMP_FOLDER_KEY_BASE = ENVIRON_KEY_BASE + "SC_TMP_"
PREFIX_KEY_BASE = ENVIRON_KEY_BASE + "PREFIX_"

PBAR_POLICY = ENVIRON_KEY_BASE + "PBAR_POLICY"
LOG_FILE_LEVEL = ENVIRON_KEY_BASE + "LOG_FILE_LEVEL"
LOG_PRINT_LEVEL = ENVIRON_KEY_BASE + "LOG_PRINT_LEVEL"
START_RAY = ENVIRON_KEY_BASE + "START_RAY"
NO_RAY = ENVIRON_KEY_BASE + "NO_RAY"
OUTPUT_PATH = ENVIRON_KEY_BASE + "OUTPUT_PATH"


global_config: dict[str, dict[str, Any]] = {
    LOG_FILE_LEVEL: dict(
        help="minimum lvl of message to be saved in the log file",
        choices=["critical", "error", "warning", "info", "debug"],
        default=None,
        type=str,
    ),
    LOG_PRINT_LEVEL: dict(
        help="minimum lvl of message to be printed to the standard output",
        choices=["critical", "error", "warning", "info", "debug"],
        default="error",
        type=str,
    ),
    PBAR_POLICY: dict(
        help="what to do with progress pars (print them, make them a txt file or nothing), default is print",
        choices=["print", "file", "both", "none"],
        default=None,
        type=str,
    ),
    START_RAY: dict(action="store_true", help="initialize ray (ray must be installed)", type=bool),
    NO_RAY: dict(action="store_true", help="force not to use ray", type=bool),
    OUTPUT_PATH: dict(
        short_name="-o", help="path to the final output folder", default=None, type=str
    ),
}


def data_folder(task_id: int) -> Optional[str]:
    idstr = str(int(task_id))
    tmp = os.getenv(TMP_FOLDER_KEY_BASE + idstr)
    return tmp


def get(key: str, default=None) -> Any:
    str_value = os.environ.get(key)
    if isinstance(str_value, str):
        try:
            t = global_config[key]["type"]
            if t == bool:
                return str_value.lower() == "true"
            return t(str_value)
        except (ValueError, KeyError):
            pass
    return default


def all_environ() -> Dict[str, str]:
    """returns a dictionary of all environment variables set by any instance of scgenerator"""
    d = dict(filter(lambda el: el[0].startswith(ENVIRON_KEY_BASE), os.environ.items()))
    return d


def output_path() -> Path:
    p = get(OUTPUT_PATH)
    if p is not None:
        return Path(p).resolve()
    return None


def pbar_policy() -> Set[Literal["print", "file"]]:
    policy = get(PBAR_POLICY)
    if policy == "print" or policy is None:
        return {"print"}
    elif policy == "file":
        return {"file"}
    elif policy == "both":
        return {"file", "print"}
    else:
        return set()


def log_file_level() -> Set[Literal["critical", "error", "warning", "info", "debug"]]:
    policy = get(LOG_FILE_LEVEL)
    try:
        policy = policy.lower()
        if policy in {"critical", "error", "warning", "info", "debug"}:
            return policy
    except AttributeError:
        pass
    return None


def log_print_level() -> Set[Literal["critical", "error", "warning", "info", "debug"]]:
    policy = get(LOG_PRINT_LEVEL)
    try:
        policy = policy.lower()
        if policy in {"critical", "error", "warning", "info", "debug"}:
            return policy
    except AttributeError:
        pass
    return None
