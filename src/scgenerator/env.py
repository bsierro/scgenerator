import os
from typing import Any, Dict, Literal, Optional, Set

from .const import (
    ENVIRON_KEY_BASE,
    LOG_FILE_LEVEL,
    LOG_PRINT_LEVEL,
    PBAR_POLICY,
    TMP_FOLDER_KEY_BASE,
    global_config,
)


def data_folder(task_id: int) -> Optional[str]:
    idstr = str(int(task_id))
    tmp = os.getenv(TMP_FOLDER_KEY_BASE + idstr)
    return tmp


def get(key: str) -> Any:
    str_value = os.environ.get(key)
    if isinstance(str_value, str):
        try:
            t = global_config[key]["type"]
            if t == bool:
                return str_value.lower() == "true"
            return t(str_value)
        except (ValueError, KeyError):
            pass
    return None


def all_environ() -> Dict[str, str]:
    """returns a dictionary of all environment variables set by any instance of scgenerator"""
    d = dict(filter(lambda el: el[0].startswith(ENVIRON_KEY_BASE), os.environ.items()))
    return d


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
