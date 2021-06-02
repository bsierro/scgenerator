import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

from .const import ENVIRON_KEY_BASE, PBAR_POLICY, TMP_FOLDER_KEY_BASE


def data_folder(task_id: int) -> Optional[Path]:
    idstr = str(int(task_id))
    tmp = os.getenv(TMP_FOLDER_KEY_BASE + idstr)
    return tmp


def all_environ() -> Dict[str, str]:
    """returns a dictionary of all environment variables set by any instance of scgenerator"""
    d = dict(filter(lambda el: el[0].startswith(ENVIRON_KEY_BASE), os.environ.items()))
    return d


def pbar_policy() -> List[Literal["print", "file"]]:
    policy = os.getenv(PBAR_POLICY)
    if policy == "print" or policy is None:
        return ["print"]
    elif policy == "file":
        return ["file"]
    elif policy == "both":
        return ["file", "print"]
    else:
        return []
