import os
import shutil
from datetime import datetime
from typing import Dict
from ruamel.yaml import YAML


# Set up yaml parsing
yaml_rt = YAML(typ="rt")          # round-trip
yaml_rt.preserve_quotes = True    # keep original quoting
yaml_rt.width = 10**9             # avoid line wrapping


def get_src_dir() -> str:
    file_path = os.path.realpath(__file__)
    src_dir = os.path.realpath(os.path.join(os.path.split(file_path)[0], ".."))
    return src_dir


def get_eval_dir() -> str:
    return get_src_sister_dir("results")


def get_tables_dir() -> str:
    return get_src_sister_dir("tables")


def get_src_sister_dir(name: str) -> str:
    sister_dir = os.path.realpath(os.path.join(get_src_dir(), "..", name))
    os.makedirs(sister_dir, exist_ok=True)
    return sister_dir


def get_scenario_path() -> str:
    return os.path.join(get_src_dir(), "framing", "tones.json")


def _get_task_root_folder() -> str:
    return os.path.realpath(os.path.join(get_src_dir(), "..", "tasks"))


def get_task_templates_folder() -> str:
    return os.path.join(_get_task_root_folder(), "templates")


def get_task_applied_folder(reset: bool = False) -> str:
    applied_folder = os.path.join(_get_task_root_folder(), "applied")
    if reset and os.path.isdir(applied_folder):
        shutil.rmtree(applied_folder)
    os.makedirs(applied_folder, exist_ok=True)
    return applied_folder


def read_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml_rt.load(f)


def write_yaml(data: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml_rt.dump(data, f)


def make_date_string():
    return (datetime.now().isoformat()
                .replace("-", "_")
                .replace("T", "__")
                .replace(".", "_"))