import os
import shutil
from datetime import datetime
from typing import Dict
import yaml


# Add constructor for preserving !function field in lm-eval task config files
# Tiny wrapper so we can round-trip the tag
class FunctionTag(str):
    pass

# Read: parse !function <value> into our wrapper
yaml.SafeLoader.add_constructor(
    '!function',
    lambda loader, node: FunctionTag(loader.construct_scalar(node))
)

# Write: emit our wrapper back as a tagged scalar (no quotes)
yaml.SafeDumper.add_representer(
    FunctionTag,
    lambda dumper, data: dumper.represent_scalar('!function', str(data))
)


def get_src_dir() -> str:
    file_path = os.path.realpath(__file__)
    src_dir = os.path.realpath(os.path.join(os.path.split(file_path)[0], ".."))
    return src_dir


def get_eval_dir() ->str:
    eval_dir = os.path.realpath(os.path.join(get_src_dir(), "..", "results"))
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def get_scenario_dir() -> str:
    return os.path.join(get_src_dir(), "framing", "scenarios")


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
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(dic: Dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(dic, f)


def make_date_string():
    return (datetime.now().isoformat()
                .replace("-", "_")
                .replace("T", "__")
                .replace(".", "_"))