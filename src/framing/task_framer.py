import shutil
import os
from glob import glob
from typing import List, Optional, Dict
from loguru import logger

from .dto import Scenario, ModifiedTask
from ..util.utils import get_scenario_dir, get_task_applied_folder, get_task_templates_folder, read_yaml, write_yaml
from ..util.fields import *


class TaskFramer:
    def __init__(self):
        # Load scenarios
        self.scenarios = self._load_scenarios()
        logger.info(f"Loaded {len(self.scenarios)} scenarios.")

        # Create and, if it exists, clear applied tasks dir
        self._applied_tasks_dir = get_task_applied_folder(reset=True)

        self._templated_tasks: List[ModifiedTask] = []

    def template_all_tasks(self) -> List[ModifiedTask]:
        for task_folder in glob(os.path.join(get_task_templates_folder(), "*")):
            self._apply_scenarios(task_folder)
        return self._templated_tasks

    def _apply_scenarios(self, task_folder: str) -> None:
        for scenario in self.scenarios:
            self._apply_scenario_to_task(task_folder, scenario)

    def _apply_scenario_to_task(self, task_folder: str, scenario: Scenario):
        # Copy task template to applied dir
        task_name_addition = f"_templated_{scenario.name}"
        origin_task = os.path.basename(task_folder)
        dst_task_name = origin_task + task_name_addition
        dst_task_path = os.path.join(self._applied_tasks_dir, dst_task_name)
        shutil.copytree(task_folder, dst_task_path)

        # Change the content of the extra message to the scenario text
        extra_message_path = os.path.join(dst_task_path, "extra_text.txt")
        with open(extra_message_path, "w") as f:
            f.write(scenario.text)

        def rename_task(task_name, task_name_addition):
            task_name = task_name.replace("_templated", "")
            return task_name + task_name_addition

        # Change task name within task yaml files
        for dir_path, dir_names, filenames in os.walk(dst_task_path):
            for yaml_filename in [f for f in filenames if os.path.splitext(f)[-1] == ".yaml"]:
                if not yaml_filename.endswith(".yaml"):
                    continue

                # Set task name
                yaml_file_path = os.path.join(dir_path, yaml_filename)
                contents = read_yaml(yaml_file_path)

                for field_name in [FIELD_TASK, FIELD_GROUP]:
                    if field_name in contents:
                        task_name = contents[field_name]
                        if isinstance(task_name, str):
                            contents[field_name] = rename_task(task_name, task_name_addition)
                        elif isinstance(task_name, list):  # list
                            contents[field_name] = [rename_task(_task_name, task_name_addition)
                                                    for _task_name in task_name]

                write_yaml(contents, yaml_file_path)

                # Save task
                if isinstance(task_name, str):
                    dst_task_name = task_name + task_name_addition
                    self._templated_tasks.append(
                        ModifiedTask(
                            name=dst_task_name,
                            origin_task=task_name,
                            scenario=scenario
                        )
                    )

    @staticmethod
    def _load_scenarios() -> List[Scenario]:
        loaded: List[Dict] = []
        for scenario_path in glob(os.path.join(get_scenario_dir(), "*.yaml")):
            loaded.append(read_yaml(scenario_path))

        # Convert to scenarios
        scenarios: List[Scenario] = []
        for dic in loaded:
            scenarios.append(
                Scenario(**dic)
            )

        return scenarios
