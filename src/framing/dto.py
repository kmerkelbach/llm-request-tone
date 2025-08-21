from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    type: str
    text: str


@dataclass
class ModifiedTask:
    name: str
    origin_task: str
    scenario: Scenario
