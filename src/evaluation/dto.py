from typing import Dict
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class EvalResult:
    model: str
    benchmark_base: str
    framework: str
    results: Dict
    date_created: str
