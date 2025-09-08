from typing import Dict
from dataclasses import dataclass


@dataclass
class EvalResult:
    model: str
    benchmark_base: str
    results: Dict
