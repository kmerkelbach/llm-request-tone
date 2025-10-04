from typing import Dict, List, Optional
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


@dataclass
class StatTestData:
    group_a_name: str
    group_a_vals: List[float]
    group_b_name: str
    group_b_vals: List[float]
    domain_name: str
    test_alpha: Optional[float]

    # the following are not known initially, only after running the test
    test_p_value: Optional[float]
    test_is_significant: Optional[bool]
