import json
import os
import random
from glob import glob

from loguru import logger
from typing import List, Dict, Optional

from .framing.dto import ModifiedTask
from .util.utils import get_eval_dir, make_date_string
from .viz.results_table import TableMaker


if __name__ == "__main__":
    # Init table maker
    results_dir = get_eval_dir()
    table_maker = TableMaker(
        results_dir=results_dir
    )