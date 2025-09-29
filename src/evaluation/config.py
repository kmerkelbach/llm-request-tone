from src.util.constants import *


benchmarks_all = [
    "mmlu_pro",
    "gpqa_diamond_cot_zeroshot",
    "gsm8k_cot_llama",
    "ifeval",
    "truthfulqa_gen",
    "mbpp_plus_instruct",
    "bbh_cot_zeroshot",
    SORRY_BENCH_NAME
]

benchmarks_selected = sorted([
    "gpqa_diamond_cot_zeroshot",
    "truthfulqa_gen",
    "mbpp_plus_instruct",
    SORRY_BENCH_NAME
])
assert all(b in benchmarks_all for b in benchmarks_selected), "Not all selected benchmarks are in the main list!"

models = sorted([
    "openai/gpt-oss-20b",  # GPT large
    "openai/gpt-oss-120b",  # GPT large
    "qwen/qwen3-next-80b-a3b-thinking",  # Qwen small
    "qwen/qwen3-235b-a22b-thinking-2507",  # Qwen large
])

lm_eval_limit = None  # set to None for no limit
