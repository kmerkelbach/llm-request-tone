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
    "openai/gpt-oss-20b",  # GPT small
    "openai/gpt-oss-120b",  # GPT large
    "meta-llama/llama-3.3-70b-instruct",  # LLama large
    "qwen/qwen-2.5-7b-instruct",  # Qwen small
    "qwen/qwen-2.5-72b-instruct",  # Qwen large
])

lm_eval_limit = None  # set to None for no limit
