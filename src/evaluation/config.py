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
    # "gpqa_diamond_cot_zeroshot",
    # "truthfulqa_gen",
    # "mbpp_plus_instruct",
    SORRY_BENCH_NAME
])
assert all(b in benchmarks_all for b in benchmarks_selected), "Not all selected benchmarks are in the main list!"

benchmarks_display_names = {
    "mmlu_pro": "MMLU Pro",
    "gpqa_diamond_cot_zeroshot": "GPQA Diamond",
    "gsm8k_cot_llama": "GSM8K",
    "ifeval": "IFEval",
    "truthfulqa_gen": "TruthfulQA",
    "mbpp_plus_instruct": "MBPP+",
    "bbh_cot_zeroshot": "BIG-Bench Hard",
    SORRY_BENCH_NAME: "SORRY-Bench",
}

models = sorted([
    "openai/gpt-oss-20b",  # GPT small
    "openai/gpt-oss-120b",  # GPT large
    "qwen/qwen-2.5-7b-instruct",  # Qwen small
    "qwen/qwen-2.5-72b-instruct",  # Qwen large
    "meta-llama/llama-3.1-8b-instruct",  # Llama small
    "meta-llama/llama-3.1-70b-instruct",  # Llama large
])

lm_eval_limit = None  # set to None for no limit

temperature = 0.7
max_num_repetitions = 3
