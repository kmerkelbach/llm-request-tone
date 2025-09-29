from ..util.constants import *


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
    # "mbpp_plus_instruct",
    SORRY_BENCH_NAME
])
assert all(b in benchmarks_all for b in benchmarks_selected), "Not all selected benchmarks are in the main list!"

models = sorted([
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    # "meta-llama/llama-3.2-3b-instruct",
    # "qwen/qwen3-30b-a3b-thinking-2507",
    # "x-ai/grok-code-fast-1",
    # "anthropic/claude-sonnet-4",
    # "deepseek/deepseek-chat-v3-0324"
])