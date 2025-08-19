from evaluation.lm_eval_shell import run_eval


if __name__ == "__main__":
    d = run_eval(
        model="deepseek/deepseek-r1-distill-llama-8b",
        tasks=["gsm8k_templated", "gsm8k_cot_templated", "gsm8k_cot_llama_templated",
               "gsm8k_cot_self_consistency_templated", "gsm8k_cot_zeroshot_templated"],
        limit=1,
        num_concurrent=4,
        silent=False
    )
    print(d)