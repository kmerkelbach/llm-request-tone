from evaluation.lm_eval_shell import run_eval


if __name__ == "__main__":
    d = run_eval(
        model="deepseek/deepseek-r1-distill-llama-8b",
        tasks=["gsm8k_templated", "mmlu_pro_templated"],
        limit=None,
        num_concurrent=32,
        silent=False,
        log_debug_prompt_file=True
    )
    print(d)