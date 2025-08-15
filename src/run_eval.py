from evaluation.lm_eval_shell import run_eval


if __name__ == "__main__":
    run_eval(
        model="deepseek/deepseek-r1-distill-llama-8b",
        limit=1
    )