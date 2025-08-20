import json
import os
from datetime import datetime
from loguru import logger


from evaluation.lm_eval_shell import run_eval


if __name__ == "__main__":
    tasks = ["gsm8k_templated", "mmlu_pro_templated"]
    eval_res = run_eval(
        model="deepseek/deepseek-r1-distill-llama-8b",
        tasks=tasks,
        limit=1,
        num_concurrent=4,
        silent=False,
        log_debug_prompt_file=False
    )

    # Save eval result to disk
    src_dir = os.path.split(os.path.realpath(__file__))[0]
    eval_dir = os.path.realpath(os.path.join(src_dir, "..", "results"))
    os.makedirs(eval_dir, exist_ok=True)

    date_str = (datetime.now().isoformat()
                .replace("-", "_")
                .replace("T", "__")
                .replace(".", "_"))
    eval_filename = f"result_eval_{date_str}.json"

    eval_path = os.path.join(eval_dir, eval_filename)
    with open(eval_path, "w") as f:
        json.dump(eval_res, f, indent=4)

    logger.info(f"Wrote eval results from evaluating {tasks} to file {eval_path}")
