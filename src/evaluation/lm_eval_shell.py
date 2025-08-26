import json
import os
from glob import glob
import random
import string
import subprocess
from typing import Optional, Union, Sequence, Dict
from loguru import logger
from textwrap import wrap


def _normalize_tasks(tasks: Optional[Union[str, Sequence[str]]]) -> str:
    """
    Accepts:
      - None
      - "gsm8k,mmlu_pro"
      - ["gsm8k", "mmlu_pro"]
      - ("gsm8k", "mmlu_pro")
    Returns a comma-separated string suitable for --tasks.
    """
    if tasks is None:
        return "gsm8k,mmlu_pro"
    if isinstance(tasks, str):
        # Allow users to pass a single task or a comma-separated string
        return ",".join([t.strip() for t in tasks.split(",") if t.strip()])
    # Iterable of tasks
    return ",".join([str(t).strip() for t in tasks if str(t).strip()])


def run_eval(
    *,
    # Required-by-you params (but all have sane defaults)
    model: str = "deepseek/deepseek-r1-distill-llama-8b",
    num_concurrent: int = 2,
    tasks: Optional[Union[str, Sequence[str]]] = ("gsm8k", "mmlu_pro"),
    num_fewshot: int = 0,
    limit: Optional[int] = 1,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    output_path: str = "results_llm_eval.json",
    base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    api_key_env: str = "OPENROUTER_API_KEY",
    batch_size: int = 1,
    apply_chat_template: bool = True,
    max_retries: int = 3,
    loglevel: str = "DEBUG",
    lm_eval_executable: str = "lm_eval",
    include_path: str = "../tasks/applied",
    silent: bool = False,
    log_debug_prompt_file: bool = False,
    unsafe_mode: bool = False
) -> Dict:
    """
    Run EleutherAI's lm-evaluation-harness CLI.

    Example:
        from run_lm_eval import run_eval
        run_eval(model="deepseek/deepseek-r1-distill-llama-8b", num_concurrent=2)

    """
    # Get API key from env (we don't put literal ${VARS} into the CLI string)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env} is not set. "
            "Export your OpenRouter API key, e.g.:\n\n"
            f'  export {api_key_env}="sk-or-..."'
        )

    # Build --model_args as a single comma-separated string
    model_args = (
        f"model={model},"
        f"base_url={base_url},"
        f"api_key={api_key},"
        f"num_concurrent={int(num_concurrent)},"
        f"max_retries={int(max_retries)}"
    )

    # Build --gen_kwargs as JSON
    gen_kwargs = json.dumps(
        {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
        separators=(",", ":"),  # compact
    )

    # Normalize tasks to a comma-separated string
    tasks_arg = _normalize_tasks(tasks)

    # Make random output path
    output_filename, output_extension = os.path.splitext(output_path)
    output_filename += "_" + "".join([random.choice(string.digits) for _ in range(6)])
    output_path = output_filename + output_extension

    # Compose the CLI arguments list (no shell=True; no quoting headaches)
    cmd: list[str] = [
        lm_eval_executable,
        "--model",
        "local-chat-completions",
        "--model_args",
        model_args,
        "--include_path",
        include_path,
        "--tasks",
        tasks_arg,
        "--num_fewshot",
        str(int(num_fewshot)),
        "--batch_size",
        str(int(batch_size)),
        "--gen_kwargs",
        gen_kwargs,
        "--output_path",
        output_path
    ]
    if log_debug_prompt_file:
        cmd.append("--log_samples")
    if limit:
        cmd += [
            "--limit",
            str(int(limit))
        ]
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if unsafe_mode:
        cmd.append("--confirm_run_unsafe_code")

    # Prepare environment (inherit + set LOGLEVEL)
    env = os.environ.copy()
    env["LOGLEVEL"] = str(loglevel)
    env["OPENAI_API_KEY"] = api_key

    if not silent:
        logger.info("\n".join(wrap(f"Running command:\n'{' '.join(cmd)}'", width=120)))

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=False,
            capture_output=silent,
            text=True,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find '{lm_eval_executable}' on PATH. "
            "Install lm-evaluation-harness (e.g. `pip install lm-eval==0.*`) "
            "or pass lm_eval_executable='/path/to/lm_eval'."
        ) from e

    # Optional: brief, safe debug echo (mask API key if you log cmd)
    # print("Ran:", " ".join(shlex.quote(part if "api_key=" not in part else "api_key=***") for part in cmd))

    # Raise a clear error if it failed, but include stderr for diagnosis.
    if result.returncode != 0:
        msg = [
            "lm_eval failed.",
            f"Exit code: {result.returncode}",
        ]
        if result.stderr:
            msg.append("stderr:\n" + result.stderr.strip())
        raise RuntimeError("\n".join(msg))

    # Open results file
    result_file = glob(os.path.join(".", output_filename + "*"))[0]
    with open(result_file, "r") as f:
        res = json.load(f)

    # Remove JSON file
    os.remove(result_file)

    return res
