# run_lm_eval.py
import json
import os
import shlex
import subprocess
from typing import Iterable, Optional, Union, Sequence


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
    model: str = "anthropic/claude-3.7-sonnet",
    num_concurrent: int = 2,
    tasks: Optional[Union[str, Sequence[str]]] = ("gsm8k", "mmlu_pro"),
    num_fewshot: int = 5,
    limit: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 512,
    output_path: str = "results_openrouter_smoke.json",
    # Fixed bits from your bash (exposed here in case you want to tweak later)
    base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    api_key_env: str = "OPENROUTER_API_KEY",
    batch_size: int = 1,
    apply_chat_template: bool = True,
    max_retries: int = 3,
    loglevel: str = "DEBUG",
    lm_eval_executable: str = "lm_eval",
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run EleutherAI's lm-evaluation-harness CLI with OpenRouter chat completions.

    Example:
        from run_lm_eval import run_eval
        run_eval(model="anthropic/claude-3.7-sonnet", num_concurrent=2)

    Parameters mirror your bash flags. Any you don't set use defaults.
    Returns subprocess.CompletedProcess (check .returncode, .stdout, .stderr).
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

    # Compose the CLI arguments list (no shell=True; no quoting headaches)
    cmd: list[str] = [
        lm_eval_executable,
        "--model",
        "local-chat-completions",
        "--model_args",
        model_args,
        "--tasks",
        tasks_arg,
        "--num_fewshot",
        str(int(num_fewshot)),
        "--batch_size",
        str(int(batch_size)),
        "--limit",
        str(int(limit)),
        "--gen_kwargs",
        gen_kwargs,
        "--output_path",
        output_path,
    ]
    if apply_chat_template:
        cmd.append("--apply_chat_template")

    # Prepare environment (inherit + set LOGLEVEL)
    env = os.environ.copy()
    env["LOGLEVEL"] = str(loglevel)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=False,  # don't raise automatically; let caller inspect returncode
            capture_output=capture_output,
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

    return result

