"""
tasks/code.py — Code generation evaluation (HumanEval).

Sandboxed execution: tries Docker first (no network, memory/CPU limits,
read-only fs), falls back to OS-level resource limits on Linux, or
timeout-only on macOS. Defense in depth — not a perfect sandbox.

Standalone — no imports from other project files.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import tempfile

import torch
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sandbox detection (runs once at import time)
# ---------------------------------------------------------------------------

def _check_docker() -> bool:
    """Return True if Docker daemon is available."""
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5.0,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_DOCKER_AVAILABLE = _check_docker()
_IS_LINUX = platform.system() == "Linux"

if _DOCKER_AVAILABLE:
    SANDBOX_MODE = "docker"
elif _IS_LINUX:
    SANDBOX_MODE = "restricted"  # resource limits via setrlimit
else:
    SANDBOX_MODE = "none"  # macOS fallback: timeout only

_SANDBOX_ANNOUNCED = False

# ---------------------------------------------------------------------------
# Execution backends
# ---------------------------------------------------------------------------

_MINIMAL_ENV = {"PATH": "/usr/bin:/usr/local/bin:/bin"}


def _run_docker(filepath: str, timeout: float) -> bool:
    """Run code inside a disposable Docker container with tight limits."""
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network=none",           # no internet
                "--memory=256m",            # 256 MB RAM cap
                "--cpus=1",                 # 1 CPU core
                "--read-only",              # read-only root fs
                "--tmpfs", "/tmp:rw,size=64m",  # writable /tmp, 64 MB
                "-v", f"{filepath}:/code.py:ro",
                "python:3.11-slim",
                "python", "/code.py",
            ],
            capture_output=True,
            timeout=timeout + 5,  # extra grace for container startup
            text=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def _make_preexec_fn():
    """Return a preexec_fn that sets resource limits (Linux only)."""
    import resource

    def _set_limits():
        # 256 MB virtual memory
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        # 10 s CPU time
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        # No forking — prevent fork bombs
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

    return _set_limits


def _run_restricted(filepath: str, timeout: float) -> bool:
    """Run with OS-level resource limits (Linux) or timeout-only (macOS)."""
    kwargs: dict = dict(
        capture_output=True,
        timeout=timeout,
        text=True,
        env=_MINIMAL_ENV,
    )
    # On Linux, apply resource limits via preexec_fn
    if _IS_LINUX:
        kwargs["preexec_fn"] = _make_preexec_fn()

    try:
        result = subprocess.run(
            [sys.executable, filepath],
            **kwargs,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _run_test(full_code: str, timeout: float = 5.0) -> bool:
    """Execute generated code + test cases in a sandbox."""
    global _SANDBOX_ANNOUNCED
    if not _SANDBOX_ANNOUNCED:
        print(f"[code eval] sandbox mode: {SANDBOX_MODE}")
        _SANDBOX_ANNOUNCED = True

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=True, dir=None,
    ) as f:
        f.write(full_code)
        f.flush()

        if SANDBOX_MODE == "docker":
            return _run_docker(f.name, timeout)
        else:
            return _run_restricted(f.name, timeout)


def _extract_completion(gen_text: str, entry_point: str) -> str:
    """Extract the function body — stop at the next top-level def or class."""
    lines = gen_text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        # Stop at a new top-level definition (but not the entry_point itself)
        if (stripped.startswith("def ") or stripped.startswith("class ")) and result:
            break
        result.append(line)
    return "\n".join(result)


@torch.no_grad()
def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate code generation.

    fast: HumanEval pass@1 (50 problems, execution-based).
    full: all 164 HumanEval problems.

    Returns: {"code_pass_at_1": float}
    """
    ds = load_dataset("openai/openai_humaneval", split="test")
    n = 5 if mode == "smoke" else 50 if mode == "fast" else len(ds)
    ds = ds.select(range(n))

    per_example: list[float] = []

    for ex in tqdm(ds, desc="code", total=n):
        prompt = ex["prompt"]
        test_code = ex["test"]
        entry_point = ex["entry_point"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = outputs[0, inputs["input_ids"].shape[1] :]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        completion = _extract_completion(gen_text, entry_point)
        full_code = prompt + completion + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

        per_example.append(1.0 if _run_test(full_code) else 0.0)

    pass_at_1 = sum(per_example) / len(per_example) if per_example else 0.0
    return {"code_pass_at_1": pass_at_1, "code_pass_at_1_scores": per_example}
