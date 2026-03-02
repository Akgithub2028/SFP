"""
tasks/code.py — Code generation evaluation (HumanEval).

~100 lines. Standalone — no imports from other project files.
"""

from __future__ import annotations

import subprocess
import tempfile

import torch
from datasets import load_dataset
from tqdm import tqdm


def _run_test(full_code: str, timeout: float = 5.0) -> bool:
    """Execute generated code + test cases in a subprocess."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


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

    passed = 0
    total = 0

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

        if _run_test(full_code):
            passed += 1
        total += 1

    return {"code_pass_at_1": passed / total if total > 0 else 0.0}
