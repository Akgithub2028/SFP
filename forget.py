"""
forget.py — Watch an LLM forget in real time.

Run:  python forget.py
Time: ~5 minutes on CPU, ~1 minute on GPU

This script:
1. Loads a tiny LLM (SmolLM2-135M)
2. Shows what it can do (math questions)
3. Fine-tunes it on a different task (code)
4. Shows it can no longer do math
5. Measures exactly how much it forgot

The "wow" demo. Self-contained, no config needed.
"""

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from data import get_batch, load_task
from model import load_model

# ---------------------------------------------------------------------------
# Inline math eval — no dataset download needed
# ---------------------------------------------------------------------------

QUESTIONS = [
    ("What is 15 + 27?", "42"),
    ("What is 8 * 7?", "56"),
    ("What is 144 / 12?", "12"),
    ("If a train travels 60 mph for 2.5 hours, how many miles does it travel?", "150"),
    ("What is 23% of 200?", "46"),
    ("A store sells apples for $1.50 each. How much do 8 apples cost?", "12"),
    ("What is the square root of 169?", "13"),
    ("If you have 3/4 of a pizza and eat 1/3 of it, what fraction is left?", "1/2"),
    ("What is 2^10?", "1024"),
    ("A rectangle is 15 cm long and 8 cm wide. What is its area in cm²?", "120"),
]


def resolve_device(device: str) -> str:
    """Turn 'auto' into the best available device string."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def eval_math(model, tokenizer, device: str) -> tuple[list[bool], list[str]]:
    """Run the inline math questions and return (correct_flags, generated_texts)."""
    model.eval()
    results = []
    generated = []
    for question, gold in QUESTIONS:
        inputs = tokenizer(question, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generated.append(text)
        results.append(gold in text)
    return results, generated


def print_eval(results: list[bool], tag: str) -> None:
    """Pretty-print math eval results."""
    for i, ((question, gold), correct) in enumerate(zip(QUESTIONS, results, strict=True)):
        icon = "✓" if correct else "✗"
        print(f"  {icon}  Q{i+1}: {question}  (answer: {gold})")
    acc = sum(results) / len(results) * 100
    print(f"\n  Math accuracy {tag}: {acc:.0f}%\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch an LLM forget in real time.")
    parser.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | mps")
    args = parser.parse_args()

    device = resolve_device(args.device)

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   Does your LLM forget? Let's find out.         ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading SmolLM2-135M-Instruct + LoRA (rank 16)...")
    model, tokenizer = load_model(lora_rank=16, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {n_params:,}")
    print(f"  Trainable (LoRA): {n_train:,}")
    print(f"  Device:           {device}")
    print()

    # ── Pre-training math eval ────────────────────────────────────────────
    print("─" * 50)
    print("STEP 1: Math eval BEFORE code training")
    print("─" * 50)
    before, _ = eval_math(model, tokenizer, device)
    print_eval(before, "BEFORE code training")

    # ── Fine-tune on code ─────────────────────────────────────────────────
    print("─" * 50)
    print("STEP 2: Fine-tuning on code (200 steps)...")
    print("─" * 50)
    code_data = load_task("code", max_samples=500)
    print(f"  Loaded {len(code_data)} code examples\n")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-4,
    )

    model.train()
    pbar = tqdm(range(200), desc="Training", unit="step")
    for _step in pbar:
        batch = get_batch(code_data, batch_size=4, tokenizer=tokenizer, device=device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    print()

    # ── Post-training math eval ───────────────────────────────────────────
    print("─" * 50)
    print("STEP 3: Math eval AFTER code training")
    print("─" * 50)
    after, _ = eval_math(model, tokenizer, device)
    print_eval(after, "AFTER code training")

    # ── Side-by-side comparison ───────────────────────────────────────────
    print("─" * 50)
    print("COMPARISON: Before vs After")
    print("─" * 50)
    for i, ((question, _gold), b, a) in enumerate(zip(QUESTIONS, before, after, strict=True)):
        b_icon = "✓" if b else "✗"
        a_icon = "✓" if a else "✗"
        print(f"  Q{i+1}: {b_icon} → {a_icon}  {question}")
    print()

    # ── Dramatic summary ──────────────────────────────────────────────────
    acc_before = sum(before) / len(before) * 100
    acc_after = sum(after) / len(after) * 100
    drop = acc_before - acc_after

    print("┌─────────────────────────────────────────────┐")
    print("│  FORGETTING REPORT                          │")
    print("│                                             │")
    acc_line = f"│  Math accuracy:  {acc_before:.0f}% -> {acc_after:.0f}%  (drop {drop:.0f}%)"
    print(acc_line.ljust(46) + "│")
    print("│  Steps trained:  200                        │")
    print("│  Task trained:   code generation            │")
    print("│                                             │")
    if drop > 0:
        print("│  The model forgot how to do math.           │")
    else:
        print("│  No forgetting detected (try more steps).   │")
    print("│                                             │")
    print("│  To prevent this, try:                      │")
    print("│    python train.py --method sfp --memory 128│")
    print("└─────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    main()
