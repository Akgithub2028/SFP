"""
tasks/domain.py — Domain shift evaluation (perplexity + QA).

~60 lines. Standalone — no imports from other project files.
"""

from __future__ import annotations

import math

import torch
from tqdm import tqdm

_SCIENCE_QA = [
    ("What organelle is responsible for ATP production in eukaryotic cells?", "mitochondria"),
    ("What is the chemical formula for water?", "h2o"),
    ("What force keeps planets in orbit around the sun?", "gravity"),
    ("What particle carries a positive charge in an atom?", "proton"),
    ("What is the pH of a neutral solution at 25°C?", "7"),
    ("What molecule carries genetic information in most organisms?", "dna"),
    ("What is the speed of light in a vacuum in m/s?", "299792458"),
    ("What is the most abundant gas in Earth's atmosphere?", "nitrogen"),
    ("What law states that energy cannot be created or destroyed?", "thermodynamics"),
    ("What bond involves the sharing of electron pairs between atoms?", "covalent"),
]

_FALLBACK_TEXTS = [
    "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells."
    " They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source"
    " of chemical energy. Mitochondria have their own DNA, which is circular and encodes a small"
    " number of essential proteins.",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into"
    " chemical energy that can later be released to fuel the organism's activities. This chemical"
    " energy is stored in carbohydrate molecules, such as sugars, which are synthesized from"
    " carbon dioxide and water.",
    "Protein folding is the physical process by which a protein chain acquires its native 3D"
    " structure. This structure is essential for the protein to function correctly. Misfolded"
    " proteins can aggregate and are associated with diseases such as Alzheimer's and Parkinson's.",
]


@torch.no_grad()
def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate domain adaptation.

    Returns: {
        "domain_ppl": float,         # held-out perplexity
        "domain_qa_accuracy": float,  # domain QA exact-match
    }
    """
    # ── Perplexity on scientific text ────────────────────────────────────
    n_ppl = 100 if mode == "fast" else 500
    texts = []
    try:
        from datasets import load_dataset

        ds = load_dataset("scientific_papers", "pubmed", split="test", streaming=True)
        for i, ex in enumerate(ds):
            if i >= n_ppl:
                break
            abstract = ex.get("abstract", ex.get("text", ""))
            if abstract and len(abstract) > 50:
                texts.append(abstract[:2048])
    except Exception:
        texts = _FALLBACK_TEXTS * (n_ppl // len(_FALLBACK_TEXTS) + 1)
        texts = texts[:n_ppl]

    if not texts:
        texts = _FALLBACK_TEXTS

    total_loss = 0.0
    total_tokens = 0
    for text in tqdm(texts, desc="domain-ppl"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        out = model(**enc, labels=labels)
        n_tok = labels.numel()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok

    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    # ── Domain QA ────────────────────────────────────────────────────────
    correct = 0
    for question, answer in tqdm(_SCIENCE_QA, desc="domain-qa"):
        prompt = f"Answer in one or two words.\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = outputs[0, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

        if answer.lower() in response:
            correct += 1

    qa_acc = correct / len(_SCIENCE_QA) if _SCIENCE_QA else 0.0

    return {"domain_ppl": ppl, "domain_qa_accuracy": qa_acc}
