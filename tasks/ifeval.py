"""
tasks/ifeval.py — Instruction-following compliance evaluation.

~100 lines. Standalone — no imports from other project files.
Objective, judge-free: checks if output satisfies programmatic constraints.
"""

from __future__ import annotations

import json
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

# ── constraint checkers ──────────────────────────────────────────────────────


def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return max(len(parts), 1) if text.strip() else 0


def _count_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _relation_check(actual: int, relation: str, target: int) -> bool:
    if relation == "at least":
        return actual >= target
    elif relation == "at most":
        return actual <= target
    return actual == target


def _check_constraint(instruction_id: str, kwargs: dict, response: str) -> bool:
    """Check a single IFEval constraint by instruction_id + kwargs."""
    kwargs = kwargs or {}

    if instruction_id == "keywords:existence":
        keywords = kwargs.get("keywords", [])
        low = response.lower()
        return all(kw.lower() in low for kw in keywords)

    if instruction_id == "keywords:frequency":
        keyword = kwargs.get("keyword", "")
        frequency = kwargs.get("frequency", 1)
        return response.lower().count(keyword.lower()) >= frequency

    if instruction_id == "keywords:forbidden_words":
        forbidden = kwargs.get("forbidden_words", [])
        low = response.lower()
        return all(fw.lower() not in low for fw in forbidden)

    if instruction_id == "keywords:letter_frequency":
        letter = kwargs.get("letter", "")
        freq = kwargs.get("let_frequency", 1)
        return response.lower().count(letter.lower()) >= freq

    if instruction_id == "length_constraints:number_words":
        relation = kwargs.get("relation", "at least")
        num = kwargs.get("num_words", 0)
        return _relation_check(_count_words(response), relation, num)

    if instruction_id == "length_constraints:number_sentences":
        relation = kwargs.get("relation", "at least")
        num = kwargs.get("num_sentences", 0)
        return _relation_check(_count_sentences(response), relation, num)

    if instruction_id == "length_constraints:number_paragraphs":
        relation = kwargs.get("relation", "at least")
        num = kwargs.get("num_paragraphs", 0)
        return _relation_check(len(_count_paragraphs(response)), relation, num)

    if instruction_id == "length_constraints:nth_paragraph_first_word":
        num_paragraphs = kwargs.get("num_paragraphs", 1)
        nth = kwargs.get("nth_paragraph", 1)
        first_word = kwargs.get("first_word", "")
        paras = _count_paragraphs(response)
        if len(paras) < num_paragraphs:
            return False
        if nth < 1 or nth > len(paras):
            return False
        words = paras[nth - 1].split()
        return bool(words) and words[0].lower().startswith(first_word.lower())

    if instruction_id == "change_case:english_uppercase":
        letters = [c for c in response if c.isalpha()]
        return all(c.isupper() for c in letters) if letters else False

    if instruction_id == "change_case:english_lowercase":
        letters = [c for c in response if c.isalpha()]
        return all(c.islower() for c in letters) if letters else False

    if instruction_id == "change_case:english_capital":
        words = [w for w in response.split() if w and w[0].isalpha()]
        if not words:
            return False
        cap_count = sum(1 for w in words if w[0].isupper())
        return cap_count / len(words) > 0.5

    if instruction_id == "detectable_format:number_bullet_lists":
        num_bullets = kwargs.get("num_bullets", 1)
        bullet_lines = [
            l for l in response.split("\n")
            if re.match(r"\s*[-*•]\s+", l) or re.match(r"\s*\d+[.)]\s+", l)
        ]
        return len(bullet_lines) >= num_bullets

    if instruction_id == "detectable_format:constrained_response":
        return len(response.strip()) > 0

    if instruction_id == "detectable_format:number_highlighted_sections":
        num_highlights = kwargs.get("num_highlights", 1)
        highlights = re.findall(r"\*{1,2}[^*]+\*{1,2}", response)
        return len(highlights) >= num_highlights

    if instruction_id == "detectable_format:multiple_sections":
        splitter = kwargs.get("section_spliter", kwargs.get("section_splitter", "\n"))
        num_sections = kwargs.get("num_sections", 1)
        sections = [s.strip() for s in response.split(splitter) if s.strip()]
        return len(sections) >= num_sections

    if instruction_id == "detectable_format:json_format":
        try:
            json.loads(response.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    if instruction_id == "detectable_format:title":
        return any(line.strip().startswith("#") for line in response.split("\n"))

    if instruction_id == "detectable_content:number_placeholders":
        num_placeholders = kwargs.get("num_placeholders", 1)
        placeholders = re.findall(r"\[[^\]]+\]", response)
        return len(placeholders) >= num_placeholders

    if instruction_id == "detectable_content:postscript":
        return bool(re.search(r"p\.?\s*s\.?", response, re.IGNORECASE))

    if instruction_id == "combination:two_responses":
        separators = ["***", "---", "Response 1", "Response 2", "RESPONSE 1", "RESPONSE 2"]
        return any(sep in response for sep in separators)

    if instruction_id == "combination:repeat_prompt":
        prompt = kwargs.get("prompt_to_repeat", "")
        return prompt.lower() in response.lower() if prompt else False

    if instruction_id == "startend:end_checker":
        end_phrase = kwargs.get("end_phrase", "")
        return response.rstrip().endswith(end_phrase) if end_phrase else False

    if instruction_id == "startend:quotation":
        stripped = response.strip()
        return (
            (stripped.startswith('"') and stripped.endswith('"'))
            or (stripped.startswith("\u201c") and stripped.endswith("\u201d"))
        )

    if instruction_id == "punctuation:no_comma":
        return "," not in response

    if instruction_id == "language:response_language":
        return True  # skip — needs classifier

    # Unrecognized constraint: strict — do not count as satisfied
    return False


@torch.no_grad()
def evaluate(model, tokenizer, mode: str = "fast") -> dict[str, float]:
    """Evaluate instruction-following compliance.

    Verifiable constraints: word count, format, keyword inclusion, etc.

    Returns: {"ifeval_accuracy": float}
    """
    ds = load_dataset("google/IFEval", split="train")
    n = 5 if mode == "smoke" else 100 if mode == "fast" else len(ds)
    ds = ds.select(range(n))

    satisfied = 0
    total = 0

    for ex in tqdm(ds, desc="ifeval", total=n):
        prompt = ex["prompt"]
        instructions = ex.get("instruction_id_list", [])
        kwargs_list = ex.get("kwargs", [{}] * len(instructions))

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = outputs[0, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Check all constraints for this example
        all_ok = all(
            _check_constraint(inst_id, kw, response)
            for inst_id, kw in zip(instructions, kwargs_list)
        )
        if all_ok:
            satisfied += 1
        total += 1

    return {"ifeval_accuracy": satisfied / total if total > 0 else 0.0}
