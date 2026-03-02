"""Constrain new LoRA to be orthogonal to previous adapters."""

import torch
from torch import Tensor

CONTRIBUTOR = "sfp team"
SETUP = "orthogonal"


def orthogonal_loss(
    model,
    batch: dict,
    prev_lora_weights: dict | None = None,
    ortho_lambda: float = 1.0,
    **kw,
) -> Tensor:
    """Constrain new LoRA to be orthogonal to previous adapters."""
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out.loss

    if not prev_lora_weights:
        return ce_loss

    ortho_reg = torch.tensor(0.0, device=ce_loss.device)
    for name, param in model.named_parameters():
        if "lora_B" not in name or not param.requires_grad:
            continue
        prev_key = name.replace("lora_B", "lora_B_prev")
        if prev_key not in prev_lora_weights:
            base_key = name
            if base_key not in prev_lora_weights:
                continue
            prev_key = base_key
        b_prev = prev_lora_weights[prev_key].to(param.device)
        # ||B_new^T @ B_prev||^2_F
        ortho_reg = ortho_reg + torch.norm(param.T @ b_prev, p="fro") ** 2

    return ce_loss + ortho_lambda * ortho_reg
