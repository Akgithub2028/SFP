"""SFP with random dimension selection — control ablation for importance ranking."""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "sfp team"
SETUP = "sfp"


def sfp_random_selection_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    basis: dict | None = None,
    anchor_acts: dict | None = None,
    lam: float = 0.1,
    **kw,
) -> Tensor:
    """Same as SFP with PCA basis but random dim selection. Control for importance."""
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out_new.loss

    if basis is None or anchor_acts is None or memory_batch is None:
        return ce_loss

    layer_names = list(basis.keys())
    current_acts: dict[str, Tensor] = {}

    def _make_hook(name: str):
        def hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            # mean-pool over sequence dimension → [batch, hidden]
            current_acts[name] = out.float().mean(dim=1)
        return hook

    handles = []
    modules = dict(model.named_modules())
    for name in layer_names:
        handles.append(modules[name].register_forward_hook(_make_hook(name)))

    try:
        model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
    finally:
        for h in handles:
            h.remove()

    preserve = torch.tensor(0.0, device=ce_loss.device)
    for name in layer_names:
        u_r = basis[name].to(current_acts[name].device)  # [hidden, r]
        projected = current_acts[name] @ u_r  # [batch, r]
        anchor = anchor_acts[name].to(projected.device)  # [n, r]
        # align batch sizes (take min)
        n = min(projected.shape[0], anchor.shape[0])
        preserve = preserve + F.mse_loss(projected[:n], anchor[:n])

    return ce_loss + lam * preserve
