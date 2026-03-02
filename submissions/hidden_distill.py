"""POD-style hidden state distillation — CE + L2 match of normalized hidden states."""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "sfp team"
SETUP = "hidden_distill"


def hidden_distill_loss(
    model,
    batch: dict,
    teacher=None,
    memory_batch: dict | None = None,
    layers: list[str] | None = None,
    **kw,
) -> Tensor:
    """POD-style: CE + L2 match of normalized hidden states at selected layers."""
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out_new.loss

    if teacher is None or memory_batch is None or not layers:
        return ce_loss

    student_acts: dict[str, Tensor] = {}
    teacher_acts: dict[str, Tensor] = {}

    def _make_hook(store: dict, name: str):
        def hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            store[name] = out
        return hook

    s_handles = []
    t_handles = []
    for name in layers:
        s_mod = dict(model.named_modules())[name]
        t_mod = dict(teacher.named_modules())[name]
        s_handles.append(s_mod.register_forward_hook(_make_hook(student_acts, name)))
        t_handles.append(t_mod.register_forward_hook(_make_hook(teacher_acts, name)))

    try:
        model(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
        with torch.no_grad():
            teacher(
                input_ids=memory_batch["input_ids"],
                attention_mask=memory_batch["attention_mask"],
            )
    finally:
        for h in s_handles + t_handles:
            h.remove()

    distill = torch.tensor(0.0, device=ce_loss.device)
    for name in layers:
        s_norm = F.normalize(student_acts[name].float(), dim=-1)
        t_norm = F.normalize(teacher_acts[name].float(), dim=-1)
        distill = distill + F.mse_loss(s_norm, t_norm)

    return ce_loss + distill
