"""LwF-style logit distillation — CE on new task + KL(teacher || student) on memory."""

import torch
import torch.nn.functional as F
from torch import Tensor

CONTRIBUTOR = "sfp team"
SETUP = "distill"


def distill_loss(
    model,
    batch: dict,
    teacher=None,
    memory_batch: dict | None = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    **kw,
) -> Tensor:
    """LwF-style: CE on new task + KL(teacher || student) on memory."""
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    ce_loss = out_new.loss

    if teacher is None or memory_batch is None:
        return ce_loss

    with torch.no_grad():
        teacher_out = teacher(
            input_ids=memory_batch["input_ids"],
            attention_mask=memory_batch["attention_mask"],
        )
    student_out = model(
        input_ids=memory_batch["input_ids"],
        attention_mask=memory_batch["attention_mask"],
    )

    t_logits = teacher_out.logits / temperature
    s_logits = student_out.logits / temperature

    kl = F.kl_div(
        F.log_softmax(s_logits, dim=-1),
        F.softmax(t_logits, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)

    return ce_loss + alpha * kl
