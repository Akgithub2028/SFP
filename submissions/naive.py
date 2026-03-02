"""Standard cross-entropy loss — no forgetting mitigation."""

from torch import Tensor

CONTRIBUTOR = "sfp team"


def naive_loss(model, batch: dict, **kw) -> Tensor:
    """Standard cross-entropy loss. No forgetting mitigation."""
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.loss
