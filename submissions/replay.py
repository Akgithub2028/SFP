"""Mix new-task CE with memory replay CE at 0.1 ratio."""

from torch import Tensor

CONTRIBUTOR = "sfp team"


def replay_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    replay_ratio: float = 0.1,
    **kw,
) -> Tensor:
    """Mix new-task CE with memory replay CE at given ratio."""
    out_new = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    if memory_batch is None:
        return out_new.loss

    out_mem = model(
        input_ids=memory_batch["input_ids"],
        attention_mask=memory_batch["attention_mask"],
        labels=memory_batch["labels"],
    )
    return (1 - replay_ratio) * out_new.loss + replay_ratio * out_mem.loss
