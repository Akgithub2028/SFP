"""Replay with higher ratio — a simple but strong baseline."""

from torch import Tensor

CONTRIBUTOR = "sfp team"


def strong_replay_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    replay_ratio: float = 0.3,
    **kw,
) -> Tensor:
    """Like replay, but with a higher replay ratio (0.3 vs 0.1)."""
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
