"""Prometheus push-gateway metrics logger — lightweight wandb replacement."""

from __future__ import annotations

import os
from urllib.request import Request, build_opener

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


def _auth_handler(url, method, timeout, headers, data):
    """Custom handler that injects X-Push-Token for nginx auth."""
    token = os.environ.get("SFP_PUSH_TOKEN", "")
    req = Request(url, data=data)
    for k, v in headers:
        req.add_header(k, v)
    if token:
        req.add_header("X-Push-Token", token)
    req.get_method = lambda: method
    resp = build_opener().open(req, timeout=timeout)
    return resp


class MetricsLogger:
    def __init__(
        self,
        method: str,
        memory: str,
        seed: int,
        model: str,
        gateway: str = "localhost:9091",
    ):
        self.gateway = gateway
        self.labels = {"method": method, "memory": memory, "seed": str(seed), "model": model}
        self.job = f"sfp_{method}_{seed}"

        self.registry = CollectorRegistry()
        label_keys = ["method", "task", "seed"]

        self.train_loss = Gauge(
            "sfp_train_loss", "Per-step training loss", label_keys, registry=self.registry
        )
        self.train_step = Gauge(
            "sfp_train_step", "Current training step", ["method", "seed"], registry=self.registry
        )
        self.eval_score = Gauge(
            "sfp_eval_score", "Eval score", [*label_keys, "metric"], registry=self.registry
        )
        self.forgetting = Gauge(
            "sfp_forgetting", "Forgetting per task", label_keys, registry=self.registry
        )
        self.retention = Gauge(
            "sfp_retention", "Overall retention", ["method", "seed"], registry=self.registry
        )
        self.plasticity = Gauge(
            "sfp_plasticity", "Overall plasticity", ["method", "seed"], registry=self.registry
        )

    def _push(self) -> None:
        push_to_gateway(
            self.gateway, job=self.job, registry=self.registry,
            grouping_key=self.labels, handler=_auth_handler,
        )

    def push_step(self, step: int, task: str, loss: float) -> None:
        s = self.labels["seed"]
        self.train_loss.labels(method=self.labels["method"], task=task, seed=s).set(loss)
        self.train_step.labels(method=self.labels["method"], seed=s).set(step)
        self._push()

    def push_eval(self, task_results: dict[str, dict[str, float]], forgetting: float) -> None:
        m, s = self.labels["method"], self.labels["seed"]
        for task, metrics in task_results.items():
            for metric_name, value in metrics.items():
                self.eval_score.labels(method=m, task=task, seed=s, metric=metric_name).set(value)
        self.forgetting.labels(method=m, task="aggregate", seed=s).set(forgetting)
        self._push()

    def push_summary(self, retention: float, plasticity: float) -> None:
        m, s = self.labels["method"], self.labels["seed"]
        self.retention.labels(method=m, seed=s).set(retention)
        self.plasticity.labels(method=m, seed=s).set(plasticity)
        self._push()

    def close(self) -> None:
        self._push()
