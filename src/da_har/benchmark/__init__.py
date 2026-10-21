"""Benchmark harness for unsupervised domain adaptation on wearable HAR.

Modules
-------
``domains``
    Explicit task definitions (which domains are source/target and why).
``protocol``
    Subject-disjoint splits plus the target-label audit.
``models``
    Reference backbone(s); deliberately not a research claim.
``methods``
    ``source_only`` / ``coral`` / ``mmd`` / ``dann`` baselines.
``trainer``
    Source-selected training loop.
``stats``
    Paired bootstrap over seeds.
``runner``
    CLI that writes ``results/da_benchmark/<run_id>.json``.
"""

from .domains import SHIFT_TYPES, DomainRequest, TaskData, TaskSpec, build_tasks
from .methods import ADAPTATION_METHODS, build_method
from .models import MaskedCNN1D, build_backbone
from .protocol import ProtocolAudit, Split, audit_task, subject_disjoint_split
from .stats import paired_bootstrap, summarize
from .trainer import TrainConfig, evaluate, train_task

__all__ = [
    "ADAPTATION_METHODS",
    "DomainRequest",
    "MaskedCNN1D",
    "ProtocolAudit",
    "SHIFT_TYPES",
    "Split",
    "TaskData",
    "TaskSpec",
    "TrainConfig",
    "audit_task",
    "build_backbone",
    "build_method",
    "build_tasks",
    "evaluate",
    "paired_bootstrap",
    "subject_disjoint_split",
    "summarize",
    "train_task",
]
