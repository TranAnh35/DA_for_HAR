"""Task definitions: which domains are source, which is target, under which shift.

A *task* is intentionally explicit.  The benchmark never guesses a split from a
global pool of windows; it materialises a source domain and a target domain
from named keys (``subject:07``, ``device:s3``, ...) and records the shift type
in the result JSON.  That is the difference between "domain adaptation" and
"train/test split with extra steps".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..data.core import DatasetBundle, DomainData
from ..data.registry import load_dataset

SHIFT_TYPES: Tuple[str, ...] = (
    "cross_subject",
    "cross_device",
    "cross_placement",
    "cross_session",
)


@dataclass(frozen=True)
class DomainRequest:
    """Named domain keys plus the loader kwargs needed to obtain them."""

    keys: Tuple[str, ...]
    loader: Mapping[str, Any] = field(default_factory=dict)

    def materialize(self, dataset: str, root: str = "Data") -> DomainData:
        bundle = _load_cached(dataset, root, _loader_key(self))
        return bundle.combined(list(self.keys), name="+".join(self.keys))


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset: str
    shift: str
    source: DomainRequest
    target: DomainRequest

    def materialize(self, root: str = "Data") -> "TaskData":
        if self.shift not in SHIFT_TYPES:
            raise ValueError(f"{self.name}: unknown shift {self.shift!r}")
        src = self.source.materialize(self.dataset, root)
        tgt = self.target.materialize(self.dataset, root)
        bundle = _load_cached(self.dataset, root, _loader_key(self.source))
        return TaskData(
            spec=self,
            source=src,
            target=tgt,
            label_names=bundle.label_names,
        )


@dataclass
class TaskData:
    spec: TaskSpec
    source: DomainData
    target: DomainData
    label_names: Tuple[str, ...]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def shift(self) -> str:
        return self.spec.shift

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def summary(self) -> Dict[str, Any]:
        return {
            "task": self.name,
            "dataset": self.spec.dataset,
            "shift": self.shift,
            "source_windows": self.source.num_windows,
            "target_windows": self.target.num_windows,
            "source_subjects": sorted(set(self.source.subjects.tolist())),
            "target_subjects": sorted(set(self.target.subjects.tolist())),
            "source_slots": list(self.source.slot_names()),
            "target_slots": list(self.target.slot_names()),
            "num_classes": self.num_classes,
            "sampling_rate": self.source.sampling_rate,
        }


@lru_cache(maxsize=16)
def _load_cached(dataset: str, root: str, loader_items: Tuple[Tuple[str, Any], ...]) -> DatasetBundle:
    kwargs = dict(loader_items)
    return load_dataset(dataset, root=root, **kwargs)


def _hashable(value: Any) -> Any:
    """Make loader kwargs usable as an lru_cache key (lists -> tuples)."""
    if isinstance(value, Mapping):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


def _loader_key(request: DomainRequest) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((k, _hashable(v)) for k, v in request.loader.items()))


def clear_cache() -> None:
    _load_cached.cache_clear()


# ---------------------------------------------------------------------------
# Per-dataset task builders
# ---------------------------------------------------------------------------
def build_tasks(
    dataset: str,
    shift: str,
    *,
    available: Optional[Sequence[str]] = None,
    subjects: Optional[Sequence[int]] = None,
    max_windows: Optional[int] = None,
) -> List[TaskSpec]:
    dataset = dataset.lower()
    if dataset == "uci_har":
        specs = _uci_tasks(shift)
    elif dataset == "hhar":
        specs = _hhar_tasks(shift, available)
    elif dataset == "pamap2":
        specs = _pamap2_tasks(shift, subjects)
    elif dataset == "wisdm":
        specs = _wisdm_tasks(shift, subjects)
    elif dataset == "opportunity":
        specs = _opportunity_tasks(shift, subjects)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")
    if max_windows is not None:
        specs = [_cap_windows(spec, int(max_windows)) for spec in specs]
    return specs


def _cap_windows(spec: TaskSpec, max_windows: int) -> TaskSpec:
    """Ask the loaders to subsample deterministically before allocating tensors."""
    import dataclasses

    def capped(request: DomainRequest) -> DomainRequest:
        loader = dict(request.loader)
        loader["max_windows_per_domain"] = max_windows
        return dataclasses.replace(request, loader=loader)

    return dataclasses.replace(spec, source=capped(spec.source), target=capped(spec.target))


def _uci_tasks(shift: str) -> List[TaskSpec]:
    if shift != "cross_subject":
        raise ValueError("uci_har only supports cross_subject")
    early = tuple(f"subject:{s:02d}" for s in range(1, 21))
    late = tuple(f"subject:{s:02d}" for s in range(21, 31))
    return [
        TaskSpec("uci_har/early_to_late", "uci_har", shift, DomainRequest(early), DomainRequest(late)),
        TaskSpec("uci_har/late_to_early", "uci_har", shift, DomainRequest(late), DomainRequest(early)),
    ]


HHAR_PHONE_MODELS: Tuple[str, ...] = ("s3", "s4", "nexus4")


def _hhar_tasks(shift: str, available: Optional[Sequence[str]]) -> List[TaskSpec]:
    if shift != "cross_device":
        raise ValueError("hhar currently supports cross_device only (subject is metadata)")
    keys = list(available or [f"device:{m}" for m in HHAR_PHONE_MODELS])
    tasks = []
    for src, tgt in (("device:s3", "device:nexus4"), ("device:nexus4", "device:s3")):
        if src in keys and tgt in keys:
            tasks.append(
                TaskSpec(
                    f"hhar/{src.split(':')[1]}_to_{tgt.split(':')[1]}",
                    "hhar",
                    shift,
                    DomainRequest((src,)),
                    DomainRequest((tgt,)),
                )
            )
    return tasks


def _pamap2_tasks(shift: str, subjects: Optional[Sequence[int]]) -> List[TaskSpec]:
    if shift == "cross_placement":
        pairs = (("wrist", "ankle"), ("ankle", "wrist"), ("chest", "wrist"))
        keys = _pamap2_keys(subjects)
        return [
            TaskSpec(
                f"pamap2/{src}_to_{tgt}",
                "pamap2",
                shift,
                DomainRequest(keys, {"sensors": (src,)}),
                DomainRequest(keys, {"sensors": (tgt,)}),
            )
            for src, tgt in pairs
        ]
    if shift == "cross_subject":
        early = [101, 102, 103, 104, 105]
        late = [106, 107, 108, 109]
        return [
            TaskSpec(
                "pamap2/early_to_late",
                "pamap2",
                shift,
                DomainRequest(tuple(f"subject:{s}" for s in early), {"sensors": ("wrist",)}),
                DomainRequest(tuple(f"subject:{s}" for s in late), {"sensors": ("wrist",)}),
            )
        ]
    raise ValueError("pamap2 supports cross_placement and cross_subject")


def _pamap2_keys(subjects: Optional[Sequence[int]]) -> Tuple[str, ...]:
    if subjects is None:
        return tuple(f"subject:{s}" for s in range(101, 110))
    return tuple(f"subject:{s}" for s in subjects)


def _wisdm_tasks(shift: str, subjects: Optional[Sequence[int]]) -> List[TaskSpec]:
    if shift == "cross_device":
        return [
            TaskSpec("wisdm/phone_to_watch", "wisdm", shift, DomainRequest(("device:phone",)), DomainRequest(("device:watch",))),
            TaskSpec("wisdm/watch_to_phone", "wisdm", shift, DomainRequest(("device:watch",)), DomainRequest(("device:phone",))),
        ]
    if shift == "cross_subject":
        early = list(range(1600, 1626))
        late = list(range(1626, 1651))
        return [
            TaskSpec(
                "wisdm/early_to_late",
                "wisdm",
                shift,
                DomainRequest(("device:phone",), {"subjects": early}),
                DomainRequest(("device:phone",), {"subjects": late}),
            ),
            TaskSpec(
                "wisdm/late_to_early",
                "wisdm",
                shift,
                DomainRequest(("device:phone",), {"subjects": late}),
                DomainRequest(("device:phone",), {"subjects": early}),
            ),
        ]
    raise ValueError("wisdm supports cross_device and cross_subject")


def _opportunity_tasks(shift: str, subjects: Optional[Sequence[int]]) -> List[TaskSpec]:
    if shift == "cross_subject":
        return [
            TaskSpec("opportunity/S12_to_S34", "opportunity", shift,
                     DomainRequest(("subject:S1", "subject:S2")),
                     DomainRequest(("subject:S3", "subject:S4"))),
            TaskSpec("opportunity/S34_to_S12", "opportunity", shift,
                     DomainRequest(("subject:S3", "subject:S4")),
                     DomainRequest(("subject:S1", "subject:S2"))),
        ]
    if shift == "cross_session":
        adl = ("ADL1", "ADL2", "ADL3")
        late = ("ADL4", "ADL5")
        return [
            TaskSpec("opportunity/adl123_to_adl45", "opportunity", shift,
                     DomainRequest(tuple(f"subject:S{s}" for s in (1, 2, 3, 4)), {"sessions": adl}),
                     DomainRequest(tuple(f"subject:S{s}" for s in (1, 2, 3, 4)), {"sessions": late})),
        ]
    raise ValueError("opportunity supports cross_subject and cross_session")
