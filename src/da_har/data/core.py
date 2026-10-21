"""Core data contracts for the DA-HAR benchmark.

Invariants (each one is covered by ``test/test_da_core.py``):

* A :class:`DomainData` always carries an explicit name, subject/session ids,
  channel metadata and the sampling rate. Nothing is inferred implicitly.
* A window never straddles a null/transient label segment, and is only kept
  when a *single* activity covers at least ``min_label_fraction`` of it. This
  is what the legacy pipeline got wrong on PAMAP2 (transient activity ``0``
  and boundary windows were allowed in, which pushes accuracy to chance).
* :func:`fit_normalizer` only ever sees the source domain. The resulting
  statistics are then applied to both domains, so no target label or target
  statistic leaks into training.
* Channel identity is ``(role, kind, axis)``. ``placement`` is *metadata*, so
  two domains recorded at different body locations can still share the same
  slot -- which is exactly the cross-placement transfer setting. Models that
  want to exploit placement may read the metadata, but are not required to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

AXES: Tuple[str, ...] = ("x", "y", "z")

__all__ = [
    "AXES",
    "ChannelSpec",
    "DatasetBundle",
    "DomainData",
    "Normalizer",
    "align_channels",
    "concat_domains",
    "fit_normalizer",
    "make_windows",
    "macro_f1",
    "pad_to_slots",
    "resample_signal",
]


# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelSpec:
    """Identity of one scalar channel inside a domain.

    Parameters
    ----------
    role:
        Positional role of the sensor inside the domain, e.g. ``imu0``. Two
        domains that put a different physical sensor at the same role are
        comparable (that is how cross-placement is represented).
    kind:
        Physical quantity: ``acc``, ``gyro``, ``mag``, ``total_acc``,
        ``body_acc`` or ``body_gyro``.
    axis:
        One of ``x``/``y``/``z``.
    placement:
        Physical body location (metadata only): ``waist``, ``wrist``,
        ``chest``, ``ankle``, ``hip``...
    unit:
        Measurement unit, for documentation only.
    """

    role: str
    kind: str
    axis: str
    placement: str
    unit: str = "g"

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("ChannelSpec.role must be a non-empty string")
        if self.axis not in AXES:
            raise ValueError(f"Unsupported axis {self.axis!r}; expected one of {AXES}")
        if not self.kind:
            raise ValueError("ChannelSpec.kind must be a non-empty string")

    @property
    def slot(self) -> str:
        """Transferable identity (placement is intentionally excluded)."""
        return f"{self.role}:{self.kind}:{self.axis}"

    @property
    def label(self) -> str:
        """Human readable name including the physical placement."""
        return f"{self.placement}:{self.kind}:{self.axis}"


def make_imu_channels(
    role: str,
    placement: str,
    kinds: Sequence[str] = ("acc", "gyro"),
    unit: str = "g",
) -> Tuple[ChannelSpec, ...]:
    """Build the ``kinds x axes`` channel tuple for one IMU."""
    return tuple(
        ChannelSpec(role=role, kind=kind, axis=axis, placement=placement, unit=unit)
        for kind in kinds
        for axis in AXES
    )


# ---------------------------------------------------------------------------
# Domain / dataset containers
# ---------------------------------------------------------------------------
@dataclass
class DomainData:
    """A named slice of windows that share one acquisition condition."""

    name: str
    X: np.ndarray  # (N, C, T) float32
    y: np.ndarray  # (N,) int64, indices into the dataset label space
    subjects: np.ndarray  # (N,) string ids
    sessions: np.ndarray  # (N,) string ids
    channels: Tuple[ChannelSpec, ...]
    sampling_rate: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64).reshape(-1)
        self.subjects = np.asarray(self.subjects).astype(str).reshape(-1)
        self.sessions = np.asarray(self.sessions).astype(str).reshape(-1)
        self.channels = tuple(self.channels)

        if self.X.ndim != 3:
            raise ValueError(f"{self.name}: X must be (N, C, T), got {self.X.shape}")
        n, c, t = self.X.shape
        if c != len(self.channels):
            raise ValueError(
                f"{self.name}: X has {c} channels but {len(self.channels)} specs were given"
            )
        if t <= 0:
            raise ValueError(f"{self.name}: empty window length")
        if len(self.y) != n or len(self.subjects) != n or len(self.sessions) != n:
            raise ValueError(f"{self.name}: labels/subjects/sessions do not match X")
        if n and self.y.min() < 0:
            raise ValueError(f"{self.name}: negative label index")
        if not np.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError(f"{self.name}: invalid sampling_rate")

    # -- basic properties ---------------------------------------------------
    @property
    def num_windows(self) -> int:
        return int(self.X.shape[0])

    @property
    def num_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def window_size(self) -> int:
        return int(self.X.shape[2])

    @property
    def window_seconds(self) -> float:
        return self.window_size / float(self.sampling_rate)

    def slot_names(self) -> Tuple[str, ...]:
        return tuple(ch.slot for ch in self.channels)

    def class_counts(self) -> Dict[int, int]:
        vals, counts = np.unique(self.y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    # -- slicing ------------------------------------------------------------
    def subset(self, index: np.ndarray, name: Optional[str] = None) -> "DomainData":
        index = np.asarray(index)
        meta = dict(self.meta)
        meta["subset_of"] = self.name
        return DomainData(
            name=name or f"{self.name}[{len(index)}]",
            X=self.X[index],
            y=self.y[index],
            subjects=self.subjects[index],
            sessions=self.sessions[index],
            channels=self.channels,
            sampling_rate=self.sampling_rate,
            meta=meta,
        )

    def take(self, n: int, seed: int = 0) -> "DomainData":
        """Deterministic subsample (useful for smoke tests).

        The domain name is preserved so the object can still be looked up by
        its bundle key after subsampling.
        """
        if n >= self.num_windows:
            return self
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.num_windows, size=n, replace=False))
        return self.subset(idx, name=self.name)


@dataclass
class DatasetBundle:
    """All domains of one dataset, sharing one label space."""

    name: str
    domains: Dict[str, DomainData]
    label_names: Tuple[str, ...]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domains:
            raise ValueError(f"{self.name}: bundle has no domains")
        for key, dom in self.domains.items():
            if not dom.num_windows:
                continue
            if int(dom.y.max()) >= len(self.label_names):
                raise ValueError(
                    f"{self.name}/{key}: label {int(dom.y.max())} outside label space "
                    f"of size {len(self.label_names)}"
                )

    def keys(self) -> Tuple[str, ...]:
        return tuple(sorted(self.domains))

    def get(self, key: str) -> DomainData:
        if key not in self.domains:
            raise KeyError(f"{self.name}: unknown domain {key!r}. Available: {self.keys()}")
        return self.domains[key]

    def combined(self, keys: Sequence[str], name: Optional[str] = None) -> DomainData:
        return concat_domains([self.get(k) for k in keys], name=name or "+".join(keys))

    def num_windows(self) -> int:
        return sum(d.num_windows for d in self.domains.values())


def concat_domains(domains: Sequence[DomainData], name: str) -> DomainData:
    """Concatenate domains that share the same channel layout and rate."""
    if not domains:
        raise ValueError("concat_domains needs at least one domain")
    ref = domains[0]
    for dom in domains[1:]:
        if dom.channels != ref.channels:
            raise ValueError(
                f"{name}: channel mismatch between {ref.name} and {dom.name}; "
                "use align_channels/pad_to_slots first"
            )
        if not np.isclose(dom.sampling_rate, ref.sampling_rate):
            raise ValueError(f"{name}: sampling-rate mismatch")
        if dom.window_size != ref.window_size:
            raise ValueError(f"{name}: window-length mismatch")
    return DomainData(
        name=name,
        X=np.concatenate([d.X for d in domains], axis=0),
        y=np.concatenate([d.y for d in domains], axis=0),
        subjects=np.concatenate([d.subjects for d in domains], axis=0),
        sessions=np.concatenate([d.sessions for d in domains], axis=0),
        channels=ref.channels,
        sampling_rate=ref.sampling_rate,
        meta={"members": [d.name for d in domains]},
    )


# ---------------------------------------------------------------------------
# Resampling and windowing
# ---------------------------------------------------------------------------
def resample_signal(
    signal: np.ndarray, src_hz: float, dst_hz: float
) -> np.ndarray:
    """Resample ``(T, C)`` from ``src_hz`` to ``dst_hz``.

    Uses a polyphase anti-alias filter when SciPy is available, otherwise a
    box-filtered linear interpolation. Doing this *before* windowing is what
    makes a 128-sample window have the same physical duration on devices with
    different rates (the legacy HHAR loader resampled only sometimes, which
    confounded every multi-rate claim).
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"expected (T, C) signal, got {signal.shape}")
    if abs(src_hz - dst_hz) < 1e-9:
        return signal.copy()

    try:  # pragma: no cover - scipy is present in the project env
        from math import gcd

        from scipy.signal import resample_poly

        src = int(round(src_hz * 1000))
        dst = int(round(dst_hz * 1000))
        g = gcd(src, dst)
        out = resample_poly(signal, dst // g, src // g, axis=0)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        # Conservative fallback: anti-alias by moving average, then interpolate.
        t_src = np.arange(signal.shape[0], dtype=np.float64) / float(src_hz)
        n_dst = int(round((signal.shape[0] / float(src_hz)) * float(dst_hz)))
        t_dst = np.arange(n_dst, dtype=np.float64) / float(dst_hz)
        ratio = max(1, int(round(float(src_hz) / float(dst_hz))))
        if ratio > 1:
            kernel = np.ones(ratio, dtype=np.float32) / ratio
            padded = np.pad(signal, ((ratio // 2, ratio // 2), (0, 0)), mode="edge")
            signal = np.stack(
                [np.convolve(padded[:, c], kernel, mode="valid") for c in range(signal.shape[1])],
                axis=1,
            ).astype(np.float32)
        out = np.stack(
            [np.interp(t_dst, t_src, signal[:, c]) for c in range(signal.shape[1])],
            axis=1,
        )
        return out.astype(np.float32)


def make_windows(
    signal: np.ndarray,
    labels: np.ndarray,
    sampling_rate: float,
    window_seconds: float,
    stride_seconds: Optional[float] = None,
    min_label_fraction: float = 0.9,
    ignore_labels: Sequence[int] = (-1,),
    max_windows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut ``signal`` into pure windows.

    Returns
    -------
    X : (N, C, T) float32
    y : (N,) int64
    starts : (N,) int64 sample offsets
    """
    signal = np.asarray(signal, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if signal.ndim != 2:
        raise ValueError(f"expected (T, C) signal, got {signal.shape}")
    if len(labels) != signal.shape[0]:
        raise ValueError("signal and labels must have the same length")
    if not 0.0 < min_label_fraction <= 1.0:
        raise ValueError("min_label_fraction must be in (0, 1]")

    win = int(round(window_seconds * sampling_rate))
    if win <= 0:
        raise ValueError("window_seconds * sampling_rate must be >= 1 sample")
    step = int(round((stride_seconds if stride_seconds else window_seconds / 2.0) * sampling_rate))
    if step <= 0:
        raise ValueError("stride must be >= 1 sample")

    total = signal.shape[0]
    if total < win:
        return (
            np.empty((0, signal.shape[1], win), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    ignore = set(int(v) for v in ignore_labels)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    starts: List[int] = []

    for start in range(0, total - win + 1, step):
        seg = labels[start : start + win]
        valid = seg[~np.isin(seg, list(ignore))]
        if valid.size < min_label_fraction * win:
            continue
        values, counts = np.unique(valid, return_counts=True)
        majority = int(counts.argmax())
        if counts[majority] < min_label_fraction * valid.size:
            continue
        xs.append(signal[start : start + win].T)
        ys.append(int(values[majority]))
        starts.append(start)
        if max_windows is not None and len(xs) >= max_windows:
            break

    if not xs:
        return (
            np.empty((0, signal.shape[1], win), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Normalisation (source-only by construction)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray  # (C, 1)
    std: np.ndarray  # (C, 1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.reshape(-1).tolist(), "std": self.std.reshape(-1).tolist()}


def fit_normalizer(X: np.ndarray, eps: float = 1e-6) -> Normalizer:
    """Fit per-channel statistics. Call this with **source data only**."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"expected (N, C, T), got {X.shape}")
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    std = np.where(std < eps, np.float32(1.0), std)
    return Normalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


# ---------------------------------------------------------------------------
# Channel alignment across domains
# ---------------------------------------------------------------------------
def align_channels(domains: Sequence[DomainData]) -> Tuple[Tuple[str, ...], List[np.ndarray]]:
    """Return the union of channel slots and a presence mask per domain."""
    slots: List[str] = []
    seen = set()
    for dom in domains:
        for slot in dom.slot_names():
            if slot not in seen:
                seen.add(slot)
                slots.append(slot)
    masks = []
    for dom in domains:
        own = set(dom.slot_names())
        masks.append(np.asarray([s in own for s in slots], dtype=bool))
    return tuple(slots), masks


def pad_to_slots(domain: DomainData, slots: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder/pad a domain to ``slots``; absent slots are zero and masked."""
    index = {s: i for i, s in enumerate(slots)}
    X = np.zeros((domain.num_windows, len(slots), domain.window_size), dtype=np.float32)
    mask = np.zeros(len(slots), dtype=bool)
    for src, ch in enumerate(domain.channels):
        if ch.slot not in index:
            raise ValueError(f"{domain.name}: slot {ch.slot} not present in target layout")
        X[:, index[ch.slot], :] = domain.X[:, src, :]
        mask[index[ch.slot]] = True
    return X, mask


# ---------------------------------------------------------------------------
# Metrics (kept dependency-free so tests run without sklearn)
# ---------------------------------------------------------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: Optional[int] = None) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0
