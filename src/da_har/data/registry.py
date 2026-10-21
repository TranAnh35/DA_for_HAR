"""Dataset registry for the DA-HAR benchmark."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .core import DatasetBundle

AVAILABLE_DATASETS: Tuple[str, ...] = (
    "uci_har",
    "hhar",
    "pamap2",
    "wisdm",
    "opportunity",
)


def load_dataset(name: str, root: str = "Data", **kwargs: Any) -> DatasetBundle:
    """Load one dataset into a :class:`DatasetBundle`.

    Loader-specific keyword arguments (``sensors``, ``target_hz``, ``models``...)
    are forwarded verbatim, which is what makes cross-placement tasks possible:
    the same dataset is loaded twice with different ``sensors``.
    """
    key = name.lower()
    if key == "uci_har":
        from . import uci_har as module
    elif key == "hhar":
        from . import hhar as module
    elif key == "pamap2":
        from . import pamap2 as module
    elif key == "wisdm":
        from . import wisdm as module
    elif key == "opportunity":
        from . import opportunity as module
    else:
        raise ValueError(f"Unknown dataset {name!r}. Available: {AVAILABLE_DATASETS}")
    return module.load_bundle(root=root, **kwargs)
