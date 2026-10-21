"""Dataset loaders that expose explicit HAR *domains*.

Every loader returns a :class:`~src.da_har.data.core.DatasetBundle` whose
domains are named after the factor that defines the shift (subject, device,
placement or session).  The benchmark then combines domains into
source/target tasks instead of relying on hidden global splits.
"""

from .core import (
    ChannelSpec,
    DatasetBundle,
    DomainData,
    Normalizer,
    align_channels,
    concat_domains,
    fit_normalizer,
    make_windows,
    pad_to_slots,
    resample_signal,
)
from .registry import AVAILABLE_DATASETS, load_dataset

__all__ = [
    "AVAILABLE_DATASETS",
    "ChannelSpec",
    "DatasetBundle",
    "DomainData",
    "Normalizer",
    "align_channels",
    "concat_domains",
    "fit_normalizer",
    "load_dataset",
    "make_windows",
    "pad_to_slots",
    "resample_signal",
]
