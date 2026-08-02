"""BrainCodes — a compact EEG foundation model over frozen BioCodec RVQ tokens.

Public API:

* :class:`BrainCodesModel`     — load weights, tokenize, extract embeddings, build heads.
* :class:`BrainCodesClassifier`— classification head module.
* :class:`EEGPreprocessor`     — resample / filter / channel-align raw EEG.
* :func:`build_classifier`     — build a head with a freezing strategy.
* :func:`coords_from_names`    — 10-05 electrode names → unit-ball coordinates.
"""

from .api import DEFAULT_ARCH, DEFAULT_TOKENIZER, BrainCodesModel
from .backbone import BioCodecTokenizer, BrainCodes
from .channels import coords_from_names
from .classifier import BrainCodesClassifier, build_classifier
from .preprocess import EEGPreprocessor

__all__ = [
    "DEFAULT_ARCH",
    "DEFAULT_TOKENIZER",
    "BioCodecTokenizer",
    "BrainCodes",
    "BrainCodesClassifier",
    "BrainCodesModel",
    "EEGPreprocessor",
    "build_classifier",
    "coords_from_names",
]

__version__ = "0.2.0"
