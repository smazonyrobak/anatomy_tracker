"""Shared preprocessing constants for learned anatomical registration."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np


# Shared verbatim by training, release export, and packaged inference.
NATIVE_SHAPE = (320, 456)
MODEL_SHAPE = (320, 464)
PAD_X = 4
PREPROCESSING_CONTRACT_V2 = (
    "native-320x456-rawuint8-brainmask-cosinefeather4-pad4-v2"
)
FEATHER_RING_VALUES = (0.8535533905932737, 0.5, 0.14644660940672627)
MASK_CONTRACT = (
    "native-binary-brain-mask;offset=iterated-3x3-chebyshev-morphology;"
    "positive=dilate;negative=erode-against-zero;"
    "outside-feather=cosine-rings-[0.8535533905932737,0.5,0.14644660940672627]"
)
MASK_CONTRACT_SHA256 = hashlib.sha256(MASK_CONTRACT.encode("utf-8")).hexdigest()


# Feather only the mask boundary; image normalization is a separate contract.
def cosine_mask_feather(mask, *, dilate, zeros_like, where):
    """Return a backend-neutral three-ring outward cosine feather."""
    support = mask
    alpha = where(support, 1.0, zeros_like(mask))
    for value in FEATHER_RING_VALUES:
        expanded = dilate(support)
        alpha = where(expanded & ~support, value, alpha)
        support = expanded
    return alpha


def numpy_cosine_mask_feather(mask: np.ndarray) -> np.ndarray:
    mask = np.ascontiguousarray(mask, dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    return np.ascontiguousarray(
        cosine_mask_feather(
            mask,
            dilate=lambda value: cv2.dilate(
                value.astype(np.uint8), kernel, iterations=1
            ).astype(bool),
            zeros_like=lambda value: np.zeros(value.shape, np.float32),
            where=np.where,
        ),
        dtype=np.float32,
    )
