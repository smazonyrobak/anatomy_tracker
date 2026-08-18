"""Frozen image-frame adapter for the DeepSlice benchmark reproduction."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


DEEPSLICE_ORIENTATION_ADAPTER_VERSION = "horizontal-raster-frame-v1"


def horizontal_image_frame_ouv(ouv: np.ndarray) -> np.ndarray:
    """Express a QuickNII plane in the horizontally reversed raster frame."""
    values = np.asarray(ouv, dtype=np.float64)
    if values.shape[-1:] != (9,):
        raise ValueError("QuickNII OUV must end in nine coordinates")
    transformed = values.copy()
    transformed[..., :3] = values[..., :3] + values[..., 3:6]
    transformed[..., 3:6] = -values[..., 3:6]
    return transformed


@contextmanager
def horizontally_flipped_deepslice_inputs(image_paths: list[Path]):
    """Create lossless one-flip inputs without changing the frozen source files."""
    paths = [Path(path) for path in image_paths]
    names = [path.name.casefold() for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("DeepSlice benchmark filenames must be unique within an experiment")
    with TemporaryDirectory(prefix="deepslice-horizontal-view-") as folder:
        root = Path(folder)
        flipped_paths = []
        for source in paths:
            destination = root / source.name
            with Image.open(source) as image:
                with image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) as flipped:
                    flipped.save(destination, format="PNG")
            flipped_paths.append(destination)
        yield flipped_paths
