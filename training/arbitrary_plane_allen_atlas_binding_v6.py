"""Pinned Allen atlas inputs for standalone arbitrary-plane v6 training.

This module is deliberately limited to deterministic atlas I/O and geometry.
It never imports a model, checkpoint, feature cache, candidate bank, training
bank, prediction, or pseudolabel implementation.

There are two verification levels:

* :func:`verify_bound_allen_atlas_v6` authenticates already-decoded in-memory
  objects.  It is suitable at a run/resume boundary and performs no NRRD I/O
  or catalogue reconstruction.
* :func:`replay_allen_atlas_binding_v6` independently repeats the complete
  raw-NRRD -> float32 atlas -> support index -> 98,304-cell catalogue path.
  It is intentionally slower and is intended for pre-run or post-run audit.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import scipy

import training.arbitrary_plane_catalogue_v3 as catalogue_v3
import training.arbitrary_plane_support as support_v1
from training.arbitrary_plane_catalogue_binding_v3 import (
    verify_catalogue_binding_v3,
)


ALLEN_ATLAS_BINDING_V6_SCHEMA = "anatomy-tracker.allen-atlas-binding/v6"
ALLEN_ATLAS_BUNDLE_V6_SCHEMA = "anatomy-tracker.bound-allen-atlas-inputs/v6"
DETERMINISTIC_SOURCE_FILES_V6 = (
    "training/arbitrary_plane_allen_atlas_binding_v6.py",
    "training/arbitrary_plane_catalogue_binding_v3.py",
    "training/arbitrary_plane_catalogue_v3.py",
    "training/arbitrary_plane_acquisition_v2.py",
    "training/arbitrary_plane_geometry.py",
    "training/arbitrary_plane_manifest.py",
    "training/arbitrary_plane_rendered_generator.py",
    "training/arbitrary_plane_support.py",
)

ATLAS_ROOT_V6 = Path(r"I:\AnatomyTracker\data\Allen Brain Atlas 25um")
TEMPLATE_PATH_V6 = ATLAS_ROOT_V6 / "average_template_25.nrrd"
ANNOTATION_PATH_V6 = ATLAS_ROOT_V6 / "annotation_25.nrrd"

TEMPLATE_RAW_SHA256_V6 = (
    "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
)
ANNOTATION_RAW_SHA256_V6 = (
    "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
)
TEMPLATE_RAW_BYTE_COUNT_V6 = 32_998_960
ANNOTATION_RAW_BYTE_COUNT_V6 = 4_035_363
PYNRRD_VERSION_V6 = "1.1.3"
NRRD_INDEX_ORDER_V6 = "F"

ATLAS_SHAPE_AP_DV_ML_V6 = (528, 320, 456)
ATLAS_ORIGIN_AP_DV_ML_UM_V6 = (0.0, 0.0, 0.0)
ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6 = (25.0, 25.0, 25.0)
ATLAS_COORDINATE_AXES_V6 = ("AP", "DV", "ML")
ATLAS_COORDINATE_AXIS_DIRECTIONS_V6 = (
    "posterior",
    "inferior",
    "right",
)

TEMPLATE_DECODED_RECEIPT_V6 = {
    "shape": [528, 320, 456],
    "dtype": "<u2",
    "sha256": "de7d49cc0210a0c3a8bfc0ab14e34e77a6fb7447741b8344c320f788b653e648",
}
ANNOTATION_DECODED_RECEIPT_V6 = {
    "shape": [528, 320, 456],
    "dtype": "<u4",
    "sha256": "36ed7e196bc41a850071246ed1d182f6331157df3a51b1145144848874025ad3",
}
ANNOTATION_SUPPORT_ARRAY_SHA256_V6 = (
    "d691efd938b9e4694e7b939aa5e71efc68441b6a25aa31f0caea51a6d26b6c8c"
)
SUPPORT_MASK_SHA256_V6 = (
    "f890fd8f09bbc3fb547bc0fcfe5d15ddbce66d72f39e926b4c23a82224953620"
)
SUPPORT_FLOAT32_RECEIPT_V6 = {
    "shape": [528, 320, 456],
    "dtype": "<f4",
    "sha256": "c1edbff5ddd90097335c80d86878f0f808429f957ec2be6488318a43f23709b6",
}
INTENSITY_FLOAT32_RECEIPT_V6 = {
    "shape": [528, 320, 456],
    "dtype": "<f4",
    "sha256": "2f9100e628e7d7e77faa3173104e2b140188b96c8607f0fb73c922628bdaee2e",
}
ATLAS_FLOAT32_RECEIPT_V6 = {
    "shape": [2, 528, 320, 456],
    "dtype": "<f4",
    "sha256": "255c2df028b8e973305d5edb2eb26dadd8c96530c638a28aec5183be83ffd832",
}
SUPPORT_FOREGROUND_VOXEL_COUNT_V6 = 32_387_385
SUPPORT_COMPONENT_COUNT_V6 = 1
SUPPORT_INDEX_SHA256_V6 = (
    "f0e89d9e2abdacdbe3eeffb55c3bcda077d38ce3419004d2fbf88aa239f4d4cc"
)

CATALOGUE_NORMAL_COUNT_V6 = 384
CATALOGUE_OFFSET_COUNT_V6 = 16
CATALOGUE_ROLL_COUNT_V6 = 16
FULL_CATALOGUE_CELL_COUNT_V6 = 98_304
CATALOGUE_RASTER_PHYSICAL_SPAN_Y_X_UM_V6 = (12_000.0, 12_000.0)
CATALOGUE_PROFILE_V6 = {
    (96, 96): {
        "catalogue_id": (
            "a3c1dab099709692f469c656ab47f55fa8d1db6dec7f9798a74a25fac4c3801d"
        ),
        "receipt_sha256": (
            "4555cfce64031b4d7ba20b8c63b1b6185dbda2e4a90b40eb19a9c1ef01dee536"
        ),
    },
    (160, 160): {
        "catalogue_id": (
            "efeba8ed231ae61e744a8b69f31ab446aa69da4e2a03b53db2b9e0ed91b6c197"
        ),
        "receipt_sha256": (
            "ab189d84a9c397eafbc82e4b4245f57e3c4b82d617b53562d7271b331637a143"
        ),
    },
}
ALLEN_ATLAS_BINDING_RECEIPT_BY_RASTER_V6 = {
    (96, 96): "fd1dc8901f9554ad28f592603d0ad794d346179e39a6a70a6b7492dc5418901c",
    (160, 160): "8963e58824721e7748ac448996da02170247c070d477ef1d8fd23ca8d5185ebc",
}

EXPECTED_NUMPY_VERSION_V6 = "2.4.4"
EXPECTED_SCIPY_VERSION_V6 = "1.17.1"
EXPECTED_QHULL_PROVENANCE_V6 = {
    "implementation": "scipy.spatial.ConvexHull",
    "qhull_options": "Qx",
    "numpy_version": EXPECTED_NUMPY_VERSION_V6,
    "scipy_version": EXPECTED_SCIPY_VERSION_V6,
    "qhull_version": "8.0.2 (2020.2.r 2020/08/31)",
    "qhull_extension_sha256": (
        "fa4f046909d1d7366b0c2da8209a41f01981565407b762a4decc43f731691622"
    ),
}

_EXPECTED_COMMON_HEADER_V6 = {
    "dimension": 3,
    "space": "left-posterior-superior",
    "sizes": [528, 320, 456],
    "space_directions": [
        [25.0, 0.0, 0.0],
        [0.0, 25.0, 0.0],
        [0.0, 0.0, 25.0],
    ],
    "kinds": ["domain", "domain", "domain"],
    "endian": "little",
    "encoding": "gzip",
    "space_origin": [0.0, 0.0, 0.0],
}
_EXPECTED_TEMPLATE_HEADER_V6 = {
    **_EXPECTED_COMMON_HEADER_V6,
    "type": "unsigned short",
}
_EXPECTED_ANNOTATION_HEADER_V6 = {
    **_EXPECTED_COMMON_HEADER_V6,
    "type": "unsigned int",
}

_BUNDLE_TOKEN = object()


def _plain(value):
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json(value) -> str:
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _with_receipt(payload: Mapping[str, object]) -> dict[str, object]:
    value = _plain(payload)
    return {**value, "receipt_sha256": _hash_json(value)}


def _payload(value: Mapping[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "receipt_sha256"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _i_file(path: Path, *, expected_byte_count: int, expected_sha256: str) -> Path:
    resolved = path.resolve()
    if os.path.splitdrive(str(resolved))[0].upper() != "I:":
        raise ValueError("pinned Allen atlas inputs must remain on I:")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if (
        resolved.stat().st_size != expected_byte_count
        or _file_sha256(resolved) != expected_sha256
    ):
        raise ValueError("pinned Allen atlas raw byte count or SHA-256 differs")
    return resolved


def _header_binding(header: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": str(header.get("type")),
        "dimension": int(header.get("dimension")),
        "space": str(header.get("space")),
        "sizes": np.asarray(header.get("sizes"), dtype=np.int64).tolist(),
        "space_directions": np.asarray(
            header.get("space directions"), dtype=np.float64
        ).tolist(),
        "kinds": [str(value) for value in header.get("kinds", ())],
        "endian": str(header.get("endian")),
        "encoding": str(header.get("encoding")),
        "space_origin": np.asarray(
            header.get("space origin"), dtype=np.float64
        ).tolist(),
    }


def _verify_runtime_dependencies() -> None:
    observed = {
        "pynrrd": importlib.metadata.version("pynrrd"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    expected = {
        "pynrrd": PYNRRD_VERSION_V6,
        "numpy": EXPECTED_NUMPY_VERSION_V6,
        "scipy": EXPECTED_SCIPY_VERSION_V6,
    }
    if observed != expected:
        raise RuntimeError(
            f"Allen v6 decoder/geometry runtime differs: {observed!r} != {expected!r}"
        )


def _decode_and_preprocess_allen_v6() -> tuple[np.ndarray, np.ndarray]:
    import nrrd

    _verify_runtime_dependencies()
    template_path = _i_file(
        TEMPLATE_PATH_V6,
        expected_byte_count=TEMPLATE_RAW_BYTE_COUNT_V6,
        expected_sha256=TEMPLATE_RAW_SHA256_V6,
    )
    annotation_path = _i_file(
        ANNOTATION_PATH_V6,
        expected_byte_count=ANNOTATION_RAW_BYTE_COUNT_V6,
        expected_sha256=ANNOTATION_RAW_SHA256_V6,
    )
    template, template_header = nrrd.read(
        str(template_path), index_order=NRRD_INDEX_ORDER_V6
    )
    annotation, annotation_header = nrrd.read(
        str(annotation_path), index_order=NRRD_INDEX_ORDER_V6
    )
    if (
        _header_binding(template_header) != _EXPECTED_TEMPLATE_HEADER_V6
        or _header_binding(annotation_header) != _EXPECTED_ANNOTATION_HEADER_V6
        or _array_receipt(template) != TEMPLATE_DECODED_RECEIPT_V6
        or _array_receipt(annotation) != ANNOTATION_DECODED_RECEIPT_V6
    ):
        raise RuntimeError("pinned Allen NRRD headers or decoded arrays differ")

    support_mask = annotation != 0
    observed_quantiles = np.quantile(template[support_mask], (0.01, 0.99))
    if tuple(float(value) for value in observed_quantiles) != (9.0, 273.0):
        raise RuntimeError("Allen in-support q01/q99 differ from 9/273")
    intensity = np.clip(
        (template.astype(np.float32) - np.float32(9.0)) / np.float32(264.0),
        np.float32(0.0),
        np.float32(1.0),
    )
    intensity[~support_mask] = np.float32(0.0)
    support_float32 = support_mask.astype(np.float32)
    atlas = np.ascontiguousarray(
        np.stack((intensity, support_float32), axis=0), dtype=np.float32
    )
    if (
        _array_receipt(intensity) != INTENSITY_FLOAT32_RECEIPT_V6
        or _array_receipt(support_float32) != SUPPORT_FLOAT32_RECEIPT_V6
        or _array_receipt(atlas) != ATLAS_FLOAT32_RECEIPT_V6
        or int(support_mask.sum()) != SUPPORT_FOREGROUND_VOXEL_COUNT_V6
        or support_v1._mask_sha256(support_mask) != SUPPORT_MASK_SHA256_V6
        or np.any(atlas[:, ~support_mask] != np.float32(0.0))
    ):
        raise RuntimeError("exact Allen v6 float32 preprocessing did not replay")
    return atlas, annotation


def _build_support_index_v6(annotation: np.ndarray) -> dict[str, object]:
    support = support_v1.build_annotation_support_index(
        annotation,
        atlas_id="Allen CCFv3",
        atlas_version="2017 25um",
        source_uri=str(ANNOTATION_PATH_V6.resolve()),
        source_sha256=ANNOTATION_RAW_SHA256_V6,
        source_entity_type="atlas-annotation",
        voxel_size_um=ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
        origin_um=ATLAS_ORIGIN_AP_DV_ML_UM_V6,
        coordinate_axes=ATLAS_COORDINATE_AXES_V6,
        coordinate_axis_directions=ATLAS_COORDINATE_AXIS_DIRECTIONS_V6,
    )
    support_v1.verify_annotation_support_index(support)
    if (
        support.get("support_index_sha256") != SUPPORT_INDEX_SHA256_V6
        or support.get("support_mask_sha256") != SUPPORT_MASK_SHA256_V6
        or support.get("foreground_voxel_count")
        != SUPPORT_FOREGROUND_VOXEL_COUNT_V6
        or support.get("component_count") != SUPPORT_COMPONENT_COUNT_V6
        or support.get("annotation_shape") != list(ATLAS_SHAPE_AP_DV_ML_V6)
        or support.get("source", {}).get("annotation_array_sha256")
        != ANNOTATION_SUPPORT_ARRAY_SHA256_V6
        or support.get("convex_hull_dependency")
        != EXPECTED_QHULL_PROVENANCE_V6
    ):
        raise RuntimeError("Allen annotation did not reproduce the pinned support index")
    return support


def _catalogue_shape(raster_shape_h_w) -> tuple[int, int]:
    shape = tuple(raster_shape_h_w)
    if shape not in CATALOGUE_PROFILE_V6:
        raise ValueError("Allen v6 catalogue raster must be exactly 96x96 or 160x160")
    return shape


def _build_catalogue_v6(
    support_index: Mapping[str, object], raster_shape_h_w
) -> dict[str, object]:
    shape = _catalogue_shape(raster_shape_h_w)
    catalogue = catalogue_v3.make_arbitrary_plane_catalogue_v3(
        None,
        ATLAS_ORIGIN_AP_DV_ML_UM_V6,
        ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6,
        support_index=support_index,
        normal_count=CATALOGUE_NORMAL_COUNT_V6,
        offset_count=CATALOGUE_OFFSET_COUNT_V6,
        roll_count=CATALOGUE_ROLL_COUNT_V6,
        raster_shape_h_w=shape,
        raster_physical_span_y_x_um=CATALOGUE_RASTER_PHYSICAL_SPAN_Y_X_UM_V6,
    )
    verify_catalogue_binding_v3(catalogue)
    profile = CATALOGUE_PROFILE_V6[shape]
    geometry = catalogue.get("support_geometry", {})
    if (
        catalogue.get("catalogue_id") != profile["catalogue_id"]
        or catalogue.get("receipt_sha256") != profile["receipt_sha256"]
        or catalogue.get("counts", {}).get("cell_count")
        != FULL_CATALOGUE_CELL_COUNT_V6
        or catalogue.get("counts", {}).get("normal_count")
        != CATALOGUE_NORMAL_COUNT_V6
        or catalogue.get("counts", {}).get("offset_count_per_normal")
        != CATALOGUE_OFFSET_COUNT_V6
        or catalogue.get("counts", {}).get("roll_count")
        != CATALOGUE_ROLL_COUNT_V6
        or geometry.get("support_index_sha256") != SUPPORT_INDEX_SHA256_V6
        or geometry.get("support_mask_receipt")
        != {
            "shape": list(ATLAS_SHAPE_AP_DV_ML_V6),
            "dtype": np.dtype(bool).str,
            "sha256": SUPPORT_MASK_SHA256_V6,
            "source": "authenticated-support-index",
        }
        or geometry.get("support_voxel_count")
        != SUPPORT_FOREGROUND_VOXEL_COUNT_V6
    ):
        raise RuntimeError("full Allen v6 catalogue did not reproduce its pinned receipt")
    return catalogue


def _raw_source_binding_v6() -> dict[str, object]:
    return {
        "template": {
            "path": str(TEMPLATE_PATH_V6.resolve()),
            "role": "Allen CCFv3 2017 25um average template raw source",
            "byte_count": TEMPLATE_RAW_BYTE_COUNT_V6,
            "raw_sha256": TEMPLATE_RAW_SHA256_V6,
        },
        "annotation": {
            "path": str(ANNOTATION_PATH_V6.resolve()),
            "role": "Allen CCFv3 2017 25um annotation raw source",
            "byte_count": ANNOTATION_RAW_BYTE_COUNT_V6,
            "raw_sha256": ANNOTATION_RAW_SHA256_V6,
        },
    }


def verify_pinned_allen_raw_sources_v6(
    binding: Mapping[str, object] | None = None,
) -> bool:
    """Hash the two current raw files without decoding or rebuilding geometry."""
    if binding is not None:
        supplied = _plain(binding)
        if (
            supplied.get("schema_version") != ALLEN_ATLAS_BINDING_V6_SCHEMA
            or supplied.get("receipt_sha256") != _hash_json(_payload(supplied))
            or supplied.get("raw_sources") != _raw_source_binding_v6()
        ):
            raise ValueError("Allen v6 binding has different raw-source identities")
    _i_file(
        TEMPLATE_PATH_V6,
        expected_byte_count=TEMPLATE_RAW_BYTE_COUNT_V6,
        expected_sha256=TEMPLATE_RAW_SHA256_V6,
    )
    _i_file(
        ANNOTATION_PATH_V6,
        expected_byte_count=ANNOTATION_RAW_BYTE_COUNT_V6,
        expected_sha256=ANNOTATION_RAW_SHA256_V6,
    )
    return True


def _binding_v6(
    support_index: Mapping[str, object], catalogue: Mapping[str, object]
) -> dict[str, object]:
    shape = _catalogue_shape(
        catalogue.get("support_geometry", {}).get("raster_shape_h_w", ())
    )
    profile = CATALOGUE_PROFILE_V6[shape]
    binding = _with_receipt(
        {
            "schema_version": ALLEN_ATLAS_BINDING_V6_SCHEMA,
            "atlas_identity": {
                "id": "Allen CCFv3",
                "version": "2017 25um",
            },
            "raw_sources": _raw_source_binding_v6(),
            "decoder": {
                "distribution": "pynrrd",
                "version": PYNRRD_VERSION_V6,
                "call": "nrrd.read",
                "index_order": NRRD_INDEX_ORDER_V6,
                "template_header": _EXPECTED_TEMPLATE_HEADER_V6,
                "annotation_header": _EXPECTED_ANNOTATION_HEADER_V6,
                "template_decoded_receipt": TEMPLATE_DECODED_RECEIPT_V6,
                "annotation_decoded_receipt": ANNOTATION_DECODED_RECEIPT_V6,
            },
            "preprocessing": {
                "implementation_dtype": "float32",
                "axis_order": "channel,AP,DV,ML",
                "intensity_channel": {
                    "source": "decoded average_template_25.nrrd",
                    "quantile_population": "decoded annotation != 0",
                    "quantile_method": "numpy.quantile default linear method",
                    "observed_q01": 9.0,
                    "observed_q99": 273.0,
                    "transform": "clip((float32(x)-float32(9))/float32(264),0,1)",
                    "exterior": "exact float32 zero where decoded annotation == 0",
                    "receipt": INTENSITY_FLOAT32_RECEIPT_V6,
                },
                "support_channel": {
                    "transform": "float32(decoded annotation != 0)",
                    "receipt": SUPPORT_FLOAT32_RECEIPT_V6,
                },
                "channel_order": ["intensity", "annotation-support"],
            },
            "decoded_atlas_receipt": ATLAS_FLOAT32_RECEIPT_V6,
            "geometry": {
                "spatial_shape_ap_dv_ml": list(ATLAS_SHAPE_AP_DV_ML_V6),
                "origin_ap_dv_ml_um": list(ATLAS_ORIGIN_AP_DV_ML_UM_V6),
                "voxel_size_ap_dv_ml_um": list(
                    ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6
                ),
                "coordinate_axes": list(ATLAS_COORDINATE_AXES_V6),
                "coordinate_axis_directions": list(
                    ATLAS_COORDINATE_AXIS_DIRECTIONS_V6
                ),
            },
            "support": {
                "definition": "decoded annotation != 0",
                "annotation_array_sha256": ANNOTATION_SUPPORT_ARRAY_SHA256_V6,
                "support_mask_sha256": SUPPORT_MASK_SHA256_V6,
                "foreground_voxel_count": SUPPORT_FOREGROUND_VOXEL_COUNT_V6,
                "component_count": SUPPORT_COMPONENT_COUNT_V6,
                "support_index_schema_version": support_index["schema_version"],
                "support_index_algorithm": support_index["algorithm"],
                "support_index_sha256": SUPPORT_INDEX_SHA256_V6,
                "convex_hull_dependency": _plain(
                    support_index["convex_hull_dependency"]
                ),
            },
            "catalogue": {
                "normal_count": CATALOGUE_NORMAL_COUNT_V6,
                "offset_count": CATALOGUE_OFFSET_COUNT_V6,
                "roll_count": CATALOGUE_ROLL_COUNT_V6,
                "cell_count": FULL_CATALOGUE_CELL_COUNT_V6,
                "raster_shape_h_w": list(shape),
                "raster_physical_span_y_x_um": list(
                    CATALOGUE_RASTER_PHYSICAL_SPAN_Y_X_UM_V6
                ),
                "catalogue_id": profile["catalogue_id"],
                "catalogue_receipt_sha256": profile["receipt_sha256"],
                "support_index_sha256": SUPPORT_INDEX_SHA256_V6,
            },
            "runtime_dependencies": {
                "pynrrd": PYNRRD_VERSION_V6,
                "numpy": EXPECTED_NUMPY_VERSION_V6,
                "scipy": EXPECTED_SCIPY_VERSION_V6,
                "qhull": EXPECTED_QHULL_PROVENANCE_V6,
            },
            "prior_model_weight_dependencies": [],
            "prior_feature_dependencies": [],
            "prior_pseudolabel_dependencies": [],
        }
    )
    if binding["receipt_sha256"] != ALLEN_ATLAS_BINDING_RECEIPT_BY_RASTER_V6[shape]:
        raise RuntimeError("Allen v6 atlas-binding receipt drifted")
    return binding


class BoundAllenAtlasInputsV6:
    """Capability containing one exact decoded atlas/support/catalogue tuple."""

    __slots__ = (
        "_token",
        "_atlas_volume_float32",
        "_support_index",
        "_catalogue",
        "_binding",
        "_frozen_binding",
    )

    def __init__(
        self,
        token,
        atlas_volume_float32: np.ndarray,
        support_index: Mapping[str, object],
        catalogue: Mapping[str, object],
        binding: Mapping[str, object],
    ) -> None:
        if token is not _BUNDLE_TOKEN:
            raise TypeError("Allen v6 bundles can only be issued by this module")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_atlas_volume_float32", atlas_volume_float32)
        object.__setattr__(self, "_support_index", support_index)
        object.__setattr__(self, "_catalogue", catalogue)
        object.__setattr__(self, "_binding", _plain(binding))
        object.__setattr__(self, "_frozen_binding", _freeze(_plain(binding)))

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("bound Allen v6 bundle fields are immutable")
        object.__setattr__(self, name, value)

    @property
    def atlas_volume_float32(self) -> np.ndarray:
        return self._atlas_volume_float32

    @property
    def support_index(self) -> Mapping[str, object]:
        return MappingProxyType(self._support_index)

    @property
    def catalogue(self) -> Mapping[str, object]:
        return MappingProxyType(self._catalogue)

    @property
    def binding(self) -> Mapping[str, object]:
        return self._frozen_binding


def _issue_bundle_v6(
    atlas_volume_float32: np.ndarray,
    support_index: Mapping[str, object],
    catalogue: Mapping[str, object],
    binding: Mapping[str, object],
) -> BoundAllenAtlasInputsV6:
    return BoundAllenAtlasInputsV6(
        _BUNDLE_TOKEN,
        atlas_volume_float32,
        support_index,
        catalogue,
        binding,
    )


def verify_bound_allen_atlas_v6(bundle: BoundAllenAtlasInputsV6) -> bool:
    """Cheaply authenticate decoded objects without reading the NRRDs."""
    if not isinstance(bundle, BoundAllenAtlasInputsV6) or bundle._token is not _BUNDLE_TOKEN:
        raise ValueError("object is not a module-issued Allen v6 bundle")
    atlas = bundle._atlas_volume_float32
    support = bundle._support_index
    catalogue = bundle._catalogue
    binding = bundle._binding
    if (
        not isinstance(atlas, np.ndarray)
        or not atlas.flags.c_contiguous
        or _array_receipt(atlas) != ATLAS_FLOAT32_RECEIPT_V6
        or not np.isfinite(atlas).all()
        or np.any((atlas[0] < np.float32(0.0)) | (atlas[0] > np.float32(1.0)))
        or np.any((atlas[1] != np.float32(0.0)) & (atlas[1] != np.float32(1.0)))
    ):
        raise ValueError("bound Allen v6 decoded atlas is not exact")
    support_mask = atlas[1] != np.float32(0.0)
    if (
        int(support_mask.sum()) != SUPPORT_FOREGROUND_VOXEL_COUNT_V6
        or support_v1._mask_sha256(support_mask) != SUPPORT_MASK_SHA256_V6
        or np.any(atlas[:, ~support_mask] != np.float32(0.0))
    ):
        raise ValueError("bound Allen v6 support channel or exterior differs")
    support_v1.verify_annotation_support_index(support)
    if (
        support.get("support_index_sha256") != SUPPORT_INDEX_SHA256_V6
        or support.get("support_mask_sha256") != SUPPORT_MASK_SHA256_V6
        or support.get("foreground_voxel_count")
        != SUPPORT_FOREGROUND_VOXEL_COUNT_V6
        or support.get("convex_hull_dependency")
        != EXPECTED_QHULL_PROVENANCE_V6
    ):
        raise ValueError("bound Allen v6 support index differs")
    verify_catalogue_binding_v3(catalogue)
    shape = _catalogue_shape(
        catalogue.get("support_geometry", {}).get("raster_shape_h_w", ())
    )
    profile = CATALOGUE_PROFILE_V6[shape]
    geometry = catalogue["support_geometry"]
    if (
        catalogue.get("catalogue_id") != profile["catalogue_id"]
        or catalogue.get("receipt_sha256") != profile["receipt_sha256"]
        or catalogue.get("counts", {}).get("cell_count")
        != FULL_CATALOGUE_CELL_COUNT_V6
        or geometry.get("support_index_sha256") != SUPPORT_INDEX_SHA256_V6
        or geometry.get("support_mask_receipt", {}).get("sha256")
        != SUPPORT_MASK_SHA256_V6
        or geometry.get("support_voxel_count")
        != SUPPORT_FOREGROUND_VOXEL_COUNT_V6
        or tuple(geometry.get("origin_ap_dv_ml_um", ()))
        != ATLAS_ORIGIN_AP_DV_ML_UM_V6
        or tuple(geometry.get("voxel_size_ap_dv_ml_um", ()))
        != ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6
    ):
        raise ValueError("bound Allen v6 complete catalogue differs")
    expected_binding = _binding_v6(support, catalogue)
    if (
        binding.get("schema_version") != ALLEN_ATLAS_BINDING_V6_SCHEMA
        or binding.get("receipt_sha256") != _hash_json(_payload(binding))
        or binding != expected_binding
        or any(
            binding.get(name) != []
            for name in (
                "prior_model_weight_dependencies",
                "prior_feature_dependencies",
                "prior_pseudolabel_dependencies",
            )
        )
    ):
        raise ValueError("bound Allen v6 provenance binding differs")
    return True


def resolve_bound_allen_atlas_v6(
    bundle: BoundAllenAtlasInputsV6,
) -> Mapping[str, object]:
    """Return verified bundle members for a receipt-bound runner."""
    verify_bound_allen_atlas_v6(bundle)
    return MappingProxyType(
        {
            "schema_version": ALLEN_ATLAS_BUNDLE_V6_SCHEMA,
            "atlas_volume_float32": bundle._atlas_volume_float32,
            # These remain the exact verified objects rather than copies so a
            # runner can serialize them without duplicating the 588 MiB atlas
            # or encountering an unpicklable mappingproxy.  Reverify after any
            # external boundary; mutation is receipt-detectable.
            "support_index": bundle._support_index,
            "catalogue": bundle._catalogue,
            "binding": bundle._frozen_binding,
        }
    )


def restore_bound_allen_atlas_v6(
    *,
    atlas_volume_float32: np.ndarray,
    support_index: Mapping[str, object],
    catalogue: Mapping[str, object],
    binding: Mapping[str, object],
) -> BoundAllenAtlasInputsV6:
    """Restore and cheaply authenticate previously frozen run inputs."""
    bundle = _issue_bundle_v6(
        atlas_volume_float32, support_index, catalogue, binding
    )
    verify_bound_allen_atlas_v6(bundle)
    return bundle


def prepare_bound_allen_atlas_v6(
    *, raster_shape_h_w: tuple[int, int]
) -> BoundAllenAtlasInputsV6:
    """Decode pinned I:-drive Allen inputs and build one exact full catalogue."""
    shape = _catalogue_shape(raster_shape_h_w)
    atlas, annotation = _decode_and_preprocess_allen_v6()
    support = _build_support_index_v6(annotation)
    catalogue = _build_catalogue_v6(support, shape)
    binding = _binding_v6(support, catalogue)
    bundle = _issue_bundle_v6(atlas, support, catalogue, binding)
    verify_bound_allen_atlas_v6(bundle)
    return bundle


def replay_allen_atlas_binding_v6(
    binding: Mapping[str, object],
) -> BoundAllenAtlasInputsV6:
    """Slowly reproduce raw decode, support, catalogue, and the exact binding."""
    supplied = _plain(binding)
    if (
        supplied.get("schema_version") != ALLEN_ATLAS_BINDING_V6_SCHEMA
        or supplied.get("receipt_sha256") != _hash_json(_payload(supplied))
    ):
        raise ValueError("Allen v6 binding receipt is invalid before replay")
    shape = _catalogue_shape(
        supplied.get("catalogue", {}).get("raster_shape_h_w", ())
    )
    replay = prepare_bound_allen_atlas_v6(raster_shape_h_w=shape)
    if _plain(replay.binding) != supplied:
        raise ValueError("raw Allen inputs did not independently reproduce the binding")
    return replay


__all__ = [
    "ALLEN_ATLAS_BINDING_V6_SCHEMA",
    "ALLEN_ATLAS_BINDING_RECEIPT_BY_RASTER_V6",
    "ALLEN_ATLAS_BUNDLE_V6_SCHEMA",
    "ANNOTATION_PATH_V6",
    "ANNOTATION_RAW_SHA256_V6",
    "ATLAS_FLOAT32_RECEIPT_V6",
    "ATLAS_ORIGIN_AP_DV_ML_UM_V6",
    "ATLAS_ROOT_V6",
    "ATLAS_SHAPE_AP_DV_ML_V6",
    "ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6",
    "BoundAllenAtlasInputsV6",
    "CATALOGUE_PROFILE_V6",
    "DETERMINISTIC_SOURCE_FILES_V6",
    "FULL_CATALOGUE_CELL_COUNT_V6",
    "NRRD_INDEX_ORDER_V6",
    "PYNRRD_VERSION_V6",
    "SUPPORT_FOREGROUND_VOXEL_COUNT_V6",
    "SUPPORT_INDEX_SHA256_V6",
    "SUPPORT_MASK_SHA256_V6",
    "TEMPLATE_PATH_V6",
    "TEMPLATE_RAW_SHA256_V6",
    "prepare_bound_allen_atlas_v6",
    "replay_allen_atlas_binding_v6",
    "resolve_bound_allen_atlas_v6",
    "restore_bound_allen_atlas_v6",
    "verify_bound_allen_atlas_v6",
    "verify_pinned_allen_raw_sources_v6",
]
