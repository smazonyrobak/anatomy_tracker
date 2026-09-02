"""One-time verified complete-catalogue binding for the v6 cascade."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

from training.arbitrary_plane_inference_v3 import verify_catalogue_binding_v3


COMPLETE_CATALOGUE_RUNTIME_V6_SCHEMA = (
    "anatomy-tracker.complete-catalogue-runtime/v6"
)
_RUNTIME_TOKEN = object()
_BATCH_TOKEN = object()
_TENSOR_NAMES = (
    "cell_id",
    "cell_states",
    "cell_log_mass",
    "representation_log_weight",
    "representation_to_canonical_raster_affine",
    "support_origin_ap_dv_ml_um",
)
_BATCHED_TENSOR_NAMES = _TENSOR_NAMES[1:5]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and not (set(value) - set("0123456789abcdef"))
    )


def _tensor_handle_state(value: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        str(value.dtype),
        str(value.device),
        value.untyped_storage().data_ptr(),
        value._version,
    )


class BoundCompleteCatalogueBatchV6:
    """Opaque batch capability issued only by a verified runtime."""

    __slots__ = ("_token", "_runtime", "_batch_size")

    def __init__(self, runtime: "CompleteCatalogueRuntimeV6", batch_size: int):
        self._token = _BATCH_TOKEN
        self._runtime = runtime
        self._batch_size = batch_size

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("bound catalogue capabilities are immutable")
        object.__setattr__(self, name, value)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def binding(self) -> Mapping[str, object]:
        return self._runtime.binding

class CompleteCatalogueRuntimeV6:
    """Device-local canonical tensors authenticated once against a trusted receipt."""

    __slots__ = ("_token", "_binding", "_tensors", "_states")

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("complete catalogue runtimes are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        catalogue: Mapping[str, object],
        *,
        expected_catalogue_receipt_sha256: str,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        if not _is_sha256(expected_catalogue_receipt_sha256):
            raise ValueError("expected catalogue receipt must be a lowercase SHA-256")
        verify_catalogue_binding_v3(catalogue)
        if catalogue.get("receipt_sha256") != expected_catalogue_receipt_sha256:
            raise ValueError("catalogue does not match the trusted expected receipt")
        probe = torch.empty((), dtype=dtype)
        if not probe.is_floating_point() or probe.is_complex():
            raise ValueError("catalogue runtime dtype must be real floating point")

        counts = catalogue["counts"]
        cell_count = counts["cell_count"]
        representation_count = counts["representation_count"]
        source = catalogue["tensors"]
        expected_shapes = {
            "cell_id": (cell_count,),
            "cell_states": (1, cell_count, 12),
            "cell_log_mass": (1, cell_count),
            "representation_log_weight": (1, cell_count, representation_count),
            "representation_to_canonical_raster_affine": (
                1,
                cell_count,
                representation_count,
                2,
                3,
            ),
        }
        if (
            not isinstance(cell_count, int)
            or isinstance(cell_count, bool)
            or cell_count < 1
            or not isinstance(representation_count, int)
            or isinstance(representation_count, bool)
            or representation_count < 1
            or any(tuple(torch.as_tensor(source[name]).shape) != shape for name, shape in expected_shapes.items())
        ):
            raise ValueError("verified catalogue tensor shapes or counts are invalid")

        tensors = {
            "cell_id": torch.as_tensor(
                source["cell_id"], device=device, dtype=torch.long
            ).contiguous().clone(),
            **{
                name: torch.as_tensor(source[name], device=device, dtype=dtype)
                .contiguous()
                .clone()
                for name in _TENSOR_NAMES
                if name in _BATCHED_TENSOR_NAMES
            },
            "support_origin_ap_dv_ml_um": torch.as_tensor(
                catalogue["support_geometry"]["support_origin_ap_dv_ml_um"],
                device=device,
                dtype=dtype,
            )
            .contiguous()
            .clone(),
        }
        if any(
            not bool(torch.isfinite(tensors[name]).all())
            for name in _TENSOR_NAMES
            if name != "cell_id"
        ):
            raise ValueError("catalogue runtime tensors must remain finite")
        if tensors["support_origin_ap_dv_ml_um"].shape != (3,):
            raise ValueError("catalogue support origin must be one finite 3-vector")
        normalization_atol = max(2.0e-6, 8.0 * torch.finfo(dtype).eps)
        if (
            not torch.allclose(
                torch.logsumexp(tensors["cell_log_mass"], dim=1),
                torch.zeros(1, device=device, dtype=dtype),
                atol=normalization_atol,
                rtol=0.0,
            )
            or not torch.allclose(
                torch.logsumexp(
                    tensors["representation_log_weight"], dim=-1
                ),
                torch.zeros((1, cell_count), device=device, dtype=dtype),
                atol=normalization_atol,
                rtol=0.0,
            )
        ):
            raise ValueError("catalogue and representation log weights must have unit mass")

        self._token = _RUNTIME_TOKEN
        self._binding = MappingProxyType(
            {
                "schema_version": COMPLETE_CATALOGUE_RUNTIME_V6_SCHEMA,
                "catalogue_id": catalogue["catalogue_id"],
                "catalogue_receipt_sha256": expected_catalogue_receipt_sha256,
                "cell_count": cell_count,
                "representation_count": representation_count,
                "device": str(tensors["cell_states"].device),
                "dtype": str(dtype),
                "support_origin_ap_dv_ml_um": tuple(
                    float(value)
                    for value in catalogue["support_geometry"][
                        "support_origin_ap_dv_ml_um"
                    ]
                ),
            }
        )
        self._tensors = MappingProxyType(tensors)
        self._states = tuple(
            _tensor_handle_state(tensors[name]) for name in _TENSOR_NAMES
        )

    @property
    def binding(self) -> Mapping[str, object]:
        return self._binding

    @property
    def cell_count(self) -> int:
        return int(self._binding["cell_count"])

    @property
    def representation_count(self) -> int:
        return int(self._binding["representation_count"])

    def expand(self, batch_size: int) -> BoundCompleteCatalogueBatchV6:
        verify_complete_catalogue_runtime_v6(self)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("catalogue batch size must be a positive integer")
        return BoundCompleteCatalogueBatchV6(self, batch_size)


def make_complete_catalogue_runtime_v6(
    catalogue: Mapping[str, object],
    *,
    expected_catalogue_receipt_sha256: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CompleteCatalogueRuntimeV6:
    return CompleteCatalogueRuntimeV6(
        catalogue,
        expected_catalogue_receipt_sha256=expected_catalogue_receipt_sha256,
        device=device,
        dtype=dtype,
    )


def verify_complete_catalogue_runtime_v6(
    runtime: CompleteCatalogueRuntimeV6,
) -> bool:
    if (
        not isinstance(runtime, CompleteCatalogueRuntimeV6)
        or runtime._token is not _RUNTIME_TOKEN
        or runtime._binding.get("schema_version")
        != COMPLETE_CATALOGUE_RUNTIME_V6_SCHEMA
        or not _is_sha256(runtime._binding.get("catalogue_receipt_sha256"))
        or set(runtime._tensors) != set(_TENSOR_NAMES)
        or tuple(_tensor_handle_state(runtime._tensors[name]) for name in _TENSOR_NAMES)
        != runtime._states
    ):
        raise ValueError("complete catalogue runtime is not intact")
    return True


def verify_bound_complete_catalogue_batch_v6(
    batch: BoundCompleteCatalogueBatchV6,
    *,
    expected_runtime: CompleteCatalogueRuntimeV6 | None = None,
) -> Mapping[str, torch.Tensor]:
    if (
        not isinstance(batch, BoundCompleteCatalogueBatchV6)
        or batch._token is not _BATCH_TOKEN
        or (expected_runtime is not None and batch._runtime is not expected_runtime)
    ):
        raise ValueError("catalogue tensors were not issued by the expected runtime")
    verify_complete_catalogue_runtime_v6(batch._runtime)
    return MappingProxyType(
        {
            "cell_id": batch._runtime._tensors["cell_id"],
            "support_origin_ap_dv_ml_um": batch._runtime._tensors[
                "support_origin_ap_dv_ml_um"
            ],
            **{
                name: batch._runtime._tensors[name].expand(
                    batch._batch_size, *batch._runtime._tensors[name].shape[1:]
                )
                for name in _BATCHED_TENSOR_NAMES
            },
        }
    )


__all__ = [
    "BoundCompleteCatalogueBatchV6",
    "COMPLETE_CATALOGUE_RUNTIME_V6_SCHEMA",
    "CompleteCatalogueRuntimeV6",
    "make_complete_catalogue_runtime_v6",
    "verify_bound_complete_catalogue_batch_v6",
    "verify_complete_catalogue_runtime_v6",
]
