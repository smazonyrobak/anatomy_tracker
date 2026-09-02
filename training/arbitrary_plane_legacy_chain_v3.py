"""Thread-safe v3 routing through the audited v2 acquisition primitives.

The scientific acquisition, slab, subject-deformation, and section operators are
unchanged.  Their function namespaces are copied once and wired to the v3
canonical geometry function, so no imported v2 module is ever mutated.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType, ModuleType

import training.arbitrary_plane_acquisition_v2 as acquisition_v2
import training.arbitrary_plane_geometry_v3 as geometry_v3
import training.arbitrary_plane_section_processing_v2 as section_processing_v2
import training.arbitrary_plane_subject_slab_v2 as subject_slab_v2
import training.arbitrary_plane_support_resolution_v2 as support_resolution_v2
import training.arbitrary_plane_synthetic_generator_v2 as synthetic_generator_v2


LEGACY_CHAIN_ADAPTER_V3_SCHEMA = "anatomy-tracker.v2-primitive-adapter/v3"
LEGACY_CHAIN_ADAPTER_V3_ALGORITHM = (
    "immutable-function-namespace-routing-to-canonical-v3-geometry/v3"
)
_SOURCE_ROOT = Path(__file__).parent


def _clone_module(module: ModuleType, name: str, replacements: dict[str, object]):
    proxy = ModuleType(name)
    namespace = proxy.__dict__
    namespace.update(module.__dict__)
    namespace["__name__"] = name
    for key, value in tuple(module.__dict__.items()):
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            cloned = FunctionType(
                value.__code__,
                namespace,
                name=value.__name__,
                argdefs=value.__defaults__,
                closure=value.__closure__,
            )
            cloned.__kwdefaults__ = value.__kwdefaults__
            cloned.__annotations__ = value.__annotations__
            cloned.__dict__.update(value.__dict__)
            cloned.__doc__ = value.__doc__
            cloned.__module__ = name
            namespace[key] = cloned
    namespace.update(replacements)
    return proxy


_acquisition = _clone_module(
    acquisition_v2,
    "training._arbitrary_plane_acquisition_v3_isolated",
    {
        "global_reference_plane_geometry": (
            geometry_v3.stable_global_reference_plane_geometry_v3
        )
    },
)
_synthetic_generator = _clone_module(
    synthetic_generator_v2,
    "training._arbitrary_plane_synthetic_generator_v3_isolated",
    {"acquisition": _acquisition},
)
_subject_slab = _clone_module(
    subject_slab_v2,
    "training._arbitrary_plane_subject_slab_v3_isolated",
    {
        "acquisition": _acquisition,
        "synthetic_generator": _synthetic_generator,
        "verify_v2_generic_global_reference_slab_render": (
            _synthetic_generator.verify_v2_generic_global_reference_slab_render
        ),
        "verify_v2_smoke_global_reference_slab_render": (
            _synthetic_generator.verify_v2_smoke_global_reference_slab_render
        ),
        "v2_generic_slab_render_receipt": (
            _synthetic_generator.v2_generic_slab_render_receipt
        ),
        "v2_slab_render_receipt": _synthetic_generator.v2_slab_render_receipt,
        "reduce_v2_slab_samples": _synthetic_generator.reduce_v2_slab_samples,
    },
)
_support_resolution = _clone_module(
    support_resolution_v2,
    "training._arbitrary_plane_support_resolution_v3_isolated",
    {
        "acquisition": _acquisition,
        "slab": _synthetic_generator,
        "subject_slab": _subject_slab,
    },
)


def make_generic_global_reference_centre_render_v3(*args, **kwargs):
    return _acquisition.make_v2_generic_global_reference_centre_render(
        *args, **kwargs
    )


def replay_generic_global_reference_centre_render_v3(*args, **kwargs):
    return _acquisition.replay_v2_generic_global_reference_centre_render(
        *args, **kwargs
    )


def verify_generic_global_reference_centre_render_v3(*args, **kwargs):
    return _acquisition.verify_v2_generic_global_reference_centre_render(
        *args, **kwargs
    )


def make_generic_global_reference_slab_render_v3(*args, **kwargs):
    return _synthetic_generator.make_v2_generic_global_reference_slab_render(
        *args, **kwargs
    )


def replay_generic_global_reference_slab_render_v3(*args, **kwargs):
    return _synthetic_generator.replay_v2_generic_global_reference_slab_render(
        *args, **kwargs
    )


def verify_generic_global_reference_slab_render_v3(*args, **kwargs):
    return _synthetic_generator.verify_v2_generic_global_reference_slab_render(
        *args, **kwargs
    )


def resolve_subject_support_v3(*args, **kwargs):
    return _support_resolution.resolve_subject_support_v2(*args, **kwargs)


def replay_subject_support_resolution_v3(*args, **kwargs):
    return _support_resolution.replay_subject_support_resolution_v2(*args, **kwargs)


def verify_subject_support_resolution_v3(*args, **kwargs):
    return _support_resolution.verify_subject_support_resolution_v2(*args, **kwargs)


def make_subject_slab_render_v3(*args, **kwargs):
    return _subject_slab.make_subject_slab_render_v2(*args, **kwargs)


def replay_subject_slab_render_v3(*args, **kwargs):
    return _subject_slab.replay_subject_slab_render_v2(*args, **kwargs)


def verify_subject_slab_render_v3(*args, **kwargs):
    return _subject_slab.verify_subject_slab_render_v2(*args, **kwargs)


def make_section_processing_render_v3(
    subject_slab_render,
    plan,
    prepared_context,
    precursor,
    *,
    subject_plan,
    batch_size=None,
):
    _subject_slab._verify_subject_slab_render_with_mapper_v2(
        subject_slab_render,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=None,
    )
    raster, mapped, source_receipt, pose_reference = (
        section_processing_v2._subject_slab_inputs(subject_slab_render, plan)
    )
    section_processing_v2._verify_subject_slab_processing_lineage(
        subject_slab_render, plan
    )
    return section_processing_v2._make_section_processing_render_from_arrays_v2(
        raster,
        mapped,
        plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
        batch_size=batch_size,
    )


def make_section_processing_render_from_generated_subject_v3(
    subject_slab_render,
    plan,
    *,
    batch_size=None,
):
    """Make a section from a just-generated receipt-bound subject slab.

    The complete subject/section replay is performed once when the prepared
    observation parent is authenticated, rather than both before and after this
    deterministic render.
    """
    if subject_slab_render.get("receipt_sha256") != acquisition_v2._payload_sha256(
        _subject_slab.subject_slab_render_receipt_v2(subject_slab_render)
    ):
        raise ValueError("generated subject slab live receipt changed")
    raster, mapped, source_receipt, pose_reference = (
        section_processing_v2._subject_slab_inputs(subject_slab_render, plan)
    )
    section_processing_v2._verify_subject_slab_processing_lineage(
        subject_slab_render, plan
    )
    return section_processing_v2._make_section_processing_render_from_arrays_v2(
        raster,
        mapped,
        plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
        batch_size=batch_size,
    )


def verify_section_processing_render_v3(
    render,
    subject_slab_render,
    plan,
    prepared_context,
    precursor,
    *,
    subject_plan,
    batch_size=None,
    subject_to_ccf_mapper=None,
):
    _subject_slab._verify_subject_slab_render_with_mapper_v2(
        subject_slab_render,
        prepared_context,
        precursor,
        subject_plan=subject_plan,
        batch_size=batch_size,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    raster, mapped, source_receipt, pose_reference = (
        section_processing_v2._subject_slab_inputs(subject_slab_render, plan)
    )
    section_processing_v2._verify_subject_slab_processing_lineage(
        subject_slab_render, plan
    )
    section_processing_v2._verify_section_processing_render_from_arrays_v2(
        render,
        raster,
        mapped,
        plan,
        source_stage_receipt=source_receipt,
        pose_anatomy_reference=pose_reference,
    )


def adapter_receipt_v3(precursor):
    geometry = precursor["geometry"]
    payload = {
        "schema_version": LEGACY_CHAIN_ADAPTER_V3_SCHEMA,
        "algorithm": LEGACY_CHAIN_ADAPTER_V3_ALGORITHM,
        "implementation_source_sha256": {
            "arbitrary_plane_legacy_chain_v3.py": acquisition_v2._normalized_text_sha256(
                _SOURCE_ROOT / "arbitrary_plane_legacy_chain_v3.py"
            ),
            "arbitrary_plane_geometry_v3.py": acquisition_v2._normalized_text_sha256(
                _SOURCE_ROOT / "arbitrary_plane_geometry_v3.py"
            ),
        },
        "geometry_contract_v3": geometry["geometry_contract_v3"],
        "global_reference_grid_id": geometry["global_reference_grid_id"],
        "compatibility_carrier": {
            "v2_plane_realization_id": precursor["v2_plane_realization_id"],
            "centre_plane_render_id": precursor["centre_plane_render_id"],
            "slab_recipe_id": precursor["slab_recipe_id"],
            "slab_render_id": precursor["slab_render_id"],
            "receipt_sha256": precursor["receipt_sha256"],
        },
        "v2_module_mutation": False,
        "parallel_row_generation_safe": True,
        "learned_dependencies": [],
    }
    return {**payload, "receipt_sha256": acquisition_v2._payload_sha256(payload)}
