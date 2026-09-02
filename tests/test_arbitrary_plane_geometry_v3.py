from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_geometry_v3 as geometry_v3
import training.arbitrary_plane_legacy_chain_v3 as legacy_chain_v3
from training.arbitrary_plane_rendered_generator import (
    effective_renderer_sampling_arrays,
)
from training.arbitrary_plane_support import build_annotation_support_index


def _large_index_support():
    annotation = np.zeros((1025, 5, 5), dtype=np.uint8)
    annotation[1:1024, 1:4, 1:4] = 1
    return build_annotation_support_index(
        annotation,
        atlas_id="large-index-fixture",
        atlas_version="v1",
        source_uri="file:///large-index-annotation.nrrd",
        source_sha256="a" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(25.0, 25.0, 25.0),
        origin_um=(0.0, 0.0, 0.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )


def test_v3_canonical_renderer_raster_survives_large_index_float32_ulp_gap():
    support = _large_index_support()
    normal = np.asarray(
        [0.5460341925166653, 0.634002931868367, -0.5476193413174775]
    )
    intervals = acquisition.shifted_component_interval_union(normal, support)
    offset = float(
        np.asarray(intervals["support_origin_interval_union_um"])[0].mean()
    )
    with pytest.raises(
        ValueError, match="effective O/U/V parameterization exceeds"
    ):
        acquisition.global_reference_plane_geometry(
            normal, offset, 6.136359426050744, support
        )

    geometry = geometry_v3.stable_global_reference_plane_geometry_v3(
        normal, offset, 6.136359426050744, support
    )
    effective = effective_renderer_sampling_arrays(
        geometry,
        tuple(support["annotation_shape"]),
        origin_ap_dv_ml_um=tuple(support["origin_um"]),
        voxel_size_ap_dv_ml_um=tuple(support["voxel_size_um"]),
    )
    assert geometry["geometry_contract_v3"]["schema_version"] == (
        geometry_v3.GEOMETRY_V3_SCHEMA
    )
    assert geometry["diagnostics"]["canonical_effective_grid_byte_equal"] is True
    assert (
        geometry["diagnostics"][
            "legacy_expanded_ouv_max_abs_index_diagnostic_only"
        ]
        > 1e-5
    )
    assert geometry["array_receipts"][
        "independent_ouv_parameterized_coordinate_raster_float32"
    ] == acquisition._array_receipt(
        effective["coordinate_raster_allen_index_float32"]
    )


def test_v3_geometry_is_antipodally_identical_without_mutating_v2():
    support = _large_index_support()
    normal = np.asarray([0.31, -0.72, 0.6208864639664438])
    intervals = acquisition.shifted_component_interval_union(normal, support)
    offset = float(
        np.asarray(intervals["support_origin_interval_union_um"])[0].mean()
    )
    geometry = geometry_v3.stable_global_reference_plane_geometry_v3(
        normal, offset, 1.137, support
    )
    antipodal = geometry_v3.stable_global_reference_plane_geometry_v3(
        -normal, -offset, 1.137, support
    )
    assert geometry["global_reference_grid_id"] == antipodal[
        "global_reference_grid_id"
    ]

    assert acquisition.global_reference_plane_geometry is not (
        geometry_v3.stable_global_reference_plane_geometry_v3
    )


def test_v3_adapter_replays_in_parallel_without_mutating_v2_module():
    annotation = np.zeros((1025, 5, 5), dtype=np.uint8)
    annotation[1:1024, 1:4, 1:4] = 1
    support = _large_index_support()
    scalar = np.indices(annotation.shape)[0].astype(np.float32)
    context = acquisition.prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///large-index-template.nrrd",
        scalar_source_sha256="b" * 64,
        template_decoder="fixture",
        annotation_decoder="fixture",
    )
    original = acquisition.global_reference_plane_geometry

    def make():
        return legacy_chain_v3.make_generic_global_reference_centre_render_v3(
            context,
            "train",
            "0x415154564f330001",
            0,
            "general_oblique",
            animal_id="animal-0",
            animal_index=0,
            specimen_id="specimen-0",
            experiment_id="experiment-0",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = tuple(pool.map(lambda _: make(), range(2)))
    legacy_chain_v3.verify_generic_global_reference_centre_render_v3(
        first, context
    )
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert first["geometry"]["geometry_contract_v3"]["schema_version"] == (
        geometry_v3.GEOMETRY_V3_SCHEMA
    )
    assert acquisition.global_reference_plane_geometry is original
