import copy
from inspect import signature

import numpy as np
import pytest

from training import arbitrary_plane_acquisition_window_v3 as window


PLAN_ARGUMENTS = {
    "root_seed": "0x415154564f320001",
    "split": "development",
    "sample_index": 4,
}
D_UM = 800.0
PARENT_SHAPE = (256, 256)
CANVAS_SHAPE = (192, 256)


def _plan(**changes):
    return window.sample_acquisition_window_plan_v3(**{**PLAN_ARGUMENTS, **changes})


def _quicknii_map(ouv, shape):
    height, width = shape
    y, x = np.indices(shape, dtype=np.float64)
    origin, edge_u, edge_v = np.asarray(ouv, dtype=np.float64)
    return np.ascontiguousarray(
        origin + (x / width)[..., None] * edge_u + (y / height)[..., None] * edge_v
    )


def _inputs():
    height, width = PARENT_SHAPE
    y, x = np.indices(PARENT_SHAPE)
    edge = D_UM * width / (width - 1)
    design_ouv = np.array(
        [[100.0, 200.0, 300.0], [edge, 0.0, 0.0], [0.0, edge, 0.0]],
        dtype=np.float64,
    )
    arrays = {
        "scalar": np.ascontiguousarray((2 * y + x).astype(np.float32)),
        "annotation": np.ascontiguousarray((1 + (x >= width // 2)).astype(np.int64)),
        "tissue": np.ones(PARENT_SHAPE, dtype=bool),
        "abstention": np.zeros(PARENT_SHAPE, dtype=bool),
        "coordinate": _quicknii_map(design_ouv, PARENT_SHAPE),
        "absolute_map_yx": np.ascontiguousarray(
            np.stack((y, x), axis=-1), dtype=np.float64
        ),
        "vector_yx": np.broadcast_to(
            np.array([2.0, 3.0], dtype=np.float64), PARENT_SHAPE + (2,)
        ).copy(),
    }
    roles = {
        "scalar": "scalar",
        "annotation": "annotation",
        "tissue": "mask",
        "abstention": "abstention-mask",
        "coordinate": "ccf-coordinate",
        "absolute_map_yx": "absolute-map-yx",
        "vector_yx": "vector-yx",
    }
    validity = {name: np.ones(PARENT_SHAPE, dtype=bool) for name in arrays}
    return arrays, roles, validity, design_ouv


def _metadata():
    upstream = {
        name: f"fixture-{name}" for name in window.UPSTREAM_REALIZATION_ID_FIELDS
    }
    section_payload = {
        "schema_version": "fixture.section-processing-receipt/v1",
        "section_processing_render_id": upstream["section_processing_render_id"],
    }
    section_receipt = {
        **section_payload,
        "receipt_sha256": window.acquisition._payload_sha256(section_payload),
    }
    lineage = {
        "split": "development",
        "animal_index": 7,
        "animal_id": "animal-007",
        "specimen_id": "specimen-007-A",
        "experiment_id": "experiment-2026-007",
        "synthetic_animal_id": "synthetic-animal-007",
        "section_index": 3,
        "section_id": "section-003",
    }
    landmarks = {
        "expert_landmarks": np.array(
            [[128.0, 128.0], [0.0, 0.0], [255.0, 255.0]], dtype=np.float64
        )
    }
    return upstream, section_receipt, lineage, landmarks


def _apply(plan=None, arrays=None, roles=None, validity=None, **changes):
    plan = _plan() if plan is None else plan
    if arrays is None:
        arrays, roles, validity, design_ouv = _inputs()
    else:
        _, _, _, design_ouv = _inputs()
    upstream, section_receipt, lineage, landmarks = _metadata()
    arguments = {
        "source_validity": validity,
        "global_reference_grid_id": "global-grid-fixture",
        "global_reference_fov_uv_um": (D_UM, D_UM),
        "design_quicknii_ouv": design_ouv,
        "centre_plane_support_mask": arrays["tissue"],
        "optical_slab_support_mass": arrays["tissue"].astype(np.float32),
        "upstream_realization_ids": upstream,
        "section_processing_receipt": section_receipt,
        "section_processing_receipt_sha256": window.acquisition._payload_sha256(
            section_receipt
        ),
        "lineage": lineage,
        "parent_landmarks_yx": landmarks,
    }
    arguments.update(changes)
    return window.apply_acquisition_window_v3(plan, arrays, roles, **arguments), (
        arrays,
        roles,
        arguments,
    )


def test_plan_api_is_fixed_scope_pose_blind_and_uses_frozen_v2_seed():
    assert set(signature(window.sample_acquisition_window_plan_v3).parameters) == {
        "root_seed",
        "split",
        "sample_index",
    }
    plan = _plan()
    assert tuple(plan["parent_shape_h_w"]) == PARENT_SHAPE
    assert tuple(plan["canvas_shape_h_w"]) == CANVAS_SHAPE
    expected_seed = window.acquisition.derive_v2_field_seed(
        PLAN_ARGUMENTS["root_seed"], "development", 4, "window", "view-plan", 0
    )
    assert plan["view_plan_seed_uint64"] == f"0x{expected_seed:016x}"
    assert plan["preflight"] == window.acquisition._preflight_provenance()
    assert plan["dependency_source_sha256"] == window.acquisition._source_hashes()
    with pytest.raises(ValueError, match="train/development"):
        _plan(split="final_test")


def test_frozen_smoke_severity_schedule_is_internal_and_exact():
    observed = [_plan(sample_index=index)["plan_severity"] for index in range(20)]
    expected = [row[1] for row in window.acquisition.V2_SMOKE_ASSIGNMENTS]
    assert observed == expected
    assert all(
        "smoke assignment" in _plan(sample_index=index)["severity_selection_policy"]
        for index in range(20)
    )


def test_plan_make_replay_verify_and_tamper_are_exact():
    plan = window.make_acquisition_window_plan_v3(**PLAN_ARGUMENTS)
    assert window.replay_acquisition_window_plan_v3(plan) == plan
    window.verify_acquisition_window_plan_v3(plan)
    changed = copy.deepcopy(plan)
    changed["preflight"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not replay"):
        window.verify_acquisition_window_plan_v3(changed)


def test_general_fov_and_design_ouv_endpoint_coherence_are_required():
    _, _, _, design = _inputs()
    fov = np.array([D_UM, D_UM - 100.0], dtype=np.float64)
    edge_lengths = fov * np.array(PARENT_SHAPE[::-1]) / (
        np.array(PARENT_SHAPE[::-1]) - 1.0
    )
    design[1] = (edge_lengths[0], 0.0, 0.0)
    design[2] = (
        0.25 * edge_lengths[1],
        np.sqrt(1.0 - 0.25**2) * edge_lengths[1],
        0.0,
    )
    artifact, _ = _apply(
        global_reference_fov_uv_um=tuple(fov),
        design_quicknii_ouv=design,
    )
    assert artifact["source_binding"]["global_reference_fov_uv_um"] == fov.tolist()
    with pytest.raises(ValueError, match="physical FOV endpoints"):
        _apply(
            global_reference_fov_uv_um=(D_UM, D_UM),
            design_quicknii_ouv=design,
        )
    design[2] = design[1]
    with pytest.raises(ValueError, match="degenerate"):
        _apply(design_quicknii_ouv=design)


@pytest.mark.parametrize(
    ("name", "array", "role"),
    [
        ("bad_mask", np.ones(PARENT_SHAPE, dtype=np.float32), "mask"),
        ("bad_annotation", np.ones(PARENT_SHAPE, dtype=np.float32), "annotation"),
        ("bad_ccf", np.ones(PARENT_SHAPE + (2,), dtype=np.float64), "ccf-coordinate"),
        ("parent_sampling_domain_mask", np.ones(PARENT_SHAPE, dtype=bool), "mask"),
        ("bad__valid_mask", np.ones(PARENT_SHAPE, dtype=bool), "mask"),
    ],
)
def test_roles_dtypes_shapes_and_reserved_names_are_strict(name, array, role):
    arrays, roles, validity, _ = _inputs()
    arrays[name] = array
    roles[name] = role
    validity[name] = np.ones(PARENT_SHAPE, dtype=bool)
    with pytest.raises(ValueError, match="role|reserved"):
        _apply(arrays=arrays, roles=roles, validity=validity)


def test_every_array_has_explicit_validity_zero_invalid_and_abstention():
    arrays, roles, validity, _ = _inputs()
    validity["coordinate"][:] = False
    arrays["coordinate"][:] = np.nan
    arrays["absolute_map_yx"][:] = 1000.0
    artifact, _ = _apply(arrays=arrays, roles=roles, validity=validity)
    coordinate_valid = artifact["arrays"]["coordinate__valid_mask"]
    map_valid = artifact["arrays"]["absolute_map_yx__valid_mask"]
    assert not coordinate_valid.any() and not map_valid.any()
    assert not artifact["arrays"]["coordinate"].any()
    assert not artifact["arrays"]["absolute_map_yx"].any()
    assert artifact["arrays"]["coordinate__abstention_mask"].all()
    assert artifact["arrays"]["absolute_map_yx__abstention_mask"].all()
    assert artifact["arrays"]["window_abstention_mask"].all()


def test_identity_absolute_map_and_ccf_ramp_follow_effective_ouv():
    artifact, _ = _apply()
    inside = artifact["arrays"]["absolute_map_yx__valid_mask"]
    y, x = np.indices(CANVAS_SHAPE, dtype=np.float64)
    expected_yx = np.stack((y, x), axis=-1)
    np.testing.assert_allclose(
        artifact["arrays"]["absolute_map_yx"][inside], expected_yx[inside], atol=1e-10
    )
    effective = np.asarray(artifact["transform"]["effective_quicknii_ouv"])
    expected_ccf = _quicknii_map(effective, CANVAS_SHAPE)
    ccf_valid = artifact["arrays"]["coordinate__valid_mask"]
    np.testing.assert_allclose(
        artifact["arrays"]["coordinate"][ccf_valid], expected_ccf[ccf_valid], atol=1e-10
    )


def test_retention_uses_last_pixel_centre_as_closed_upper_bound():
    mask = np.zeros(PARENT_SHAPE, dtype=bool)
    mask[191, 255] = True
    mask[192, 255] = True
    identity_plan = {
        "parent_to_canvas_affine_float64": np.eye(3).tolist(),
        "canvas_shape_h_w": list(CANVAS_SHAPE),
    }
    assert window._retained_fraction(mask, identity_plan) == 0.5
    assert window._retained_mass_fraction(mask.astype(np.float32), identity_plan) == 0.5


def test_lineage_upstream_section_receipt_and_landmarks_are_bound():
    artifact, (_, _, arguments) = _apply()
    assert set(window.UPSTREAM_REALIZATION_ID_FIELDS) == {
        "v2_plane_realization_id",
        "slab_render_id",
        "subject_slab_render_id",
        "section_processing_plan_id",
        "section_processing_realization_id",
        "synthetic_section_processing_id",
        "section_processing_render_id",
    }
    assert "v2_acquisition_realization_id" not in arguments["upstream_realization_ids"]
    assert artifact["lineage"]["specimen_id"] == "specimen-007-A"
    assert artifact["lineage"]["experiment_id"] == "experiment-2026-007"
    assert artifact["lineage"]["synthetic_animal_id"] == "synthetic-animal-007"
    assert artifact["source_binding"]["upstream_realization_ids"] == arguments[
        "upstream_realization_ids"
    ]
    assert set(artifact["landmarks_yx"]) == {"expert_landmarks"}
    assert artifact["landmark_validity"]["expert_landmarks"].dtype == bool
    bad_ids = dict(arguments["upstream_realization_ids"])
    bad_ids.pop("slab_render_id")
    with pytest.raises(ValueError, match="upstream realization IDs"):
        _apply(upstream_realization_ids=bad_ids)
    with pytest.raises(ValueError, match="not authenticated"):
        _apply(section_processing_receipt_sha256="0" * 64)


def test_authenticated_label_changes_never_change_plan_rng_but_change_realization():
    baseline_plan = _plan()
    baseline, (_, _, arguments) = _apply(plan=baseline_plan)
    renamed = dict(arguments["lineage"])
    renamed["specimen_id"] = "renamed-specimen"
    changed, _ = _apply(plan=baseline_plan, lineage=renamed)
    assert changed["window_plan"] == baseline["window_plan"]
    assert changed["acquisition_window_realization_id"] != baseline[
        "acquisition_window_realization_id"
    ]


def test_realization_make_replay_verify_and_all_tamper_are_exact():
    artifact, (arrays, roles, arguments) = _apply()
    made = window.make_acquisition_window_realization_v3(
        artifact["window_plan"], arrays, roles, **arguments
    )
    replay = window.replay_acquisition_window_realization_v3(
        artifact["window_plan"], arrays, roles, **arguments
    )
    assert artifact["receipt_sha256"] == made["receipt_sha256"]
    assert artifact["receipt_sha256"] == replay["receipt_sha256"]
    window.verify_acquisition_window_realization_v3(
        artifact, arrays, roles, **arguments
    )
    extra = copy.deepcopy(artifact)
    extra["unreceipted"] = "tamper"
    with pytest.raises(ValueError, match="does not replay"):
        window.verify_acquisition_window_realization_v3(
            extra, arrays, roles, **arguments
        )
    changed_array = copy.deepcopy(artifact)
    changed_array["arrays"]["scalar"][0, 0] += np.float32(1.0)
    with pytest.raises(ValueError, match="does not replay"):
        window.verify_acquisition_window_realization_v3(
            changed_array, arrays, roles, **arguments
        )
    changed_landmark = copy.deepcopy(artifact)
    changed_landmark["landmarks_yx"]["expert_landmarks"][0, 0] += 1.0
    with pytest.raises(ValueError, match="does not replay"):
        window.verify_acquisition_window_realization_v3(
            changed_landmark, arrays, roles, **arguments
        )


def test_weighted_optical_mass_retention_is_not_booleanized():
    arrays, roles, validity, _ = _inputs()
    mass = np.zeros(PARENT_SHAPE, dtype=np.float32)
    mass[128, 128] = 1.0
    mass[0, 0] = 9.0
    artifact, _ = _apply(
        arrays=arrays,
        roles=roles,
        validity=validity,
        optical_slab_support_mass=mass,
    )
    expected = window._retained_mass_fraction(mass, artifact["window_plan"])
    assert artifact["retention_audit"]["retained_optical_slab_support_fraction"] == expected


def test_v2_sources_remain_unmodified_by_parallel_v3_module():
    assert window._SOURCE.name == "arbitrary_plane_acquisition_window_v3.py"
    assert window.WINDOW_PLAN_V3_SCHEMA.endswith("/v3")
    assert window.WINDOW_REALIZATION_V3_SCHEMA.endswith("/v3")
