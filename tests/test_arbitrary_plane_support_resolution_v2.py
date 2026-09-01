import copy

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_subject_slab_v2 as subject_slab
import training.arbitrary_plane_support_resolution_v2 as resolution
import training.arbitrary_plane_synthetic_generator_v2 as slab
from training.arbitrary_plane_support import build_annotation_support_index


def _prepared_context():
    annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    annotation[2:15, 3:13, 1:11] = 7
    annotation[7:11, 4:8, 7:11] = 19
    ap, dv, ml = np.indices(annotation.shape)
    scalar = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.float32)
    support = build_annotation_support_index(
        annotation,
        atlas_id="support-resolution-fixture-ccf",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(11.0, 17.0, 29.0),
        origin_um=(-71.0, 23.0, 107.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return acquisition.prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="fixture decoder",
        annotation_decoder="fixture decoder",
    )


@pytest.fixture(scope="module")
def prepared():
    return _prepared_context()


def _arguments(**overrides):
    arguments = {
        "subject_plan": None,
        "master_root_seed": "0x5355505245534f4c",
        "split": "train",
        "split_index": 3,
        "animal_index": 5,
        "animal_id": "animal-five",
        "section_index": 41,
        "plane_stratum": "general_oblique",
        "nominal_cut_thickness_um": 55.0,
        "specimen_id": "specimen-five-a",
        "experiment_id": "experiment-five",
        "max_attempts": 8,
    }
    arguments.update(overrides)
    return arguments


@pytest.fixture(scope="module")
def real_bundle(prepared):
    result = resolution.resolve_subject_support_v2(prepared, **_arguments())
    assert result["resolution"]["status"] == "accepted"
    return result


def _fake_attempt_factory(accepted_attempts):
    accepted_attempts = set(accepted_attempts)

    def fake_attempt(
        prepared_context,
        *,
        subject_plan,
        lineage,
        config,
        attempt_seed,
        batch_size,
    ):
        index = attempt_seed["attempt_index"]
        token = acquisition._payload_sha256(attempt_seed)
        accepted = index in accepted_attempts
        precursor = {
            "schema_version": slab.V2_GENERIC_SLAB_SCHEMA,
            "algorithm": slab.V2_GENERIC_SLAB_ALGORITHM,
            "v2_plane_realization_id": f"plane-{token}",
            "centre_plane_render_id": f"centre-{token}",
            "slab_recipe_id": f"recipe-{token}",
            "slab_render_id": f"slab-{token}",
            "receipt_sha256": f"precursor-receipt-{token}",
        }
        support_acceptance = subject_slab._support_acceptance(17 if accepted else 0)
        probe = {
            "subject_centre_support_probe_id": f"probe-{token}",
            "receipt_sha256": f"probe-receipt-{token}",
            "mapped_centre_coordinate_receipt": {
                "dtype": "<f4",
                "shape": [2, 2, 3],
                "array_sha256": token,
            },
            "centre_annotation_receipt": {
                "dtype": "<i8",
                "shape": [2, 2],
                "array_sha256": token,
            },
            "support_acceptance": support_acceptance,
        }
        return precursor, probe

    return fake_attempt


def _rereceipt(result):
    artifact = result["resolution"]
    artifact["support_resolution_plan_id"] = acquisition._payload_sha256(
        resolution._plan_identity_payload(artifact)
    )
    artifact["subject_support_resolution_id"] = acquisition._payload_sha256(
        resolution._resolution_identity_payload(artifact)
    )
    artifact["receipt_sha256"] = acquisition._payload_sha256(
        resolution.subject_support_resolution_receipt_v2(artifact)
    )


def test_attempt_root_uses_numeric_coordinates_not_animal_labels(prepared, monkeypatch):
    base = resolution.derive_subject_support_attempt_root_seed_v2(
        "0x5355505245534f4c", 3, 5, 41, 0
    )
    assert len(
        {
            base,
            resolution.derive_subject_support_attempt_root_seed_v2(
                "0x5355505245534f4c", 4, 5, 41, 0
            ),
            resolution.derive_subject_support_attempt_root_seed_v2(
                "0x5355505245534f4c", 3, 6, 41, 0
            ),
            resolution.derive_subject_support_attempt_root_seed_v2(
                "0x5355505245534f4c", 3, 5, 42, 0
            ),
            resolution.derive_subject_support_attempt_root_seed_v2(
                "0x5355505245534f4c", 3, 5, 41, 1
            ),
        }
    ) == 5

    monkeypatch.setattr(
        resolution, "_make_verified_attempt", _fake_attempt_factory({0})
    )
    first = resolution.resolve_subject_support_v2(prepared, **_arguments())
    renamed = resolution.resolve_subject_support_v2(
        prepared,
        **_arguments(
            animal_id="renamed-animal-five",
            specimen_id="renamed-specimen",
            experiment_id="renamed-experiment",
        ),
    )
    assert first["resolution"]["attempts"][0]["attempt_seed"] == renamed[
        "resolution"
    ]["attempts"][0]["attempt_seed"]
    assert first["resolution"]["lineage"] != renamed["resolution"]["lineage"]


def test_bounded_first_success_and_authenticated_exhaustion(prepared, monkeypatch):
    monkeypatch.setattr(
        resolution, "_make_verified_attempt", _fake_attempt_factory({1, 3})
    )
    accepted = resolution.resolve_subject_support_v2(
        prepared, **_arguments(max_attempts=6)
    )
    artifact = accepted["resolution"]
    assert [attempt["attempt_index"] for attempt in artifact["attempts"]] == [0, 1]
    assert all(
        attempt["plane_sample_index"] == artifact["configuration"]["section_index"]
        and attempt["plane_stratum"] == artifact["configuration"]["plane_stratum"]
        for attempt in artifact["attempts"]
    )
    assert [attempt["accepted"] for attempt in artifact["attempts"]] == [False, True]
    assert artifact["accepted_attempt_index"] == 1
    assert accepted["accepted_precursor"] is not None
    assert accepted["accepted_probe"] is not None

    monkeypatch.setattr(
        resolution, "_make_verified_attempt", _fake_attempt_factory(set())
    )
    exhausted = resolution.resolve_subject_support_v2(
        prepared, **_arguments(max_attempts=3)
    )
    artifact = exhausted["resolution"]
    assert artifact["status"] == "exhausted"
    assert [attempt["attempt_index"] for attempt in artifact["attempts"]] == [0, 1, 2]
    assert not any(attempt["accepted"] for attempt in artifact["attempts"])
    assert artifact["accepted_attempt_index"] is None
    assert exhausted["accepted_precursor"] is None
    assert exhausted["accepted_probe"] is None
    assert "synthetic_realization_id" not in repr(exhausted)
    resolution._validate_structure(exhausted)
    resolution._validate_semantics(exhausted)


def test_real_animal_label_rename_preserves_pose_stream(prepared, real_bundle):
    renamed = resolution.resolve_subject_support_v2(
        prepared,
        **_arguments(
            animal_id="renamed-animal-five",
            specimen_id="renamed-specimen",
            experiment_id="renamed-experiment",
        ),
    )
    assert renamed["resolution"]["status"] == "accepted"
    assert renamed["resolution"]["attempts"][0]["attempt_seed"] == real_bundle[
        "resolution"
    ]["attempts"][0]["attempt_seed"]
    assert renamed["accepted_precursor"]["sampling"] == real_bundle[
        "accepted_precursor"
    ]["sampling"]
    assert renamed["accepted_precursor"]["geometry"]["array_receipts"] == real_bundle[
        "accepted_precursor"
    ]["geometry"]["array_receipts"]
    assert renamed["resolution"]["lineage"] != real_bundle["resolution"]["lineage"]


def test_real_resolution_replay_strict_verification_and_no_final_id(
    prepared, real_bundle
):
    artifact = real_bundle["resolution"]
    assert [attempt["attempt_index"] for attempt in artifact["attempts"]] == list(
        range(len(artifact["attempts"]))
    )
    assert artifact["attempts"][-1]["accepted"] is True
    assert artifact["attempts"][-1]["precursor_reference"]["slab_render_id"] == (
        real_bundle["accepted_precursor"]["slab_render_id"]
    )
    assert real_bundle["accepted_precursor"]["generator"]["resolved_config"][
        "sample_index"
    ] == artifact["configuration"]["section_index"]
    assert artifact["decision_disclosure"][
        "precursor_reference_scalar_rendered"
    ] is True
    assert artifact["decision_disclosure"][
        "precursor_reference_scalar_used_for_decision"
    ] is False
    assert artifact["decision_disclosure"]["post_deformation_scalar_sampled"] is False
    assert "synthetic_realization_id" not in repr(real_bundle)

    replay = resolution.replay_subject_support_resolution_v2(
        real_bundle, prepared, subject_plan=None
    )
    assert replay["resolution"] == artifact
    resolution.verify_subject_support_resolution_v2(
        real_bundle, prepared, **_arguments()
    )


def test_resolution_tamper_is_rejected_before_or_by_replay(prepared, real_bundle):
    changed = copy.deepcopy(real_bundle)
    changed["resolution"]["extra"] = 1
    with pytest.raises(ValueError, match="missing, extra"):
        resolution.verify_subject_support_resolution_v2(
            changed, prepared, **_arguments()
        )

    changed = copy.deepcopy(real_bundle)
    changed["resolution"]["attempts"][0]["attempt_seed"][
        "attempt_root_seed_uint64"
    ] = "0x0000000000000000"
    _rereceipt(changed)
    with pytest.raises(ValueError, match="attempt order, seed, or decision"):
        resolution.verify_subject_support_resolution_v2(
            changed, prepared, **_arguments()
        )

    changed = copy.deepcopy(real_bundle)
    changed["resolution"]["lineage"]["animal_id"] = "coherently-rereceipted-label"
    _rereceipt(changed)
    with pytest.raises(ValueError, match="replay"):
        resolution.verify_subject_support_resolution_v2(
            changed, prepared, **_arguments()
        )


def test_accepted_full_subject_slab_matches_exact_probe(prepared, real_bundle):
    full = subject_slab.make_subject_slab_render_v2(
        prepared,
        real_bundle["accepted_precursor"],
        subject_plan=None,
    )
    resolution.verify_accepted_subject_slab_matches_support_resolution_v2(
        real_bundle, full
    )
    assert full["support_probe_reference"] == {
        "subject_centre_support_probe_id": real_bundle["accepted_probe"][
            "subject_centre_support_probe_id"
        ],
        "receipt_sha256": real_bundle["accepted_probe"]["receipt_sha256"],
    }

    changed = copy.deepcopy(full)
    changed["support_acceptance"]["centre_plane_brain_pixel_count"] += 1
    with pytest.raises(ValueError, match="does not match"):
        resolution.verify_accepted_subject_slab_matches_support_resolution_v2(
            real_bundle, changed
        )


@pytest.mark.parametrize("value", [1.0, 1.9, True, np.bool_(False)])
def test_attempt_roots_reject_noninteger_schedule_coordinates(value):
    with pytest.raises((TypeError, ValueError)):
        resolution.derive_subject_support_attempt_root_seed_v2(7, value, 2, 3, 4)


@pytest.mark.parametrize(
    "override",
    [
        {"split_index": 1.9},
        {"animal_index": True},
        {"section_index": 2.0},
        {"max_attempts": 3.0},
        {"parent_shape_h_w": (16.0, 16)},
    ],
)
def test_resolution_rejects_noninteger_authenticated_schedule_fields(
    prepared, override
):
    with pytest.raises((TypeError, ValueError)):
        resolution.resolve_subject_support_v2(prepared, **_arguments(**override))
