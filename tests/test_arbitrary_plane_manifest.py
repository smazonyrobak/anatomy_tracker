import hashlib
import json

import numpy as np
import pytest

from training.arbitrary_plane_manifest import (
    CANDIDATE_SCHEMA,
    MANIFEST_SCHEMA,
    REFERENCE_STRATUM,
    RNG_FIELDS,
    STRESS_STRATUM,
    build_annotation_support,
    canonicalize_plane,
    canonicalize_rp2_normal,
    load_arbitrary_plane_manifest,
    make_arbitrary_plane_manifest,
    make_brain_intersecting_candidates,
    plane_intersects_annotation_support,
    replay_arbitrary_plane_manifest,
    replay_brain_intersecting_candidates,
    rp2_geodesic_plane_delta,
    sample_uniform_rolls,
    sample_uniform_rp2_normals,
    save_arbitrary_plane_manifest,
    support_projection,
)


def _support(annotation=None):
    if annotation is None:
        zz, yy, xx = np.mgrid[:17, :15, :13]
        annotation = ((((zz - 8) / 7) ** 2 + ((yy - 7) / 6) ** 2 + ((xx - 6) / 5) ** 2) <= 1).astype(np.uint16)
    return build_annotation_support(
        annotation,
        atlas_id="allen-ccf",
        atlas_version="2017-25um",
        source_uri="file:///frozen/annotation_25.nrrd",
        source_sha256="2" * 64,
        voxel_size_um=(25.0, 25.0, 25.0),
        source_entity_type="atlas",
        coordinate_axis_directions=("posterior", "ventral", "right"),
        origin_um=(-200.0, -175.0, -150.0),
    )


def _payload_hash(payload):
    canonical = json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_hash(artifact, hash_field):
    return _payload_hash({key: value for key, value in artifact.items() if key != hash_field})


def _serialized_seed_values(value):
    seeds = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "root_seed":
                seeds.append(item)
            elif key.endswith("_seed_uint64"):
                seeds.extend(item.values() if isinstance(item, dict) else [item])
            seeds.extend(_serialized_seed_values(item))
    elif isinstance(value, list):
        for item in value:
            seeds.extend(_serialized_seed_values(item))
    return seeds


def test_antipodal_canonicalization_is_deterministic_and_plane_equivalent():
    normals = np.asarray(
        [
            [1.0, -2.0, 0.5],
            [-3.0, -3.0, 1.0],
            [0.0, 0.0, -4.0],
        ]
    )
    canonical = canonicalize_rp2_normal(normals)

    assert np.array_equal(canonical, canonicalize_rp2_normal(-normals))
    assert np.allclose(np.linalg.norm(canonical, axis=1), 1.0)
    pivots = np.argmax(np.abs(canonical), axis=1)
    assert np.all(canonical[np.arange(len(canonical)), pivots] > 0.0)
    normal, offset, sign = canonicalize_plane([-1.0, 2.0, 0.0], -30.0)
    equivalent_normal, equivalent_offset, _ = canonicalize_plane([1.0, -2.0, 0.0], 30.0)
    assert sign == 1
    assert np.allclose(normal, [-1.0 / np.sqrt(5.0), 2.0 / np.sqrt(5.0), 0.0])
    assert offset == pytest.approx(-30.0 / np.sqrt(5.0))
    assert np.array_equal(normal, equivalent_normal)
    assert offset == equivalent_offset


def test_normal_and_roll_sampling_are_uniform_and_repeatable():
    normals = sample_uniform_rp2_normals(20_000, 48123)
    rolls = sample_uniform_rolls(20_000, 48123)

    assert np.array_equal(normals, sample_uniform_rp2_normals(20_000, 48123))
    assert not np.array_equal(normals, sample_uniform_rp2_normals(20_000, 48124))
    assert np.mean(normals**2, axis=0) == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=0.012)
    assert np.mean(np.abs(normals), axis=0) == pytest.approx([0.5, 0.5, 0.5], abs=0.012)
    assert np.abs(np.mean(np.exp(1j * rolls))) < 0.02
    assert rolls.min() >= 0.0
    assert rolls.max() < 2.0 * np.pi


def test_support_hash_binds_annotation_geometry_and_projection():
    support = _support()
    repeated = _support()
    changed_annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    changed_annotation[3:14, 3:12, 3:10] = 1
    changed = _support(changed_annotation)
    projection = support_projection([0.2, -0.4, 0.8], support)
    annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    annotation[3:14, 3:12, 3:10] = 1
    translated = build_annotation_support(
        annotation,
        atlas_id="allen-ccf",
        atlas_version="2017-25um",
        source_uri="file:///frozen/annotation_25.nrrd",
        source_sha256="2" * 64,
        voxel_size_um=(25.0, 25.0, 25.0),
        source_entity_type="atlas",
        coordinate_axis_directions=("posterior", "ventral", "right"),
        origin_um=(800.0, 825.0, 850.0),
    )

    assert support["support_sha256"] == repeated["support_sha256"]
    assert support["support_sha256"] != changed["support_sha256"]
    assert support["source"]["source_sha256"] == "2" * 64
    assert support["source"]["source_sha256_semantics"] == "sha256-of-raw-bytes-at-source-uri"
    assert support["projection_origin_um"] == [12.5, 12.5, 12.5]
    assert support["projection_scaling_status"].startswith("prototype-only")
    assert "scans all occupied voxels for every plane" in support["projection_scaling_limit"]
    assert support["source"]["source_entity_type"] == "atlas"
    assert support["atlas"]["coordinate_axis_directions"] == ["posterior", "ventral", "right"]
    assert support["source"]["annotation_array_sha256"] != changed["source"]["annotation_array_sha256"]
    assert projection == support_projection([-0.2, 0.4, -0.8], support)
    assert projection["bounds_um"][0] < projection["bounds_um"][1]
    assert support_projection([0.2, -0.4, 0.8], changed)["bounds_um"] == pytest.approx(
        support_projection([0.2, -0.4, 0.8], translated)["bounds_um"]
    )

    with pytest.raises(ValueError, match="raw bytes at source_uri"):
        build_annotation_support(
            np.ones((2, 2, 2), dtype=np.uint8),
            atlas_id="allen-ccf",
            atlas_version="2017-25um",
            source_uri="file:///frozen/annotation_25.nrrd",
            voxel_size_um=(25.0, 25.0, 25.0),
            source_entity_type="atlas",
            coordinate_axis_directions=("posterior", "ventral", "right"),
        )


def test_manifest_is_hash_bound_replayable_and_preserves_provenance(tmp_path):
    support = _support()
    first = make_arbitrary_plane_manifest(80, "train", 99173, support)
    repeated = make_arbitrary_plane_manifest(80, "train", 99173, support)
    other = make_arbitrary_plane_manifest(80, "train", 99174, support)

    assert first == repeated
    assert first["schema_version"] == MANIFEST_SCHEMA
    assert first["manifest_sha256"] != other["manifest_sha256"]
    assert set(first["rng"]["per_sample_stream_fields"]) == set(RNG_FIELDS) - {"stratum"}
    assert first["provenance"]["atlas"]["id"] == "allen-ccf"
    assert first["provenance"]["source"]["annotation_uri"] == "file:///frozen/annotation_25.nrrd"
    assert first["provenance"]["animal_id"] is None
    assert first["provenance"]["specimen_id"] is None
    assert first["provenance"]["experiment_id"] is None
    assert len({sample["plane_realization_id"] for sample in first["samples"]}) == 80
    assert all("synthetic_realization_id" not in sample for sample in first["samples"])
    assert first["sampling"]["identifier_contract"]["synthetic_realization_id"].startswith("reserved")
    assert replay_arbitrary_plane_manifest(first, support) == first

    path = tmp_path / "arbitrary_plane_manifest_v3.json"
    save_arbitrary_plane_manifest(path, first)
    assert load_arbitrary_plane_manifest(path) == first
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["samples"][0]["signed_offset_um"] += 1.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_sha256"):
        load_arbitrary_plane_manifest(path)


def test_uint64_seeds_are_canonical_json_strings_and_roundtrip_above_2_to_53():
    support = _support()
    manifest = make_arbitrary_plane_manifest(6, "train", 2**63 + 17, support)
    candidates = make_brain_intersecting_candidates(
        manifest,
        0,
        support,
        8,
        2**63 + 19,
        max_geodesic_deg=10.0,
        max_offset_um=30.0,
    )

    manifest_roundtrip = json.loads(json.dumps(manifest))
    candidates_roundtrip = json.loads(json.dumps(candidates))
    assert manifest_roundtrip == manifest
    assert candidates_roundtrip == candidates
    assert replay_arbitrary_plane_manifest(manifest_roundtrip, support) == manifest
    assert replay_brain_intersecting_candidates(candidates_roundtrip, manifest_roundtrip, support) == candidates
    for artifact in (manifest, candidates):
        seeds = _serialized_seed_values(artifact)
        assert seeds
        assert all(
            len(seed) == 18
            and seed.startswith("0x")
            and seed == seed.lower()
            and all(character in "0123456789abcdef" for character in seed[2:])
            for seed in seeds
        )
        assert all(not isinstance(seed, int) for seed in seeds)
    assert int(manifest["root_seed"], 16) > 2**53
    assert int(candidates["root_seed"], 16) > 2**53


def test_generator_receipts_bind_source_commit_and_fully_resolved_configs():
    support = _support()
    source_commit = "a" * 40
    manifest = make_arbitrary_plane_manifest(
        6,
        "development",
        73,
        support,
        stress_fraction=0.5,
        stress_boundary_fraction=0.07,
        animal_id="animal-1",
        specimen_id="specimen-1",
        experiment_id="experiment-1",
        max_rejection_attempts=512,
        generator_source_commit=source_commit,
    )
    receipt = manifest["generator"]

    assert receipt["implementation"]["source_path"] == "training/arbitrary_plane_manifest.py"
    assert len(receipt["implementation"]["source_sha256"]) == 64
    assert receipt["implementation"]["source_commit"] == source_commit
    assert receipt["resolved_config"] == {
        "schema_version": MANIFEST_SCHEMA,
        "sampler_algorithm": manifest["sampler_algorithm"],
        "count": 6,
        "split": "development",
        "root_seed": "0x0000000000000049",
        "support_sha256": support["support_sha256"],
        "stress_fraction": 0.5,
        "reference_fraction_bounds": [0.0, 1.0],
        "stress_boundary_fraction": 0.07,
        "animal_id": "animal-1",
        "specimen_id": "specimen-1",
        "experiment_id": "experiment-1",
        "max_rejection_attempts": 512,
        "numpy_version": np.__version__,
    }
    assert replay_arbitrary_plane_manifest(
        manifest, support, generator_source_commit=source_commit
    ) == manifest
    with pytest.raises(ValueError, match="source commit"):
        replay_arbitrary_plane_manifest(manifest, support)

    config_tamper = json.loads(json.dumps(manifest))
    config_tamper["generator"]["resolved_config"]["count"] = 7
    config_tamper["manifest_sha256"] = _artifact_hash(config_tamper, "manifest_sha256")
    with pytest.raises(ValueError, match="resolved config"):
        replay_arbitrary_plane_manifest(
            config_tamper, support, generator_source_commit=source_commit
        )

    source_tamper = json.loads(json.dumps(manifest))
    source_tamper["generator"]["implementation"]["source_sha256"] = "0" * 64
    source_tamper["manifest_sha256"] = _artifact_hash(source_tamper, "manifest_sha256")
    with pytest.raises(ValueError, match="Generator source"):
        replay_arbitrary_plane_manifest(
            source_tamper, support, generator_source_commit=source_commit
        )

    candidates = make_brain_intersecting_candidates(
        manifest,
        2,
        support,
        8,
        79,
        max_geodesic_deg=15.0,
        max_offset_um=40.0,
        generator_source_commit=source_commit,
        manifest_generator_source_commit=source_commit,
    )
    candidate_config = candidates["generator"]["resolved_config"]
    assert candidate_config["manifest_sha256"] == manifest["manifest_sha256"]
    assert candidate_config["plane_realization_id"] == manifest["samples"][2]["plane_realization_id"]
    assert candidate_config["root_seed"] == "0x000000000000004f"
    assert candidate_config["manifest_generator_source_commit"] == source_commit
    assert replay_brain_intersecting_candidates(
        candidates,
        manifest,
        support,
        generator_source_commit=source_commit,
        manifest_generator_source_commit=source_commit,
    ) == candidates

    candidate_tamper = json.loads(json.dumps(candidates))
    candidate_tamper["generator"]["resolved_config"]["max_offset_um"] = 41.0
    candidate_tamper["candidate_set_sha256"] = _artifact_hash(
        candidate_tamper, "candidate_set_sha256"
    )
    with pytest.raises(ValueError, match="resolved config"):
        replay_brain_intersecting_candidates(
            candidate_tamper,
            manifest,
            support,
            generator_source_commit=source_commit,
            manifest_generator_source_commit=source_commit,
        )


def test_candidate_construction_and_replay_reject_coherently_rehashed_manifest_receipt_tampering():
    support = _support()
    manifest = make_arbitrary_plane_manifest(6, "train", 101, support)
    candidates = make_brain_intersecting_candidates(
        manifest,
        1,
        support,
        8,
        103,
        max_geodesic_deg=12.0,
        max_offset_um=35.0,
    )

    source_tamper = json.loads(json.dumps(manifest))
    source_tamper["generator"]["implementation"]["source_sha256"] = "0" * 64
    source_tamper["manifest_sha256"] = _artifact_hash(source_tamper, "manifest_sha256")
    with pytest.raises(ValueError, match="Generator source"):
        make_brain_intersecting_candidates(
            source_tamper,
            1,
            support,
            8,
            103,
            max_geodesic_deg=12.0,
            max_offset_um=35.0,
        )

    config_tamper = json.loads(json.dumps(manifest))
    config_tamper["generator"]["resolved_config"]["count"] = 7
    config_tamper["generator"]["resolved_config_sha256"] = _payload_hash(
        config_tamper["generator"]["resolved_config"]
    )
    config_tamper["manifest_sha256"] = _artifact_hash(config_tamper, "manifest_sha256")
    with pytest.raises(ValueError, match="Manifest replay"):
        make_brain_intersecting_candidates(
            config_tamper,
            1,
            support,
            8,
            103,
            max_geodesic_deg=12.0,
            max_offset_um=35.0,
        )

    sample_tamper = json.loads(json.dumps(manifest))
    sample = sample_tamper["samples"][1]
    sample["signed_offset_um"] += 1.0
    sample["plane_geometry_sha256"] = _payload_hash(
        {
            "schema": "synthetic-plane-geometry/v1",
            "support_sha256": support["support_sha256"],
            "normal_rp2": sample["normal_rp2"],
            "signed_offset_um": sample["signed_offset_um"],
            "roll_rad": sample["roll_rad"],
        }
    )
    realization_sample = {
        key: value
        for key, value in sample.items()
        if key not in {"plane_geometry_sha256", "plane_realization_id"}
    }
    sample["plane_realization_id"] = _payload_hash(
        {
            "schema": "synthetic-plane-realization/v1",
            "sampler": sample_tamper["sampler_algorithm"],
            "support_sha256": support["support_sha256"],
            "root_seed": sample_tamper["root_seed"],
            "generator_source_sha256": sample_tamper["generator"]["implementation"][
                "source_sha256"
            ],
            "resolved_config_sha256": sample_tamper["generator"]["resolved_config_sha256"],
            **realization_sample,
        }
    )
    sample_tamper["manifest_sha256"] = _artifact_hash(sample_tamper, "manifest_sha256")
    with pytest.raises(ValueError, match="Manifest replay"):
        make_brain_intersecting_candidates(
            sample_tamper,
            1,
            support,
            8,
            103,
            max_geodesic_deg=12.0,
            max_offset_um=35.0,
        )

    replay_tamper = json.loads(json.dumps(candidates))
    replay_tamper["manifest_sha256"] = source_tamper["manifest_sha256"]
    replay_tamper["generator"]["resolved_config"]["manifest_sha256"] = source_tamper[
        "manifest_sha256"
    ]
    replay_tamper["generator"]["resolved_config_sha256"] = _payload_hash(
        replay_tamper["generator"]["resolved_config"]
    )
    replay_tamper["candidate_set_sha256"] = _artifact_hash(
        replay_tamper, "candidate_set_sha256"
    )
    with pytest.raises(ValueError, match="Generator source"):
        replay_brain_intersecting_candidates(replay_tamper, source_tamper, support)


def test_train_and_development_rng_domains_do_not_duplicate_geometry():
    support = _support()
    train = make_arbitrary_plane_manifest(64, "train", 99173, support)
    development = make_arbitrary_plane_manifest(64, "development", 99173, support)

    assert train["rng"]["split_domain_seed_uint64"] != development["rng"]["split_domain_seed_uint64"]
    assert {sample["plane_geometry_sha256"] for sample in train["samples"]}.isdisjoint(
        sample["plane_geometry_sha256"] for sample in development["samples"]
    )
    with pytest.raises(ValueError, match="Development-stage"):
        make_arbitrary_plane_manifest(2, "test", 99173, support)


def test_reference_and_stress_offsets_are_intersecting_named_strata():
    support = _support()
    manifest = make_arbitrary_plane_manifest(
        100,
        "development",
        57201,
        support,
        stress_fraction=0.25,
        stress_boundary_fraction=0.08,
    )
    strata = [sample["stratum"] for sample in manifest["samples"]]

    assert strata.count(STRESS_STRATUM) == 25
    assert strata.count(REFERENCE_STRATUM) == 75
    assert manifest["sampling"]["intersection_contract"] == "plane intersects at least one occupied annotation voxel box"
    assert "orientation-balanced" in manifest["sampling"]["reference_measure"]
    assert "not the Crofton" in manifest["sampling"]["reference_measure"]
    assert manifest["sampling"]["finite_raster_support_status"].startswith("not evaluated")
    assert "full brain-support projection" in manifest["sampling"]["strata"][REFERENCE_STRATUM]["description"]
    assert "deliberate" in manifest["sampling"]["strata"][STRESS_STRATUM]["description"]
    for sample in manifest["samples"]:
        fraction = sample["offset_fraction_of_support_projection"]
        if sample["stratum"] == REFERENCE_STRATUM:
            assert 0.0 <= fraction <= 1.0
            assert sample["stress_projection_side"] is None
        else:
            assert fraction <= 0.08 or fraction >= 0.92
            assert sample["stress_projection_side"] in {"lower", "upper"}
            assert (fraction <= 0.08) == (sample["stress_projection_side"] == "lower")
            side_seed = int(sample["rng"]["field_stream_seed_uint64"]["stress_side"], 16)
            expected_side = (
                "lower"
                if int(np.random.Generator(np.random.PCG64(side_seed)).integers(0, 2)) == 0
                else "upper"
            )
            assert sample["stress_projection_side"] == expected_side
        intersects, hits = plane_intersects_annotation_support(
            sample["normal_rp2"], sample["signed_offset_um"], support
        )
        assert intersects
        assert hits == sample["support"]["intersecting_voxel_count"] > 0
        projection = support_projection(sample["normal_rp2"], support)
        assert projection["projection_sha256"] == sample["support"]["projection_sha256"]

    with pytest.raises(ValueError, match="frozen reference measure"):
        make_arbitrary_plane_manifest(
            2,
            "development",
            57201,
            support,
            reference_fraction_bounds=(0.1, 0.9),
        )


def test_real_subject_ids_are_not_replaced_by_plane_realization_ids():
    manifest = make_arbitrary_plane_manifest(
        3,
        "development",
        17,
        _support(),
        animal_id="animal-07",
        specimen_id="specimen-07a",
        experiment_id="experiment-071",
    )

    assert manifest["provenance"]["animal_id"] == "animal-07"
    assert manifest["provenance"]["specimen_id"] == "specimen-07a"
    assert manifest["provenance"]["experiment_id"] == "experiment-071"
    assert all(sample["animal_id"] == "animal-07" for sample in manifest["samples"])
    assert all(sample["specimen_id"] == "specimen-07a" for sample in manifest["samples"])
    assert all(sample["experiment_id"] == "experiment-071" for sample in manifest["samples"])
    assert all(sample["plane_realization_id"] for sample in manifest["samples"])
    assert all("synthetic_realization_id" not in sample for sample in manifest["samples"])


def test_candidates_use_geodesic_and_physical_deltas_and_all_intersect_brain():
    support = _support()
    manifest = make_arbitrary_plane_manifest(12, "train", 3471, support, stress_fraction=0.0)
    sample = manifest["samples"][5]
    first = make_brain_intersecting_candidates(
        manifest,
        5,
        support,
        32,
        8117,
        max_geodesic_deg=12.0,
        max_offset_um=35.0,
    )
    repeated = make_brain_intersecting_candidates(
        manifest,
        5,
        support,
        32,
        8117,
        max_geodesic_deg=12.0,
        max_offset_um=35.0,
    )

    assert first == repeated
    assert replay_brain_intersecting_candidates(first, manifest, support) == first
    assert first["schema_version"] == CANDIDATE_SCHEMA
    center = first["candidates"][first["center_candidate_index"]]
    assert center["normal_delta_logmap_ap_dv_ml_rad"] == [0.0, 0.0, 0.0]
    assert center["offset_delta_um"] == 0.0
    assert center["normal_rp2"] == sample["normal_rp2"]
    assert center["signed_offset_um"] == sample["signed_offset_um"]
    assert first["scope"] == "unoriented-infinite-plane-only"
    assert "parallel-transport" in first["finite_raster_frame_status"]
    assert "in_plane_roll" in first["excluded_degrees_of_freedom"]
    assert "not calibrated posterior mass" in first["posterior_use_status"]
    assert all("roll_rad" not in candidate for candidate in first["candidates"])
    assert len({candidate["candidate_id"] for candidate in first["candidates"]}) == 32
    assert len({candidate["plane_pose_sha256"] for candidate in first["candidates"]}) == 32
    for candidate in first["candidates"]:
        assert candidate["normal_geodesic_rad"] <= np.deg2rad(12.0) + 1e-12
        assert abs(candidate["offset_delta_um"]) <= 35.0 + 1e-12
        expected_geodesic = np.arccos(
            np.clip(abs(np.dot(sample["normal_rp2"], candidate["normal_rp2"])), 0.0, 1.0)
        )
        assert candidate["normal_geodesic_rad"] == pytest.approx(expected_geodesic, abs=1e-12)
        logmap = np.asarray(candidate["normal_delta_logmap_ap_dv_ml_rad"])
        assert np.linalg.norm(logmap) == pytest.approx(candidate["normal_geodesic_rad"], abs=1e-10)
        assert np.dot(logmap, sample["normal_rp2"]) == pytest.approx(0.0, abs=1e-10)
        assert candidate["brain_intersection"]
        assert candidate["intersecting_voxel_count"] > 0
        normal = np.asarray(candidate["normal_rp2"])
        pivot = np.argmax(np.abs(normal))
        assert normal[pivot] > 0.0


def test_candidate_builder_rejects_tampered_standalone_sample_and_shuffles_center():
    support = _support()
    manifest = make_arbitrary_plane_manifest(6, "train", 723, support, stress_fraction=0.0)
    tampered = json.loads(json.dumps(manifest))
    tampered["samples"][2]["signed_offset_um"] += 1.0

    with pytest.raises(ValueError, match="manifest_sha256"):
        make_brain_intersecting_candidates(
            tampered, 2, support, 8, 991, max_geodesic_deg=5.0, max_offset_um=20.0
        )

    center_indices = {
        make_brain_intersecting_candidates(
            manifest, 2, support, 8, seed, max_geodesic_deg=5.0, max_offset_um=20.0
        )["center_candidate_index"]
        for seed in range(8)
    }
    assert len(center_indices) > 1


def test_support_points_are_bound_to_the_support_hash():
    support = _support()
    changed = {**support, "points_um": np.asarray(support["points_um"]).copy()}
    changed["points_um"][0, 0] += 1.0

    with pytest.raises(ValueError, match="occupied_points_um_sha256"):
        make_arbitrary_plane_manifest(2, "train", 11, changed)


def test_local_plane_delta_is_continuous_across_the_rp2_fold_seam():
    base_raw = np.asarray([1.0, -0.99, 0.0])
    base_raw /= np.linalg.norm(base_raw)
    candidate_raw = np.asarray([0.99, -1.0, 0.0])
    candidate_raw /= np.linalg.norm(candidate_raw)
    base_normal, base_offset, _ = canonicalize_plane(base_raw, 10.0)
    candidate_normal, candidate_offset, _ = canonicalize_plane(candidate_raw, 12.0)
    equivalent_normal, equivalent_offset, _ = canonicalize_plane(
        -candidate_normal, -candidate_offset
    )
    delta = rp2_geodesic_plane_delta(
        base_normal, base_offset, candidate_normal, candidate_offset
    )
    equivalent_delta = rp2_geodesic_plane_delta(
        base_normal, base_offset, equivalent_normal, equivalent_offset
    )

    assert np.dot(base_normal, candidate_normal) < 0.0
    assert delta == equivalent_delta
    assert delta["rp2_alignment_sign_to_base"] == -1
    assert delta["normal_geodesic_rad"] == pytest.approx(
        np.arccos(abs(np.dot(base_normal, candidate_normal))), abs=1e-12
    )
    assert delta["offset_delta_um"] == pytest.approx(2.0)


def test_coordinate_free_logmap_has_no_axis_choice_seam():
    deltas = []
    for epsilon in (-1e-12, 1e-12):
        base = np.asarray([0.2 + epsilon, 0.2 - epsilon, 0.959166304])
        base /= np.linalg.norm(base)
        direction = np.asarray([0.0, 0.0, 1.0])
        direction -= np.dot(direction, base) * base
        direction /= np.linalg.norm(direction)
        candidate = np.cos(0.01) * base + np.sin(0.01) * direction
        deltas.append(
            np.asarray(
                rp2_geodesic_plane_delta(base, 0.0, candidate, 0.0)[
                    "normal_delta_logmap_ap_dv_ml_rad"
                ]
            )
        )

    assert np.linalg.norm(deltas[0] - deltas[1]) < 1e-12


def test_replay_rejects_different_annotation_support():
    support = _support()
    manifest = make_arbitrary_plane_manifest(4, "development", 101, support)
    changed_annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    changed_annotation[2:15, 2:13, 2:11] = 1

    with pytest.raises(ValueError, match="support"):
        replay_arbitrary_plane_manifest(manifest, _support(changed_annotation))


def test_integer_and_numpy_scalar_contract_inputs_replay_with_identical_hashes():
    support = _support()
    manifest = make_arbitrary_plane_manifest(
        np.int64(8),
        np.str_("train"),
        np.int64(7601),
        support,
        stress_fraction=np.float32(0.25),
        reference_fraction_bounds=np.asarray([0, 1], dtype=np.int32),
        stress_boundary_fraction=np.float32(0.1),
        animal_id=np.int64(17),
        specimen_id=np.int32(23),
        experiment_id=np.int16(29),
        max_rejection_attempts=np.int32(512),
    )
    candidates = make_brain_intersecting_candidates(
        manifest,
        0,
        support,
        np.int64(16),
        np.int32(9811),
        max_geodesic_deg=35,
        max_offset_um=100,
        include_center=np.bool_(True),
        max_rejection_attempts=np.int64(1024),
    )

    assert replay_arbitrary_plane_manifest(manifest, support) == manifest
    assert replay_brain_intersecting_candidates(candidates, manifest, support) == candidates
    assert type(manifest["count"]) is int
    assert type(manifest["sampling"]["stress_fraction"]) is float
    assert type(manifest["sampling"]["max_rejection_attempts"]) is int
    assert type(manifest["provenance"]["animal_id"]) is int
    assert type(manifest["provenance"]["experiment_id"]) is int
    assert type(candidates["max_geodesic_deg"]) is float
    assert type(candidates["max_offset_um"]) is float
    assert candidates["root_seed"] == "0x0000000000002653"
