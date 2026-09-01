import copy

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.run_subject_deformed_slab_qualification_v2 as runner
import training.run_subject_deformed_slab_multiresolution_v2 as replacement_runner
import training.subject_deformed_slab_multiresolution_assessment_v2 as multiresolution
import training.subject_deformed_slab_multiresolution_bundle_v2 as multiresolution_bundle
import training.subject_deformed_slab_qualification_v2 as qualification
import training.verify_subject_deformed_slab_multiresolution_bundle_v2 as independent_verifier
from training.arbitrary_plane_support import build_annotation_support_index


def _prepared_context(offset=0.0):
    annotation = np.zeros((9, 8, 7), dtype=np.uint16)
    annotation[1:8, 1:7, 1:6] = 7
    ap, dv, ml = np.indices(annotation.shape)
    scalar = (offset + 1 + ap + 2 * dv + 3 * ml).astype(np.float32)
    support = build_annotation_support_index(
        annotation,
        atlas_id=f"subject-deformed-slab-fixture-{offset}",
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
        scalar_source_sha256=("4" if offset == 0.0 else "5") * 64,
        template_decoder="fixture decoder",
        annotation_decoder="fixture decoder",
    )


@pytest.fixture(scope="module")
def prepared():
    return _prepared_context()


def _fixture_records():
    panel = qualification.subject_deformed_slab_qualification_panel_v2()
    animals = []
    for index, manifest in enumerate(panel["animals"]):
        animals.append(
            {
                "animal_manifest": copy.deepcopy(manifest),
                "subject_deformation_plan_id": f"deformation-plan-{index}",
                "subject_deformation_realization_id": f"deformation-realization-{index}",
                "synthetic_animal_id": f"synthetic-animal-{index}",
                "resolved_config": {"fixture_animal_index": manifest["animal_index"]},
                "rng_sources": {"fixture_seed": manifest["subject_deformation_root_seed_uint64"]},
                "receipt": qualification._receipt_binding(
                    {"stage": "deformation", "animal_index": manifest["animal_index"]}
                ),
            }
        )
    animal_records = {value["animal_manifest"]["animal_index"]: value for value in animals}
    cases = []
    for index, spec in enumerate(panel["cases"]):
        animal = next(
            value for value in panel["animals"] if value["animal_index"] == spec["animal_index"]
        )
        animal_record = animal_records[spec["animal_index"]]
        centre = np.full((2, 3), index, dtype=np.float32)
        centre_physical = centre.astype(np.float64) * 25.0
        labels = np.full((2,), index + 1, dtype=np.int64)
        support = np.asarray([True, False])
        centre_invariance = {
            name: {
                "coarse_receipt": acquisition._array_receipt(array),
                "refined_receipt": acquisition._array_receipt(array),
                "byte_identical": True,
            }
            for name, array in (
                ("mapped_allen_coordinates", centre),
                ("mapped_physical_coordinates", centre_physical),
                ("annotations", labels),
                ("support", support),
            )
        }
        centre_invariance["all_centre_targets_byte_identical"] = True
        metrics = {name: 0.0 for name in qualification._METRIC_KEYS}
        accepted_attempt_index = index % 2
        attempt_root_seed_uint64 = f"0x{index + 1:016x}"
        precursor_references = {}
        for level, axial_step_um_max in (
            ("coarse", qualification.COARSE_AXIAL_STEP_UM_MAX),
            ("refined", qualification.REFINED_AXIAL_STEP_UM_MAX),
        ):
            precursor_references[level] = {
                "v2_plane_realization_id": f"pose-{index}",
                "centre_plane_render_id": f"centre-{index}",
                "slab_render_id": f"{level}-precursor-{index}",
                "receipt": qualification._receipt_binding(
                    {
                        "generator": {
                            "resolved_config": {
                                "root_seed_uint64": attempt_root_seed_uint64,
                                "sample_index": spec["section_index"],
                                "plane_stratum": spec["plane_stratum"],
                            }
                        },
                        "slab_recipe": {
                            "axial_step_um_max": axial_step_um_max,
                        },
                    }
                ),
            }
        deformation_reference = {
            "subject_deformation_plan_id": animal_record[
                "subject_deformation_plan_id"
            ],
            "subject_deformation_realization_id": animal_record[
                "subject_deformation_realization_id"
            ],
        }
        subject_slab_references = {
            level: {
                "subject_coordinate_map_id": f"{level}-coordinate-{index}",
                "subject_slab_render_id": f"{level}-subject-slab-{index}",
                "receipt": qualification._receipt_binding(
                    {
                        "coordinate_identity_payload": {
                            "deformation_reference": deformation_reference,
                            "synthetic_animal_id": animal_record[
                                "synthetic_animal_id"
                            ],
                        }
                    }
                ),
            }
            for level in ("coarse", "refined")
        }
        case = {
            "schema_version": qualification.SUBJECT_DEFORMED_SLAB_CASE_V2_SCHEMA,
            "case_spec": copy.deepcopy(spec),
            "animal_reference": {
                "animal_id": animal["animal_id"],
                "animal_index": animal["animal_index"],
                "specimen_id": animal["specimen_id"],
                "experiment_id": animal["experiment_id"],
                "subject_deformation_plan_id": animal_record[
                    "subject_deformation_plan_id"
                ],
                "subject_deformation_realization_id": animal_record[
                    "subject_deformation_realization_id"
                ],
                "synthetic_animal_id": animal_record["synthetic_animal_id"],
            },
            "support_resolution_reference": {
                "support_resolution_plan_id": f"support-plan-{index}",
                "subject_support_resolution_id": f"support-resolution-{index}",
                "accepted_attempt_index": accepted_attempt_index,
                "configuration": {
                    "master_root_seed_uint64": spec[
                        "support_master_root_seed_uint64"
                    ],
                    "split_index": spec["split_index"],
                    "animal_index": spec["animal_index"],
                    "section_index": spec["section_index"],
                    "plane_stratum": spec["plane_stratum"],
                    "nominal_cut_thickness_um": 55.0,
                    "axial_step_um_max": 12.5,
                    "parent_shape_h_w": [256, 256],
                    "max_attempts": 8,
                },
                "lineage": {
                    "split": "development",
                    "animal_id": animal["animal_id"],
                    "animal_index": animal["animal_index"],
                    "specimen_id": animal["specimen_id"],
                    "experiment_id": animal["experiment_id"],
                },
                "accepted_attempt_seed": {
                    "master_root_seed_uint64": spec[
                        "support_master_root_seed_uint64"
                    ],
                    "split_index": spec["split_index"],
                    "animal_index": spec["animal_index"],
                    "section_index": spec["section_index"],
                    "attempt_index": accepted_attempt_index,
                    "attempt_root_seed_uint64": attempt_root_seed_uint64,
                },
                "accepted_precursor_reference": {
                    "slab_render_id": precursor_references["coarse"][
                        "slab_render_id"
                    ],
                    "receipt_sha256": precursor_references["coarse"]["receipt"][
                        "receipt_sha256"
                    ],
                },
                "accepted_probe_reference": {"probe_id": f"probe-{index}"},
                "receipt": qualification._receipt_binding(
                    {"stage": "support-resolution", "case_index": index}
                ),
            },
            "precursor_references": precursor_references,
            "subject_slab_references": subject_slab_references,
            "same_pose_attempt": True,
            "axial_steps_um_max": {"coarse": 12.5, "refined": 6.25},
            "centre_invariance": centre_invariance,
            "union_nonzero_support_pixel_count": 2,
            "authenticated_scalar_range_denominator": 100.0,
            "metrics": metrics,
            "thresholds": dict(qualification.THRESHOLDS),
        }
        cases.append(qualification._finalize_case(case))
    return animals, cases


@pytest.fixture
def stubbed_panel(monkeypatch):
    animals, cases = _fixture_records()

    def run(prepared_context, *, batch_size):
        return copy.deepcopy(animals), copy.deepcopy(cases)

    monkeypatch.setattr(qualification, "_run_predeclared_panel_v2", run)
    return animals, cases


def _rereceipt(report):
    for case in report["cases"]:
        case["passed"] = bool(
            case["centre_invariance"]["all_centre_targets_byte_identical"]
            and case["same_pose_attempt"]
            and qualification._metric_pass(case["metrics"])
        )
        case["case_receipt_sha256"] = acquisition._payload_sha256(
            {key: value for key, value in case.items() if key != "case_receipt_sha256"}
        )
    report["all_cases_passed"] = all(case["passed"] for case in report["cases"])
    report["qualification_receipt_sha256"] = acquisition._payload_sha256(
        {
            key: value
            for key, value in report.items()
            if key != "qualification_receipt_sha256"
        }
    )


def test_panel_is_small_development_only_and_spans_required_strata():
    panel = qualification.subject_deformed_slab_qualification_panel_v2()
    assert panel["split"] == "development"
    assert len(panel["animals"]) == 2
    assert len(panel["cases"]) == 6
    assert {case["plane_stratum"] for case in panel["cases"]} == {
        "reference",
        "near_AP",
        "near_DV",
        "near_ML",
        "general_oblique",
        "edge_or_partial",
    }
    assert len({animal["animal_id"] for animal in panel["animals"]}) == 2
    assert all(animal["specimen_id"] and animal["experiment_id"] for animal in panel["animals"])


def test_small_array_comparison_and_centre_invariance():
    def raster(delta):
        occupancy = np.asarray([[1.0, 0.5 + delta], [0.01, 0.0]], np.float32)
        return {
            "scalar": np.asarray([[10.0, 20.0 + delta], [5.0, 0.0]], np.float32),
            "slab_brain_occupancy": occupancy,
            "slab_label_purity": occupancy,
            "centre_label_support_weight": occupancy,
            "slab_modal_annotation": np.asarray([[1, 1], [2, 0]], np.int64),
            "slab_observable_support_mask": occupancy > 0,
            "slab_supervision_weight_or_abstention": {
                "dense_correspondence_weight": occupancy,
                "abstention_mask": occupancy == 0,
            },
        }

    metrics, count = qualification._comparison_metrics(raster(0.0), raster(0.01), 100.0)
    assert count == 3
    assert set(metrics) == qualification._METRIC_KEYS
    assert qualification._metric_pass(metrics)

    mapped = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    mapped_physical = mapped.astype(np.float64) * 25.0
    labels = np.asarray([[1, 0], [2, 2]], np.int64)
    support = labels != 0

    def stage(offset_count, centre_index):
        mapped_stack = np.zeros((offset_count, 2, 2, 3), np.float32)
        physical_stack = np.zeros((offset_count, 2, 2, 3), np.float64)
        label_stack = np.zeros((offset_count, 2, 2), np.int64)
        mapped_stack[centre_index] = mapped
        physical_stack[centre_index] = mapped_physical
        label_stack[centre_index] = labels
        return {
            "coordinate_map": {
                "kernel": {"centre_index": centre_index},
                "arrays": {
                    "mapped_allen_index_coordinates_float32": mapped_stack,
                    "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64": physical_stack,
                },
            },
            "sample_arrays": {"annotation_samples_int64": label_stack},
            "raster": {"centre_plane_support_mask": support},
        }

    invariant = qualification._centre_invariance(stage(3, 1), stage(5, 2))
    assert invariant["all_centre_targets_byte_identical"] is True
    assert all(
        invariant[name]["byte_identical"]
        for name in (
            "mapped_allen_coordinates",
            "mapped_physical_coordinates",
            "annotations",
            "support",
        )
    )


def test_deterministic_replay_and_provenance_binding(prepared, stubbed_panel):
    report = qualification.evaluate_subject_deformed_slab_qualification_v2(prepared)
    qualification.verify_subject_deformed_slab_qualification_v2(report, prepared)
    assert report["animal_count"] == 2
    assert report["case_count"] == 6
    assert report["all_cases_passed"] is True
    assert report["cohort_policy"] == {
        "development_only": True,
        "benchmark_animals_used": False,
        "final_test_animals_used": False,
        "full_benchmark": False,
    }
    assert not any(report["learned_dependencies"].values())
    assert "synthetic_realization_id" not in repr(report)
    assert report["qualification_receipt_sha256"] == acquisition._payload_sha256(
        {
            key: value
            for key, value in report.items()
            if key != "qualification_receipt_sha256"
        }
    )


@pytest.mark.parametrize(
    "tamper",
    ["metric", "centre", "upstream-receipt", "case-order", "final-id", "receipt"],
)
def test_strict_tamper_rejection(prepared, stubbed_panel, tamper):
    report = qualification.evaluate_subject_deformed_slab_qualification_v2(prepared)
    changed = copy.deepcopy(report)
    if tamper == "metric":
        changed["cases"][0]["metrics"]["normalized_scalar_mae"] = 0.001
        _rereceipt(changed)
    elif tamper == "centre":
        changed["cases"][0]["centre_invariance"]["support"][
            "refined_receipt"
        ]["array_sha256"] = "f" * 64
        _rereceipt(changed)
    elif tamper == "upstream-receipt":
        binding = changed["cases"][0]["precursor_references"]["coarse"]["receipt"]
        binding["receipt_payload"]["coherent_tamper"] = True
        binding["receipt_sha256"] = acquisition._payload_sha256(
            binding["receipt_payload"]
        )
        _rereceipt(changed)
    elif tamper == "case-order":
        changed["cases"][0], changed["cases"][1] = changed["cases"][1], changed["cases"][0]
        _rereceipt(changed)
    elif tamper == "final-id":
        changed["synthetic_realization_id"] = "forbidden"
        _rereceipt(changed)
    else:
        changed["qualification_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        qualification.verify_subject_deformed_slab_qualification_v2(changed, prepared)


def test_report_is_bound_to_context(prepared, stubbed_panel):
    report = qualification.evaluate_subject_deformed_slab_qualification_v2(prepared)
    with pytest.raises(ValueError):
        qualification.verify_subject_deformed_slab_qualification_v2(
            report, _prepared_context(offset=1.0)
        )


def test_real_runner_requires_opt_in_before_atlas_io(monkeypatch):
    calls = []
    monkeypatch.delenv(runner.OPT_IN_ENVIRONMENT, raising=False)
    monkeypatch.setattr(runner, "_file_sha256", lambda path: calls.append(("hash", path)))
    monkeypatch.setattr(
        runner.nrrd, "read", lambda *args, **kwargs: calls.append(("read", args))
    )

    with pytest.raises(PermissionError):
        runner.main()
    assert calls == []


def test_real_runner_cannot_claim_the_rejected_legacy_gate_passed(monkeypatch):
    calls = []
    monkeypatch.setenv(runner.OPT_IN_ENVIRONMENT, "1")
    monkeypatch.setattr(runner, "_file_sha256", lambda path: calls.append(("hash", path)))
    monkeypatch.setattr(
        runner.nrrd,
        "read", lambda *args, **kwargs: calls.append(("read", args))
    )

    with pytest.raises(RuntimeError, match="multiresolution replacement gate"):
        runner.main()
    assert calls == []


def _failed_legacy_report(prepared, monkeypatch):
    report = qualification.evaluate_subject_deformed_slab_qualification_v2(prepared)
    report["cases"][1]["metrics"]["dense_correspondence_weight_absolute_error_p99"] = 1.0
    report["cases"][4]["metrics"]["slab_modal_annotation_disagreement_fraction"] = 1.0
    _rereceipt(report)
    monkeypatch.setattr(
        multiresolution.legacy,
        "evaluate_subject_deformed_slab_qualification_v2",
        lambda context, *, batch_size=None: copy.deepcopy(report),
    )
    return report


def _fixture_raster(step_index, arm_index):
    scalar = np.arange(16, dtype=np.float32).reshape(4, 4) + arm_index
    scalar[0, 0] += np.float32(0.05 * step_index)
    centre = np.ones((4, 4), dtype=np.int64)
    centre[3, 3] = 0
    centre_support = centre != 0
    occupancy = centre_support.astype(np.float32)
    occupancy[0, 0] = np.float32(0.55 + 0.05 * step_index)
    purity = centre_support.astype(np.float32)
    purity[1, 1] = np.float32(0.6 + 0.05 * step_index)
    centre_weight = purity.copy()
    dense = np.where(
        centre_support,
        np.clip((centre_weight.astype(np.float64) - 0.5) / 0.3, 0.0, 1.0),
        0.0,
    ).astype(np.float32)
    modal = centre.copy()
    modal[1, 1] = 2
    return {
        "scalar": scalar,
        "centre_plane_annotation": centre,
        "centre_plane_support_mask": centre_support,
        "slab_brain_occupancy": occupancy,
        "slab_observable_support_mask": occupancy > 0,
        "slab_modal_annotation": modal,
        "slab_label_purity": purity,
        "centre_label_support_weight": centre_weight,
        "slab_supervision_weight_or_abstention": {
            "dense_correspondence_weight": dense,
            "abstention_mask": (~centre_support) | (centre_weight <= 0.5),
        },
    }


def _fixture_precursor(plan, step, step_index):
    pose = plan["selected_first_failure"]["pose_binding"]
    legacy_name = {
        12.5: "legacy_12.5_um_precursor_reference",
        6.25: "legacy_6.25_um_precursor_reference",
    }.get(step)
    legacy_reference = None if legacy_name is None else pose[legacy_name]
    raster = _fixture_raster(step_index, 0)
    raster.update(multiresolution.slab._slab_raster_metadata(raster))
    return {
        "v2_plane_realization_id": pose["v2_plane_realization_id"],
        "centre_plane_render_id": pose["centre_plane_render_id"],
        "slab_render_id": (
            legacy_reference["slab_render_id"]
            if legacy_reference is not None
            else f"precursor-{step:g}"
        ),
        "receipt_sha256": (
            legacy_reference["receipt"]["receipt_sha256"]
            if legacy_reference is not None
            else f"precursor-receipt-{step:g}"
        ),
        "slab_recipe": {"axial_step_um_max": step},
        "raster": raster,
    }


def _fixture_subject_artifact(precursor, step_index, arm_index):
    depth = 3
    raster = _fixture_raster(step_index, arm_index)
    centre = raster["centre_plane_annotation"]
    labels = np.stack([centre, centre, centre]).astype(np.int64)
    labels[0, 0, 0] = 0
    labels[:, 1, 1] = np.asarray([1, 2, 2])
    scalar_samples = np.stack(
        [raster["scalar"] - 1, raster["scalar"], raster["scalar"] + 1]
    ).astype(np.float32)
    base_coordinates = np.indices((4, 4)).transpose(1, 2, 0).astype(np.float32)
    base_coordinates = np.concatenate(
        [
            np.full((4, 4, 1), arm_index, np.float32),
            base_coordinates,
        ],
        axis=2,
    )
    coordinates = np.stack(
        [base_coordinates - step_index, base_coordinates, base_coordinates + step_index]
    ).astype(np.float32)
    physical = coordinates.astype(np.float64) * 25.0
    coordinate_arrays = {
        "subject_renderer_centres_allen_index_float32": coordinates.copy(),
        "subject_allen_index_coordinates_float32": coordinates.copy(),
        "subject_physical_coordinates_ap_dv_ml_um_float64": physical.copy(),
        "mapped_ccf_physical_coordinates_ap_dv_ml_um_float64": physical.copy(),
        "mapped_allen_index_coordinates_float32": coordinates.copy(),
    }
    sample_arrays = {
        "scalar_samples_float32": scalar_samples,
        "annotation_samples_int64": labels,
    }
    raster_arrays = multiresolution.subject_slab._reduced_arrays(raster)
    identity = arm_index == 1
    return {
        "identity_reference_path": identity,
        "precursor_reference": {"slab_render_id": precursor["slab_render_id"]},
        "subject_coordinate_map_id": f"coordinate-{arm_index}-{step_index}",
        "subject_slab_render_id": f"render-{arm_index}-{step_index}",
        "receipt_sha256": f"render-receipt-{arm_index}-{step_index}",
        "coordinate_map": {
            "kernel": {"centre_index": depth // 2},
            "arrays": coordinate_arrays,
            "array_receipts": {
                name: acquisition._array_receipt(array)
                for name, array in coordinate_arrays.items()
            },
        },
        "sample_arrays": sample_arrays,
        "sample_array_receipts": {
            name: acquisition._array_receipt(array)
            for name, array in sample_arrays.items()
        },
        "raster": raster,
        "raster_array_receipts": {
            name: acquisition._array_receipt(array)
            for name, array in raster_arrays.items()
        },
    }


def _fixture_multiresolution_renders(plan):
    precursors = {}
    renders = {arm: {} for arm in multiresolution.ARM_NAMES}
    for step_index, step in enumerate(multiresolution.AXIAL_STEPS_UM_MAX):
        key = f"{step:g}"
        precursor = _fixture_precursor(plan, step, step_index)
        precursors[key] = precursor
        for arm_index, arm in enumerate(multiresolution.ARM_NAMES):
            renders[arm][key] = _fixture_subject_artifact(
                precursor, step_index, arm_index
            )
    return {"precursors": precursors, "renders": renders}


def test_fixed_case_plan_binds_first_failure_and_has_no_thresholds(
    prepared, stubbed_panel, monkeypatch
):
    report = _failed_legacy_report(prepared, monkeypatch)
    plan = multiresolution.make_fixed_case_multiresolution_plan_v2(report, prepared)
    multiresolution.verify_fixed_case_multiresolution_plan_v2(plan, report, prepared)
    selected = plan["selected_first_failure"]
    assert selected["case_order_index"] == 1
    assert selected["case_receipt_sha256"] == report["cases"][1][
        "case_receipt_sha256"
    ]
    assert selected["support_attempt_binding"] == report["cases"][1][
        "support_resolution_reference"
    ]
    assert plan["render_contract"]["redraw_allowed"] is False
    assert plan["render_contract"]["axial_steps_um_max"] == [
        12.5,
        6.25,
        3.125,
        1.5625,
    ]
    assert set(plan["render_contract"]["arms"]) == set(multiresolution.ARM_NAMES)
    assert plan["analysis_contract"]["acceptance_thresholds"] is None
    assert set(plan["analysis_contract"]["metric_families"]) == set(
        multiresolution.METRIC_FAMILIES
    )
    assert set(plan["analysis_contract"]["strata"]) == set(
        multiresolution.STRATUM_NAMES
    )
    changed = copy.deepcopy(plan)
    changed["selected_first_failure"]["support_attempt_binding"][
        "accepted_attempt_index"
    ] += 1
    changed["plan_receipt_sha256"] = acquisition._payload_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "plan_receipt_sha256"
        }
    )
    with pytest.raises(ValueError):
        multiresolution.verify_fixed_case_multiresolution_plan_v2(
            changed, report, prepared
        )


def test_eight_render_orchestrator_reuses_pose_attempt_and_has_identity_control(
    prepared, stubbed_panel, monkeypatch
):
    report = _failed_legacy_report(prepared, monkeypatch)
    plan = multiresolution.make_fixed_case_multiresolution_plan_v2(report, prepared)
    binding = plan["selected_first_failure"]["subject_deformation_reference"]
    subject_plan = {
        "resolved_config": {"deformation_stratum": "standard"},
        "subject_deformation_plan_id": binding["subject_deformation_plan_id"],
        "subject_deformation_realization_id": binding[
            "subject_deformation_realization_id"
        ],
        "synthetic_animal_id": binding["synthetic_animal_id"],
    }
    public_receipt = {
        **copy.deepcopy(binding["receipt"]["receipt_payload"]),
        "receipt_sha256": binding["receipt"]["receipt_sha256"],
    }
    calls = []

    monkeypatch.setattr(
        multiresolution.deformation,
        "subject_deformation_plan_receipt_v2",
        lambda value: copy.deepcopy(public_receipt),
    )

    def make_precursor(context, split, root_seed, sample_index, stratum, **kwargs):
        step = kwargs["axial_step_um_max"]
        calls.append(("precursor", step, root_seed, sample_index, stratum))
        return _fixture_precursor(
            plan, step, multiresolution.AXIAL_STEPS_UM_MAX.index(step)
        )

    monkeypatch.setattr(
        multiresolution.slab,
        "make_v2_generic_global_reference_slab_render",
        make_precursor,
    )
    monkeypatch.setattr(
        multiresolution.slab,
        "verify_v2_generic_global_reference_slab_render",
        lambda *args, **kwargs: None,
    )

    def precursor_receipt(precursor):
        step = precursor["slab_recipe"]["axial_step_um_max"]
        name = {
            12.5: "legacy_12.5_um_precursor_reference",
            6.25: "legacy_6.25_um_precursor_reference",
        }[step]
        return copy.deepcopy(
            plan["selected_first_failure"]["pose_binding"][name]["receipt"][
                "receipt_payload"
            ]
        )

    monkeypatch.setattr(
        multiresolution.slab,
        "v2_generic_slab_render_receipt",
        precursor_receipt,
    )

    def make_subject(context, precursor, *, subject_plan, batch_size=None):
        step = precursor["slab_recipe"]["axial_step_um_max"]
        arm_index = int(subject_plan is None)
        calls.append(("subject", step, arm_index))
        return _fixture_subject_artifact(
            precursor,
            multiresolution.AXIAL_STEPS_UM_MAX.index(step),
            arm_index,
        )

    monkeypatch.setattr(
        multiresolution.subject_slab, "make_subject_slab_render_v2", make_subject
    )
    monkeypatch.setattr(
        multiresolution.subject_slab,
        "verify_subject_slab_render_v2",
        lambda *args, **kwargs: None,
    )
    rendered = multiresolution.render_fixed_case_multiresolution_v2(
        prepared, plan, subject_plan
    )
    assert list(rendered["precursors"]) == ["12.5", "6.25", "3.125", "1.5625"]
    assert [call[1] for call in calls if call[0] == "precursor"] == list(
        multiresolution.AXIAL_STEPS_UM_MAX
    )
    support = plan["selected_first_failure"]["support_attempt_binding"]
    assert all(
        call[2:]
        == (
            support["accepted_attempt_seed"]["attempt_root_seed_uint64"],
            support["configuration"]["section_index"],
            support["configuration"]["plane_stratum"],
        )
        for call in calls
        if call[0] == "precursor"
    )
    assert [(call[1], call[2]) for call in calls if call[0] == "subject"] == [
        (step, arm_index)
        for step in multiresolution.AXIAL_STEPS_UM_MAX
        for arm_index in (0, 1)
    ]


def test_threshold_free_assessment_binds_raw_arrays_and_stratifies_metrics(
    prepared, stubbed_panel, monkeypatch
):
    report = _failed_legacy_report(prepared, monkeypatch)
    plan = multiresolution.make_fixed_case_multiresolution_plan_v2(report, prepared)
    rendered = _fixture_multiresolution_renders(plan)
    assessment = multiresolution.assemble_fixed_case_multiresolution_assessment_v2(
        plan, rendered
    )
    multiresolution.verify_fixed_case_multiresolution_assessment_v2(
        assessment, plan, rendered
    )
    assert len(assessment["render_records"]) == 8
    assert assessment["acceptance_thresholds"] is None
    assert assessment["scientific_decision"] == (
        "not_evaluated_pending_threshold_predeclaration"
    )
    for arm in multiresolution.ARM_NAMES:
        assert assessment["centre_invariance_by_arm"][arm][
            "all_centre_targets_byte_identical_across_all_levels"
        ]
        measurements = assessment["stratified_measurements_by_arm"][arm]
        assert set(measurements["strata"]) == set(multiresolution.STRATUM_NAMES)
        assert all(
            measurements["strata"][name]["pixel_count"] > 0
            for name in multiresolution.STRATUM_NAMES
        )
        for level in measurements["comparisons_to_1.5625_um"].values():
            for stratum in level.values():
                assert set(stratum) == {"pixel_count", *multiresolution.METRIC_FAMILIES}
    assert all(
        set(record["raw_array_receipts"])
        == {
            "precursor_raster",
            "subject_coordinate_map",
            "subject_samples",
            "subject_raster",
        }
        for record in assessment["render_records"]
    )
    changed = copy.deepcopy(assessment)
    changed["acceptance_thresholds"] = {"invented": 1.0}
    changed["assessment_receipt_sha256"] = acquisition._payload_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "assessment_receipt_sha256"
        }
    )
    with pytest.raises(ValueError):
        multiresolution.verify_fixed_case_multiresolution_assessment_v2(
            changed, plan, rendered
        )
    rendered["renders"]["identity_control"]["3.125"]["raster"]["scalar"][0, 0] += 1
    with pytest.raises(ValueError, match="raw array receipt"):
        multiresolution.assemble_fixed_case_multiresolution_assessment_v2(
            plan, rendered
        )


def test_replacement_runner_requires_opt_in_and_external_empty_output(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.delenv(replacement_runner.OPT_IN_ENVIRONMENT, raising=False)
    monkeypatch.delenv(replacement_runner.OUTPUT_ENVIRONMENT, raising=False)
    monkeypatch.setattr(
        replacement_runner.bundle,
        "load_pinned_allen_context",
        lambda path: calls.append(path),
    )
    with pytest.raises(PermissionError):
        replacement_runner.main()
    monkeypatch.setenv(replacement_runner.OPT_IN_ENVIRONMENT, "1")
    with pytest.raises(ValueError, match="explicit absolute directory"):
        replacement_runner.main()
    monkeypatch.setenv(
        replacement_runner.OUTPUT_ENVIRONMENT,
        str(replacement_runner.ROOT / "forbidden-output"),
    )
    with pytest.raises(ValueError, match="outside the repository"):
        replacement_runner.main()
    monkeypatch.setenv(
        replacement_runner.OUTPUT_ENVIRONMENT, str(tmp_path / "already-exists")
    )
    (tmp_path / "already-exists").mkdir()
    with pytest.raises(FileExistsError):
        replacement_runner.main()
    assert calls == []


def test_replacement_pinned_decoder_is_checked_before_atlas_io(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(multiresolution_bundle.nrrd, "__version__", "different")
    monkeypatch.setattr(
        multiresolution_bundle,
        "_file_sha256",
        lambda path: calls.append(path),
    )
    with pytest.raises(ValueError, match="pinned decoder"):
        multiresolution_bundle.pinned_allen_inputs(tmp_path)
    assert calls == []


def test_replacement_runner_is_the_only_authorized_legacy_gate_exception(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setenv(replacement_runner.OPT_IN_ENVIRONMENT, "1")
    monkeypatch.setenv(
        replacement_runner.OUTPUT_ENVIRONMENT, str(tmp_path / "frozen")
    )
    monkeypatch.setattr(
        replacement_runner.gate_status,
        "legacy_gate_contract",
        lambda: {"decision": "legacy_gate_passed", "qualification_eligible": True},
    )
    monkeypatch.setattr(
        replacement_runner.bundle,
        "load_pinned_allen_context",
        lambda path: calls.append(path),
    )
    with pytest.raises(RuntimeError, match="does not authorize"):
        replacement_runner.main()
    assert calls == []


def test_replacement_runner_mocked_operational_order(monkeypatch, tmp_path):
    output = tmp_path / "frozen"
    staging = tmp_path / ".frozen.partial"
    context = object()
    allen_inputs = {"pinned": True}
    report = {"failed": True}
    plan = {"selected_first_failure": {"animal_manifest": {"animal": 1}}}
    subject_plan = {"subject": True}
    rendered = {"raw": True}
    result = {"assessment": True}
    calls = []
    monkeypatch.setenv(replacement_runner.OPT_IN_ENVIRONMENT, "1")
    monkeypatch.setenv(replacement_runner.OUTPUT_ENVIRONMENT, str(output))
    monkeypatch.setattr(
        replacement_runner.bundle,
        "load_pinned_allen_context",
        lambda path: calls.append("inputs") or (context, allen_inputs),
    )
    monkeypatch.setattr(
        replacement_runner.legacy,
        "evaluate_subject_deformed_slab_qualification_v2",
        lambda value: calls.append("legacy-report") or report,
    )
    monkeypatch.setattr(
        replacement_runner.assessment,
        "make_fixed_case_multiresolution_plan_v2",
        lambda failed, prepared: calls.append("plan") or plan,
    )
    monkeypatch.setattr(
        replacement_runner.legacy,
        "_make_subject_plan",
        lambda prepared, animal: calls.append("subject-plan") or subject_plan,
    )
    monkeypatch.setattr(
        replacement_runner.assessment,
        "render_fixed_case_multiresolution_v2",
        lambda prepared, frozen_plan, deformation: calls.append("eight-renders")
        or rendered,
    )
    monkeypatch.setattr(
        replacement_runner.assessment,
        "assemble_fixed_case_multiresolution_assessment_v2",
        lambda frozen_plan, raw: calls.append("assessment") or result,
    )
    monkeypatch.setattr(
        replacement_runner.bundle,
        "write_staged_bundle_v2",
        lambda destination, **kwargs: calls.append("write-staging") or staging,
    )
    monkeypatch.setattr(
        replacement_runner.bundle,
        "verify_frozen_bundle_v2",
        lambda path, prepared, inputs: calls.append("independent-verify")
        or {
            "bundle_receipt_sha256": "1" * 64,
            "plan_receipt_sha256": "2" * 64,
            "assessment_receipt_sha256": "3" * 64,
            "raw_render_count": 8,
            "qualification_eligible": False,
            "acceptance_thresholds": None,
        },
    )
    monkeypatch.setattr(
        replacement_runner.bundle,
        "publish_staged_bundle_v2",
        lambda partial, destination: calls.append("atomic-publish"),
    )
    monkeypatch.setattr(replacement_runner, "print", lambda *a, **k: None, raising=False)
    replacement_runner.main()
    assert calls == [
        "inputs",
        "legacy-report",
        "plan",
        "subject-plan",
        "eight-renders",
        "assessment",
        "write-staging",
        "independent-verify",
        "atomic-publish",
    ]


def test_atomic_bundle_roundtrip_recomputes_assessment_from_raw_arrays(
    prepared, stubbed_panel, monkeypatch, tmp_path
):
    report = _failed_legacy_report(prepared, monkeypatch)
    plan = multiresolution.make_fixed_case_multiresolution_plan_v2(report, prepared)
    rendered = _fixture_multiresolution_renders(plan)
    result = multiresolution.assemble_fixed_case_multiresolution_assessment_v2(
        plan, rendered
    )
    subject_plan = {
        "subject_deformation_plan_id": "fixture-plan",
        "state": {"raw_coefficients": np.arange(6, dtype=np.float32).reshape(2, 3)},
    }
    allen_inputs = {
        "decoder": "fixture decoder",
        "template": {"sha256": "a" * 64},
        "annotation": {"sha256": "b" * 64},
    }
    output = tmp_path / "frozen-bundle"
    staging = multiresolution_bundle.write_staged_bundle_v2(
        output,
        allen_inputs=allen_inputs,
        failed_report=report,
        plan=plan,
        subject_plan=subject_plan,
        rendered=rendered,
        result=result,
    )
    assert staging.exists() and not output.exists()
    monkeypatch.setattr(
        multiresolution_bundle.deformation,
        "verify_subject_deformation_plan_v2",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        multiresolution_bundle.assessment,
        "_subject_plan_matches",
        lambda frozen_plan, deformation_plan: True,
    )
    monkeypatch.setattr(
        multiresolution_bundle.slab,
        "verify_v2_generic_global_reference_slab_render",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        multiresolution_bundle.subject_slab,
        "verify_subject_slab_render_v2",
        lambda *args, **kwargs: None,
    )
    verified = multiresolution_bundle.verify_frozen_bundle_v2(
        staging, prepared, allen_inputs
    )
    assert verified["raw_render_count"] == 8
    assert verified["qualification_eligible"] is False
    assert verified["acceptance_thresholds"] is None
    multiresolution_bundle.publish_staged_bundle_v2(staging, output)
    assert output.exists() and not staging.exists()
    assert multiresolution_bundle.verify_frozen_bundle_v2(
        output, prepared, allen_inputs
    ) == verified
    with pytest.raises(FileExistsError):
        multiresolution_bundle.write_staged_bundle_v2(
            output,
            allen_inputs=allen_inputs,
            failed_report=report,
            plan=plan,
            subject_plan=subject_plan,
            rendered=rendered,
            result=result,
        )
    (output / "unexpected.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory or provenance"):
        multiresolution_bundle.verify_frozen_bundle_v2(
            output, prepared, allen_inputs
        )


def test_independent_verifier_loads_pinned_inputs_and_frozen_output(
    monkeypatch, tmp_path
):
    output = tmp_path / "frozen"
    output.mkdir()
    context = object()
    inputs = object()
    calls = []
    monkeypatch.setenv(independent_verifier.OUTPUT_ENVIRONMENT, str(output))
    monkeypatch.setattr(
        independent_verifier.bundle,
        "load_pinned_allen_context",
        lambda path: calls.append("inputs") or (context, inputs),
    )
    monkeypatch.setattr(
        independent_verifier.bundle,
        "verify_frozen_bundle_v2",
        lambda path, prepared, allen: calls.append((path, prepared, allen))
        or {
            "bundle_receipt_sha256": "1" * 64,
            "plan_receipt_sha256": "2" * 64,
            "assessment_receipt_sha256": "3" * 64,
            "raw_render_count": 8,
            "qualification_eligible": False,
            "acceptance_thresholds": None,
        },
    )
    monkeypatch.setattr(independent_verifier, "print", lambda *a, **k: None, raising=False)
    independent_verifier.main()
    assert calls == ["inputs", (output.resolve(), context, inputs)]
