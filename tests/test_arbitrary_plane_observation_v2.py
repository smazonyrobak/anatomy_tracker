import copy
import inspect

import numpy as np
import pytest

import training.arbitrary_plane_acquisition_v2 as acquisition
import training.arbitrary_plane_observation_v2 as observation


ROOT_SEED = "0x4f42534552564532"
BASE_KWARGS = {
    "subject_plan": None,
    "root_seed": ROOT_SEED,
    "split": "train",
    "split_index": 1,
    "animal_index": 7,
    "animal_id": "animal-007",
    "section_index": 3,
    "observation_index": 2,
}


def _processed_arrays(render):
    raster = render["raster"]
    state = render["state"]
    return {
        "scalar": raster["scalar"],
        "labels": raster["slab_modal_annotation"],
        "tissue": raster["slab_observable_support_mask"],
        "dense_weight": raster["slab_supervision_weight_or_abstention"][
            "dense_correspondence_weight"
        ],
        "abstention": raster["slab_supervision_weight_or_abstention"][
            "abstention_mask"
        ],
        "mapped": render["mapped_ccf_physical_coordinates_ap_dv_ml_um"],
        "bilinear_valid": state["bilinear_domain_valid_mask"],
        "nearest_valid": state["nearest_domain_valid_mask"],
        "dense_valid": state["dense_coordinate_valid_mask"],
    }


def _fake_processing_receipt(render):
    payload = {
        "section_processing_render_id": render["section_processing_render_id"],
        "source_input_reference": render["source_input_reference"],
        "plan_reference": render["plan_reference"],
        "pose_anatomy_policy": render["pose_anatomy_policy"],
        "live_array_receipts": {
            name: acquisition._array_receipt(value)
            for name, value in _processed_arrays(render).items()
        },
    }
    return {**payload, "receipt_sha256": acquisition._payload_sha256(payload)}


def _refresh_processed_receipts(render):
    render["fake_array_receipts"] = {
        name: acquisition._array_receipt(value)
        for name, value in _processed_arrays(render).items()
    }
    render["receipt_sha256"] = acquisition._payload_sha256(
        _fake_processing_receipt(render)
    )


def _fake_inputs():
    height, width = 84, 116
    y, x = np.mgrid[:height, :width]
    tissue = ((x - 57.0) / 47.0) ** 2 + ((y - 41.0) / 31.0) ** 2 <= 1.0
    scalar = (
        100.0
        + 1.7 * y
        + 0.9 * x
        + 14.0 * np.sin(x / 9.0)
        - 8.0 * np.cos(y / 7.0)
    ).astype(np.float32)
    labels = np.zeros((height, width), dtype=np.int64)
    labels[tissue & (x < 45)] = 11
    labels[tissue & (x >= 45) & (x < 72)] = 23
    labels[tissue & (x >= 72)] = 37
    bilinear_valid = np.ones((height, width), dtype=bool)
    nearest_valid = np.ones((height, width), dtype=bool)
    bilinear_valid[[0, -1], :] = False
    bilinear_valid[:, [0, -1]] = False
    nearest_valid[[0, -1], :] = False
    nearest_valid[:, [0, -1]] = False
    dense_valid = tissue.copy()
    dense_valid[tissue & ((y < 15) | (x > 96))] = False
    dense_weight = np.zeros((height, width), dtype=np.float32)
    dense_weight[dense_valid & ((x + y) % 3 == 0)] = np.float32(0.25)
    dense_weight[dense_valid & ((x + y) % 3 != 0)] = np.float32(1.0)
    abstention = ~dense_valid
    mapped = np.stack(
        (
            180.0 + 25.0 * y,
            -75.0 + 31.0 * x,
            420.0 + 9.0 * x + 7.0 * y,
        ),
        axis=-1,
    ).astype(np.float64)
    mapped[~dense_valid] = np.nan
    subject_slab_render = {
        "subject_slab_render_id": "subject-slab-id",
        "receipt_sha256": "subject-slab-receipt",
    }
    processing_plan = {
        "section_processing_plan_id": "processing-plan-id",
        "section_processing_realization_id": "processing-realization-id",
        "synthetic_section_processing_id": "synthetic-processing-id",
        "resolved_config": {
            "image_shape_h_w": [height, width],
            "pixel_pitch_y_x_um": [17.0, 29.0],
        },
        "provenance": {
            "split": "train",
            "animal_index": 7,
            "animal_id": "animal-007",
            "section_index": 3,
            "section_id": "section-003",
        },
    }
    pose_reference = {
        "source_subject_coordinate_map_id": "coordinate-map-id",
        "context_reference": {"v2_context_sha256": "c" * 64},
        "precursor_reference": {"slab_render_id": "precursor-slab-id"},
        "centre_plane_fit": {
            "model": "affine-plus-retained-nonlinear-residual",
            "residual_receipt_sha256": "r" * 64,
        },
    }
    processed_render = {
        "section_processing_render_id": "processing-render-id",
        "source_input_reference": {
            "source_stage_receipt": {
                "subject_slab_render_id": subject_slab_render[
                    "subject_slab_render_id"
                ]
            },
            "source_input_receipt_sha256": "source-input-receipt",
        },
        "plan_reference": {
            "section_processing_plan_id": processing_plan[
                "section_processing_plan_id"
            ],
            "section_processing_realization_id": processing_plan[
                "section_processing_realization_id"
            ],
            "synthetic_section_processing_id": processing_plan[
                "synthetic_section_processing_id"
            ],
            "section_processing_provenance": copy.deepcopy(
                processing_plan["provenance"]
            ),
        },
        "mapping_contract": {
            "forward": "subject section Y-X to processed Y-X is exp(+v)",
            "render_pullback": "processed pixel centre to subject Y-X is exp(-v)",
        },
        "pose_anatomy_policy": {"pose_anatomy_reference": pose_reference},
        "raster": {
            "scalar": scalar,
            "slab_modal_annotation": labels,
            "slab_observable_support_mask": tissue,
            "slab_supervision_weight_or_abstention": {
                "dense_correspondence_weight": dense_weight,
                "abstention_mask": abstention,
            },
        },
        "state": {
            "bilinear_domain_valid_mask": bilinear_valid,
            "nearest_domain_valid_mask": nearest_valid,
            "dense_coordinate_valid_mask": dense_valid,
        },
        "mapped_ccf_physical_coordinates_ap_dv_ml_um": mapped,
    }
    _refresh_processed_receipts(processed_render)
    return {
        "processed_render": processed_render,
        "subject_slab_render": subject_slab_render,
        "processing_plan": processing_plan,
        "prepared_context": {"v2_context_sha256": "c" * 64},
        "precursor": {"slab_render_id": "precursor-slab-id"},
    }


def _coherent_lineage_inputs(inputs, **changes):
    changed = copy.deepcopy(inputs)
    provenance = changed["processing_plan"]["provenance"]
    provenance.update(changes)
    token = acquisition._payload_sha256(provenance)
    plan = changed["processing_plan"]
    plan["section_processing_plan_id"] = f"processing-plan-{token}"
    plan["section_processing_realization_id"] = f"processing-realization-{token}"
    plan["synthetic_section_processing_id"] = f"synthetic-processing-{token}"
    slab_id = f"subject-slab-{token}"
    changed["subject_slab_render"]["subject_slab_render_id"] = slab_id
    render = changed["processed_render"]
    render["source_input_reference"]["source_stage_receipt"][
        "subject_slab_render_id"
    ] = slab_id
    render["source_input_reference"]["source_input_receipt_sha256"] = token
    render["plan_reference"].update(
        {
            "section_processing_plan_id": plan["section_processing_plan_id"],
            "section_processing_realization_id": plan[
                "section_processing_realization_id"
            ],
            "synthetic_section_processing_id": plan[
                "synthetic_section_processing_id"
            ],
            "section_processing_provenance": copy.deepcopy(provenance),
        }
    )
    render["pose_anatomy_policy"]["pose_anatomy_reference"][
        "source_subject_coordinate_map_id"
    ] = f"coordinate-map-{token}"
    render["section_processing_render_id"] = f"processing-render-{token}"
    _refresh_processed_receipts(render)
    return changed


def _with_sparse_tissue(inputs, coordinates):
    changed = copy.deepcopy(inputs)
    render = changed["processed_render"]
    shape = render["raster"]["scalar"].shape
    tissue = np.zeros(shape, dtype=bool)
    for y, x in coordinates:
        tissue[y, x] = True
    labels = np.zeros(shape, dtype=np.int64)
    labels[tissue] = 11
    dense_weight = np.zeros(shape, dtype=np.float32)
    for ordinal, (y, x) in enumerate(coordinates):
        dense_weight[y, x] = np.float32(0.25 if ordinal % 2 == 0 else 1.0)
    mapped = render["mapped_ccf_physical_coordinates_ap_dv_ml_um"].copy()
    mapped[~tissue] = np.nan
    render["raster"]["slab_modal_annotation"] = labels
    render["raster"]["slab_observable_support_mask"] = tissue
    render["raster"]["slab_supervision_weight_or_abstention"] = {
        "dense_correspondence_weight": dense_weight,
        "abstention_mask": ~tissue,
    }
    render["state"]["dense_coordinate_valid_mask"] = tissue.copy()
    render["mapped_ccf_physical_coordinates_ap_dv_ml_um"] = mapped
    render["section_processing_render_id"] = "processing-render-sparse-" + str(
        len(coordinates)
    )
    _refresh_processed_receipts(render)
    return changed


def _stratum_indices(limit=256):
    prior = observation._engineering_priors()["damage"]
    strata = tuple(prior["stratum_probabilities"])
    probabilities = np.asarray(
        [prior["stratum_probabilities"][name] for name in strata]
    )
    selected = {}
    for observation_index in range(limit):
        seed = observation.derive_observation_seed_v2(
            ROOT_SEED,
            "train",
            1,
            7,
            3,
            observation_index,
            "damage",
            "stratum",
        )
        stratum = str(
            np.random.Generator(np.random.PCG64DXSM(seed)).choice(
                np.asarray(strata), p=probabilities
            )
        )
        selected.setdefault(stratum, observation_index)
        if len(selected) == len(strata):
            break
    assert set(selected) == set(strata)
    return selected


def _verify_fake_processing_chain(
    render,
    subject_slab_render,
    plan,
    prepared_context,
    precursor,
    *,
    subject_plan,
):
    live_receipts = {
        name: acquisition._array_receipt(value)
        for name, value in _processed_arrays(render).items()
    }
    if (
        render["fake_array_receipts"] != live_receipts
        or render["receipt_sha256"]
        != acquisition._payload_sha256(_fake_processing_receipt(render))
        or render["source_input_reference"]["source_stage_receipt"][
            "subject_slab_render_id"
        ]
        != subject_slab_render["subject_slab_render_id"]
        or render["plan_reference"]["section_processing_plan_id"]
        != plan["section_processing_plan_id"]
        or render["plan_reference"]["section_processing_realization_id"]
        != plan["section_processing_realization_id"]
        or render["plan_reference"]["synthetic_section_processing_id"]
        != plan["synthetic_section_processing_id"]
        or render["plan_reference"]["section_processing_provenance"]
        != plan["provenance"]
        or tuple(render["raster"]["scalar"].shape)
        != tuple(plan["resolved_config"]["image_shape_h_w"])
        or render["pose_anatomy_policy"]["pose_anatomy_reference"][
            "context_reference"
        ]["v2_context_sha256"]
        != prepared_context["v2_context_sha256"]
        or render["pose_anatomy_policy"]["pose_anatomy_reference"][
            "precursor_reference"
        ]["slab_render_id"]
        != precursor["slab_render_id"]
    ):
        raise ValueError("fake authenticated processing chain does not match")


@pytest.fixture(scope="module", autouse=True)
def patched_processing_contract():
    patch = pytest.MonkeyPatch()
    patch.setattr(
        observation,
        "verify_section_processing_render_v2",
        _verify_fake_processing_chain,
    )
    patch.setattr(
        observation,
        "section_processing_render_receipt_v2",
        _fake_processing_receipt,
    )
    yield
    patch.undo()


@pytest.fixture(scope="module")
def inputs():
    return _fake_inputs()


def _make(inputs, modality="brightfield-nissl-like", **overrides):
    parameters = {**BASE_KWARGS, **overrides, "modality": modality}
    return observation.make_arbitrary_plane_observation_v2(
        inputs["processed_render"],
        inputs["subject_slab_render"],
        inputs["processing_plan"],
        inputs["prepared_context"],
        inputs["precursor"],
        **parameters,
    )


def _verify(artifact, inputs, modality="brightfield-nissl-like", **overrides):
    parameters = {**BASE_KWARGS, **overrides, "modality": modality}
    observation.verify_arbitrary_plane_observation_v2(
        artifact,
        inputs["processed_render"],
        inputs["subject_slab_render"],
        inputs["processing_plan"],
        inputs["prepared_context"],
        inputs["precursor"],
        **parameters,
    )


@pytest.fixture(scope="module")
def brightfield(inputs):
    return _make(inputs)


@pytest.fixture(scope="module")
def fluorescence(inputs):
    return _make(inputs, "fluorescence")


def test_rng_is_length_prefixed_numeric_and_never_accepts_animal_id():
    assert "animal_id" not in inspect.signature(
        observation.derive_observation_seed_v2
    ).parameters
    seed = observation.derive_observation_seed_v2(
        ROOT_SEED, "train", 1, 7, 3, 2, "appearance", "shot-noise", 0
    )
    assert seed == observation.derive_observation_seed_v2(
        ROOT_SEED, "train", 1, 7, 3, 2, "appearance", "shot-noise", 0
    )
    assert seed != observation.derive_observation_seed_v2(
        ROOT_SEED, "train", 1, 8, 3, 2, "appearance", "shot-noise", 0
    )
    assert seed != observation.derive_observation_seed_v2(
        ROOT_SEED, "train", 1, 7, 3, 2, "appearance", "read-noise", 0
    )
    assert seed != observation.derive_observation_seed_v2(
        ROOT_SEED, "development", 1, 7, 3, 2, "appearance", "shot-noise", 0
    )


def test_rng_coordinates_roots_and_names_require_exact_types():
    expected = observation.derive_observation_seed_v2(
        ROOT_SEED, "train", 1, 7, 3, 2, "appearance", "shot-noise", 0
    )
    assert expected == observation.derive_observation_seed_v2(
        np.uint64(int(ROOT_SEED, 16)),
        "train",
        np.int64(1),
        np.int32(7),
        np.int64(3),
        np.int32(2),
        "appearance",
        "shot-noise",
        np.int64(0),
    )
    baseline = [ROOT_SEED, "train", 1, 7, 3, 2, "appearance", "shot-noise", 0]
    invalid = (
        (0, 1.0),
        (0, True),
        (1, 7),
        (1, ""),
        (2, 1.0),
        (2, True),
        (3, 7.0),
        (3, np.bool_(True)),
        (4, np.float64(3)),
        (5, False),
        (6, object()),
        (7, 9),
        (8, 0.0),
        (2, 1 << 64),
    )
    for position, value in invalid:
        arguments = baseline.copy()
        arguments[position] = value
        with pytest.raises(ValueError):
            observation.derive_observation_seed_v2(*arguments)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("root_seed", 17.0),
        ("root_seed", True),
        ("split", 7),
        ("split", ""),
        ("split_index", 1.0),
        ("split_index", False),
        ("animal_index", 7.0),
        ("animal_index", True),
        ("section_index", np.float64(3)),
        ("section_index", np.bool_(True)),
        ("observation_index", 2.0),
        ("observation_index", False),
    ),
)
def test_public_make_and_verify_reject_noninteger_provenance(
    inputs, brightfield, name, value
):
    with pytest.raises(ValueError):
        _make(inputs, **{name: value})
    with pytest.raises(ValueError):
        _verify(brightfield, inputs, **{name: value})


def test_public_make_and_verify_accept_numpy_integer_provenance(inputs):
    numeric = {
        "root_seed": np.uint64(int(ROOT_SEED, 16)),
        "split_index": np.int64(1),
        "animal_index": np.int32(7),
        "section_index": np.int64(3),
        "observation_index": np.int32(2),
    }
    artifact = _make(inputs, **numeric)
    assert artifact["provenance"]["split_index"] == 1
    assert type(artifact["provenance"]["split_index"]) is int
    _verify(artifact, inputs, **numeric)


def test_modalities_have_different_forward_noise_and_background_models(
    inputs, brightfield, fluorescence
):
    assert brightfield["modality_model"]["forward_assumption"].startswith(
        "Beer-Lambert"
    )
    assert fluorescence["modality_model"]["forward_assumption"].startswith(
        "additive positive"
    )
    assert "bright glass" in brightfield["modality_model"]["background_assumption"]
    assert "dark field" in fluorescence["modality_model"]["background_assumption"]
    assert not np.array_equal(
        brightfield["arrays"]["raw_acquired_image_float32"],
        fluorescence["arrays"]["raw_acquired_image_float32"],
    )
    assert (
        brightfield["arrays"]["acquired_background_float32"].mean()
        > fluorescence["arrays"]["acquired_background_float32"].mean()
    )
    assert brightfield["disclosure"][
        "atlas_labels_used_as_synthetic_ground_truth_only"
    ]
    assert not brightfield["disclosure"]["atlas_labels_exposed_to_model_inputs"]
    assert all(not values for values in brightfield["asset_dependencies"].values())
    assert "synthetic_realization_id" not in repr(brightfield)
    _verify(brightfield, inputs)
    _verify(fluorescence, inputs, "fluorescence")


def test_crop_preserves_processed_coordinates_validity_and_pose_reference(
    inputs, brightfield
):
    arrays = brightfield["arrays"]
    crop = brightfield["crop_window"]
    render = inputs["processed_render"]
    top, left = crop["top_left_y_x"]
    height, width = crop["output_shape_h_w"]
    region = np.s_[top : top + height, left : left + width]
    assert np.array_equal(
        arrays["source_scalar_crop_float32"], render["raster"]["scalar"][region]
    )
    assert np.array_equal(
        arrays["processed_mapped_ccf_physical_coordinates_crop_float64"],
        render["mapped_ccf_physical_coordinates_ap_dv_ml_um"][region],
        equal_nan=True,
    )
    for artifact_key, state_key in (
        ("processed_bilinear_domain_valid_mask", "bilinear_domain_valid_mask"),
        ("processed_nearest_domain_valid_mask", "nearest_domain_valid_mask"),
        ("processed_dense_coordinate_valid_mask", "dense_coordinate_valid_mask"),
    ):
        assert np.array_equal(arrays[artifact_key], render["state"][state_key][region])
    assert np.array_equal(
        arrays["source_dense_correspondence_weight_float32"],
        render["raster"]["slab_supervision_weight_or_abstention"][
            "dense_correspondence_weight"
        ][region],
    )
    assert np.array_equal(
        arrays["source_dense_correspondence_abstention_mask"],
        render["raster"]["slab_supervision_weight_or_abstention"]["abstention_mask"][
            region
        ],
    )
    assert crop["operator"] == "integer parent-raster slice; no interpolation or resize"
    assert crop["processed_pixel_pitch_y_x_um"] == [17.0, 29.0]
    assert crop["processed_closed_face_window_y_x_um"] == [
        [top * 17.0, left * 29.0],
        [(top + height) * 17.0, (left + width) * 29.0],
    ]
    assert crop["processed_mapped_ccf_coordinate_crop_receipt"] == acquisition._array_receipt(
        arrays["processed_mapped_ccf_physical_coordinates_crop_float64"]
    )
    pose = render["pose_anatomy_policy"]["pose_anatomy_reference"]
    assert crop["pose_anatomy_reference_sha256"] == acquisition._payload_sha256(
        acquisition._json_value(pose)
    )
    assert crop["processed_mapping_contract_sha256"] == acquisition._payload_sha256(
        acquisition._json_value(render["mapping_contract"])
    )
    assert "observation-only window" in crop["plane_target_policy"]
    assert "must not be reinterpreted as a new plane" in crop[
        "downstream_coordinate_contract"
    ]
    assert "nonlinear residual" in crop["nonlinear_coordinate_policy"]
    assert brightfield["upstream_reference"]["section_processing_render_id"] == render[
        "section_processing_render_id"
    ]
    assert brightfield["upstream_reference"]["source_subject_coordinate_map_id"] == (
        pose["source_subject_coordinate_map_id"]
    )


def test_damage_and_correspondence_algebra_are_exact(inputs, brightfield):
    arrays = brightfield["arrays"]
    tissue = arrays["source_tissue_ground_truth_mask"]
    domain = arrays["source_correspondence_domain_mask"]
    physical = arrays["physical_loss_mask"]
    occlusion = arrays["occlusion_mask"]
    artifact = arrays["appearance_artifact_mask"]
    assert not np.any(physical & occlusion)
    assert not np.any(physical & artifact)
    assert not np.any(occlusion & artifact)
    assert not np.any((physical | occlusion | artifact) & ~tissue)
    damage = physical | occlusion | artifact
    assert np.array_equal(arrays["damage_union_mask"], damage)
    assert np.array_equal(arrays["observation_invalid_mask"], tissue & damage)
    assert np.array_equal(
        arrays["valid_correspondence_mask"], domain & tissue & ~damage
    )
    assert np.array_equal(
        arrays["valid_correspondence_weight_float32"],
        np.where(
            arrays["valid_correspondence_mask"],
            arrays["source_dense_correspondence_weight_float32"],
            np.float32(0),
        ).astype(np.float32),
    )
    assert not arrays["valid_correspondence_weight_float32"][
        damage
        | arrays["source_dense_correspondence_abstention_mask"]
        | ~tissue
    ].view(np.uint32).any()
    assert not np.any(domain & ~arrays["processed_dense_coordinate_valid_mask"])
    assert not np.any(
        arrays["valid_correspondence_mask"]
        & (damage | arrays["outside_correspondence_domain_mask"])
    )
    parameters = brightfield["parameters"]["damage"]
    assert parameters["realized_event_count"] == len(parameters["events"])
    assert parameters["damaged_tissue_fraction"] == float(
        damage.sum() / tissue.sum()
    )


def test_clean_mild_moderate_severe_mixture_is_bounded_and_clean_is_exact(inputs):
    prior = observation._engineering_priors()["damage"]
    strata = tuple(prior["stratum_probabilities"])
    selected = _stratum_indices()
    maximum_fractions = []
    maximum_event_counts = []
    for stratum in strata:
        artifact = _make(inputs, observation_index=selected[stratum])
        _verify(artifact, inputs, observation_index=selected[stratum])
        parameters = artifact["parameters"]["damage"]
        specification = prior["strata"][stratum]
        requested_count = parameters["requested_event_count"]
        realized_count = parameters["realized_event_count"]
        damage = artifact["arrays"]["damage_union_mask"]
        tissue = artifact["arrays"]["source_tissue_ground_truth_mask"]
        assert parameters["stratum"] == stratum
        assert specification["event_count_range_inclusive"][0] <= requested_count <= (
            specification["event_count_range_inclusive"][1]
        )
        assert realized_count == min(
            requested_count, parameters["damage_budget_pixels"]
        )
        assert parameters["support_limited"] is (
            realized_count != requested_count
        )
        assert parameters["no_redraw"] is True
        assert parameters["no_target_overlap_conditioning"] is True
        assert parameters["damaged_tissue_fraction"] <= specification[
            "maximum_damaged_tissue_fraction"
        ]
        assert parameters["damaged_tissue_fraction"] == float(
            damage.sum() / tissue.sum()
        )
        if stratum == "clean":
            assert requested_count == realized_count == 0 and not damage.any()
            assert np.array_equal(
                artifact["arrays"]["raw_acquired_image_float32"],
                artifact["arrays"]["pre_damage_acquired_image_float32"],
            )
        maximum_fractions.append(specification["maximum_damaged_tissue_fraction"])
        maximum_event_counts.append(specification["event_count_range_inclusive"][1])
    assert maximum_fractions == sorted(maximum_fractions)
    assert maximum_event_counts == sorted(maximum_event_counts)
    assert any(
        geometry != "ellipse"
        for values in prior["geometry_families"].values()
        for geometry in values
    )


def test_one_and_two_pixel_intersections_preserve_strata_with_support_cap(inputs):
    selected = _stratum_indices()
    prior = observation._engineering_priors()["damage"]
    sparse_inputs = (
        _with_sparse_tissue(inputs, [(42, 58)]),
        _with_sparse_tissue(inputs, [(42, 58), (42, 59)]),
    )
    for sparse in sparse_inputs:
        for stratum, observation_index in selected.items():
            artifact = _make(sparse, observation_index=observation_index)
            _verify(artifact, sparse, observation_index=observation_index)
            tissue_count = int(
                artifact["arrays"]["source_tissue_ground_truth_mask"].sum()
            )
            parameters = artifact["parameters"]["damage"]
            requested = parameters["requested_event_count"]
            budget = int(
                np.floor(
                    prior["strata"][stratum]["maximum_damaged_tissue_fraction"]
                    * tissue_count
                )
            )
            assert parameters["stratum"] == stratum
            assert tissue_count == len(
                np.argwhere(
                    sparse["processed_render"]["raster"][
                        "slab_observable_support_mask"
                    ]
                )
            )
            assert parameters["damage_budget_pixels"] == budget
            assert parameters["realized_event_count"] == min(requested, budget)
            assert parameters["support_limited"] is (requested > budget)
            assert parameters["no_redraw"] is True
            assert parameters["no_target_overlap_conditioning"] is True
            assert "authenticated cropped tissue support only" in parameters[
                "event_realization_policy"
            ]


def test_fractional_dense_weight_is_exact_and_all_damage_categories_zero_it(inputs):
    selected = {}
    for observation_index in range(96):
        artifact = _make(inputs, observation_index=observation_index)
        arrays = artifact["arrays"]
        for category, key in (
            ("physical-loss", "physical_loss_mask"),
            ("occlusion", "occlusion_mask"),
            ("appearance-artifact", "appearance_artifact_mask"),
        ):
            if arrays[key].any():
                selected.setdefault(category, (observation_index, artifact, key))
        if len(selected) == 3:
            break
    assert set(selected) == {"physical-loss", "occlusion", "appearance-artifact"}
    for observation_index, artifact, damage_key in selected.values():
        _verify(artifact, inputs, observation_index=observation_index)
        arrays = artifact["arrays"]
        source_weight = arrays["source_dense_correspondence_weight_float32"]
        valid = arrays["valid_correspondence_mask"]
        valid_weight = arrays["valid_correspondence_weight_float32"]
        assert {0.0, 0.25, 1.0}.issubset(set(np.unique(source_weight).tolist()))
        assert np.array_equal(
            valid_weight,
            np.where(valid, source_weight, np.float32(0)).astype(np.float32),
        )
        assert not valid_weight[arrays[damage_key]].view(np.uint32).any()
        tampered = copy.deepcopy(artifact)
        yy, xx = np.argwhere(tampered["arrays"][damage_key])[0]
        tampered["arrays"]["valid_correspondence_weight_float32"][
            yy, xx
        ] = np.float32(0.25)
        with pytest.raises(ValueError, match="receipt"):
            _verify(tampered, inputs, observation_index=observation_index)


def test_damage_geometry_families_and_budget_clipping_are_exercised():
    shape = (48, 64)
    y, x = np.mgrid[: shape[0], : shape[1]]
    tissue = ((x - 31.5) / 26.0) ** 2 + ((y - 23.5) / 19.0) ** 2 <= 1.0
    priors = observation._engineering_priors()
    for category, expected in priors["damage"]["geometry_families"].items():
        observed = set()
        for observation_index in range(64):
            provenance = {
                "root_seed_uint64": ROOT_SEED,
                "split": "train",
                "split_index": 1,
                "animal_index": 7,
                "animal_id": "animal-007",
                "section_index": 3,
                "observation_index": observation_index,
            }
            mask, parameters = observation._event_mask(
                tissue,
                np.zeros(shape, dtype=bool),
                category,
                0,
                [0.05, 0.10],
                int(tissue.sum()),
                provenance,
                {},
                priors,
            )
            observed.add(parameters["geometry"])
            assert mask.any()
            assert parameters["proposed_pixel_count"] == parameters[
                "retained_pixel_count"
            ]
            assert parameters["budget_clipped"] is False
            if observed == set(expected):
                break
        assert observed == set(expected)

    provenance["observation_index"] = 999
    mask, parameters = observation._event_mask(
        tissue,
        np.zeros(shape, dtype=bool),
        "occlusion",
        0,
        [0.10, 0.10],
        1,
        provenance,
        {},
        priors,
    )
    assert mask.sum() == parameters["retained_pixel_count"] == 1
    assert parameters["proposed_pixel_count"] > 1
    assert parameters["budget_clipped"] is True


def test_raw_and_all_brush_descendants_share_one_acquisition(inputs, brightfield):
    arrays = brightfield["arrays"]
    descendants = brightfield["descendants"]
    assert set(descendants) == set(observation.DESCENDANT_MODES)
    assert all(
        item["acquired_observation_id"] == brightfield["acquired_observation_id"]
        for item in descendants.values()
    )
    assert {
        mode for mode, item in descendants.items() if item["trainable"]
    } == {
        "smart-brush-accurate",
        "smart-brush-imperfect",
        "smart-brush-absent",
    }
    assert descendants["raw"]["parameters"] == {
        "role": "nontrainable acquired-input audit mirror",
        "equivalent_trainable_mode": "smart-brush-absent",
        "sampling_policy": "never count raw and absent as separate training inputs",
    }
    raw = arrays["raw_acquired_image_float32"]
    accurate = descendants["smart-brush-accurate"]["arrays"]
    imperfect = descendants["smart-brush-imperfect"]["arrays"]
    absent = descendants["smart-brush-absent"]["arrays"]
    assert np.array_equal(
        accurate["selected_input_mask"], arrays["observable_footprint_mask"]
    )
    assert not accurate["model_input_image_float32"][
        ~accurate["selected_input_mask"]
    ].view(np.uint32).any()
    assert not imperfect["model_input_image_float32"][
        ~imperfect["selected_input_mask"]
    ].view(np.uint32).any()
    assert not np.array_equal(
        imperfect["selected_input_mask"], accurate["selected_input_mask"]
    )
    assert np.array_equal(
        imperfect["brush_mask_error_mask"],
        imperfect["selected_input_mask"] ^ accurate["selected_input_mask"],
    )
    assert np.array_equal(absent["model_input_image_float32"], raw)
    assert np.array_equal(
        descendants["raw"]["arrays"]["model_input_image_float32"], raw
    )
    assert descendants["smart-brush-absent"]["parameters"][
        "raw_audit_mirror_mode"
    ] == "raw"
    assert brightfield["disclosure"]["brush_availability_model_input"] == (
        "explicit descendant scalar; the model loader broadcasts it as a constant channel"
    )
    assert raw[~arrays["source_tissue_ground_truth_mask"]].any()
    assert brightfield["parameters"]["mask_algebra"][
        "brush_mask_error_excluded_from_truth"
    ]
    assert all(not key.startswith("damage/") for key in brightfield["rng_sources"]["brush"])
    assert "observation augmentation stream only" in brightfield["provenance"][
        "rng_dynamic_coordinates"
    ]
    assert "independent of authenticated upstream generator coordinates" in brightfield[
        "provenance"
    ]["rng_dynamic_coordinates"]


def test_empty_imperfect_brush_is_explicitly_tagged(inputs):
    sparse = _with_sparse_tissue(inputs, [(42, 58)])
    for observation_index in range(64):
        artifact = _make(sparse, observation_index=observation_index)
        imperfect = artifact["descendants"]["smart-brush-imperfect"]
        if not imperfect["arrays"]["selected_input_mask"].any():
            assert imperfect["parameters"]["empty_selection"] is True
            assert (
                imperfect["parameters"]["selection_failure_tag"]
                == "empty-imperfect-brush-selection"
            )
            _verify(artifact, sparse, observation_index=observation_index)
            break
    else:
        pytest.fail("bounded deterministic search did not exercise empty imperfect brush")


def test_identity_is_rng_independent_but_must_match_coherent_upstream_lineage(
    inputs, brightfield
):
    with pytest.raises(ValueError, match="upstream lineage"):
        _make(inputs, animal_id="animal-renamed")
    renamed_inputs = _coherent_lineage_inputs(
        inputs, animal_id="animal-renamed"
    )
    renamed = _make(renamed_inputs, animal_id="animal-renamed")
    assert renamed["rng_sources"] == brightfield["rng_sources"]
    assert renamed["observation_plan_id"] != brightfield["observation_plan_id"]
    assert renamed["acquired_observation_id"] != brightfield["acquired_observation_id"]
    assert renamed["array_receipts"] == brightfield["array_receipts"]
    for mode in observation.DESCENDANT_MODES:
        assert renamed["descendants"][mode]["array_receipts"] == brightfield[
            "descendants"
        ][mode]["array_receipts"]
    with pytest.raises(ValueError, match="upstream lineage"):
        _make(inputs, animal_index=8)
    numeric_inputs = _coherent_lineage_inputs(inputs, animal_index=8)
    changed_numeric = _make(numeric_inputs, animal_index=8)
    assert not np.array_equal(
        changed_numeric["arrays"]["raw_acquired_image_float32"],
        brightfield["arrays"]["raw_acquired_image_float32"],
    )
    for overrides in ({"split": "development"}, {"section_index": 4}):
        with pytest.raises(ValueError, match="upstream lineage"):
            _make(inputs, **overrides)


def test_replay_nested_source_and_upstream_tamper_rejection(inputs, brightfield):
    changed = copy.deepcopy(brightfield)
    changed["arrays"]["raw_acquired_image_float32"][0, 0] += np.float32(0.1)
    with pytest.raises(ValueError, match="receipt"):
        _verify(changed, inputs)

    changed = copy.deepcopy(brightfield)
    source_weight = changed["arrays"][
        "source_dense_correspondence_weight_float32"
    ]
    yy, xx = np.argwhere(source_weight == np.float32(0.25))[0]
    source_weight[yy, xx] = np.float32(0.5)
    with pytest.raises(ValueError, match="receipt"):
        _verify(changed, inputs)

    changed = copy.deepcopy(brightfield)
    changed["arrays"]["source_dense_correspondence_abstention_mask"][0, 0] ^= True
    with pytest.raises(ValueError, match="receipt"):
        _verify(changed, inputs)

    changed = copy.deepcopy(brightfield)
    changed["arrays"]["valid_correspondence_weight_float32"][0, 0] = np.float32(
        0.25
    )
    with pytest.raises(ValueError, match="receipt"):
        _verify(changed, inputs)

    changed = copy.deepcopy(brightfield)
    changed["descendants"]["smart-brush-imperfect"]["extra"] = 1
    with pytest.raises(ValueError, match="extra"):
        _verify(changed, inputs)

    changed = copy.deepcopy(brightfield)
    changed["crop_window"]["extra"] = 1
    with pytest.raises(ValueError, match="extra"):
        _verify(changed, inputs)

    changed = copy.deepcopy(brightfield)
    changed["implementation_source_sha256"][
        "arbitrary_plane_observation_v2.py"
    ] = "0" * 64
    with pytest.raises(ValueError, match="source|plan"):
        _verify(changed, inputs)

    changed = copy.deepcopy(brightfield)
    changed["crop_window"]["top_left_y_x"][0] += 1
    with pytest.raises(ValueError, match="crop|receipt"):
        _verify(changed, inputs)

    changed = copy.deepcopy(brightfield)
    changed["synthetic_realization_id"] = "premature"
    with pytest.raises(ValueError, match="premature final"):
        _verify(changed, inputs)

    changed_inputs = copy.deepcopy(inputs)
    changed_inputs["processed_render"]["raster"]["scalar"][0, 0] += np.float32(1.0)
    with pytest.raises(ValueError, match="processing chain"):
        _verify(brightfield, changed_inputs)

    changed_inputs = copy.deepcopy(inputs)
    changed_inputs["processed_render"]["pose_anatomy_policy"][
        "pose_anatomy_reference"
    ]["centre_plane_fit"]["residual_receipt_sha256"] = "z" * 64
    _refresh_processed_receipts(changed_inputs["processed_render"])
    with pytest.raises(ValueError, match="plan|source|receipt"):
        _verify(brightfield, changed_inputs)

    changed_inputs = copy.deepcopy(inputs)
    changed_inputs["processing_plan"]["section_processing_plan_id"] = "wrong-plan"
    with pytest.raises(ValueError, match="processing chain"):
        _verify(brightfield, changed_inputs)

    with pytest.raises(ValueError, match="upstream lineage"):
        _verify(brightfield, inputs, animal_id="wrong-animal-label")
