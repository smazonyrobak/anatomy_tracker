import copy

import numpy as np
import pytest
import torch

import training.arbitrary_plane_staged_training as staged
import training.arbitrary_plane_inference_v3 as inference_v3
from training.arbitrary_plane_catalogue_v3 import make_arbitrary_plane_catalogue_v3
from training.arbitrary_plane_inference_v3 import (
    CACHE_NUMERICAL_ATOL,
    CACHE_NUMERICAL_RTOL,
    load_arbitrary_plane_inference_v3,
    load_arbitrary_plane_catalogue_feature_cache_v3,
    make_arbitrary_plane_catalogue_feature_cache_v3,
    make_arbitrary_plane_joint_checkpoint_v3,
    make_inference_contract_v3,
    run_arbitrary_plane_inference_v3,
    verify_arbitrary_plane_catalogue_feature_cache_v3,
    verify_arbitrary_plane_joint_checkpoint_v3,
)
from training.arbitrary_plane_joint_model import ArbitraryPlaneJointModel
from training.arbitrary_plane_uncertainty_v3 import fit_temperature_on_heldout_animals_v3


def _catalogue():
    return make_arbitrary_plane_catalogue_v3(
        np.ones((6, 6, 6), dtype=bool),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        normal_count=2,
        offset_count=1,
        roll_count=1,
        raster_shape_h_w=(8, 8),
        raster_physical_span_y_x_um=(4.0, 4.0),
    )


def _config():
    return {
        "atlas_channels": 2,
        "feature_channels": 4,
        "hidden_channels": 6,
        "correlation_radius": 1,
        "update_limits": (0.08, 0.08, 0.5, 0.08, 0.5, 0.5, 0.04, 0.04, 0.04),
        "plane_tangent_scales": (0.08, 0.08, 0.5),
        "max_velocity_fraction_yx": (0.05, 0.04),
        "deformation_integration_steps": 3,
        "deformation_support_floor": 1e-4,
        "deformation_maximum_velocity_gradient": 0.35,
        "proposal_count": None,
        "proposal_channels": 16,
        "proposal_mixture_components": 8,
        "proposal_offset_scale_um": 10000.0,
    }


def _provenance():
    return {
        "initialization": "fresh_random",
        "architecture_source": "training.arbitrary_plane_joint_model",
        "prior_trained_model_dependencies": [],
        "prior_model_feature_dependencies": [],
        "pseudolabel_dependencies": [],
        "dataset_provenance": ["synthetic-generator-v3"],
        "animal_specimen_experiment_id_contract": "IDs retained; future splits are strictly animal-level",
    }


def _atlas_semantics():
    return {
        "schema_version": "anatomy-tracker.atlas-semantics/v3",
        "atlas_name": "test atlas",
        "atlas_version": "test-v1",
        "processed_channel_names": ["test_intensity", "test_support"],
        "processed_channel_recipes": ["value / 100", "test support channel"],
        "source_assets": [
            {"asset_role": "test", "uri": "test://atlas", "sha256": "d" * 64}
        ],
        "source_format": "synthetic test tensor",
        "nrrd_index_order": "F",
        "array_axis_order": ["AP", "DV", "ML"],
        "positive_axis_directions": ["positive AP", "positive DV", "positive ML"],
        "voxel_center_convention": "integer array coordinates denote voxel centres",
        "normalization_parameters": {"divisor": 100.0},
    }


def _checkpoint(tmp_path):
    catalogue = _catalogue()
    state = staged.initialize_staged_training(
        _config(),
        {
            "seed": 91,
            "pose_warmup_steps": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "top_k": 1,
            "refinement_steps": 1,
            "joint_pose_only_steps": 0,
            "retrieval_shape_h_w": (8, 8),
            "catalogue_chunk_size": 1,
            "amp": False,
            "amp_initial_scale": 128.0,
            "gradient_clip_norm": 5.0,
        },
        catalogue_id=catalogue["catalogue_id"],
        catalogue_receipt_sha256=catalogue["receipt_sha256"],
        catalogue_cell_count=int(catalogue["counts"]["cell_count"]),
        generator_ids=("synthetic-generator-v3",),
        device="cpu",
    )
    identity = {
        "training_row_id": "training-row-91",
        "training_row_receipt_sha256": "e" * 64,
        "synthetic_realization_id": "synthetic-realization-91",
        "animal_id": "train-animal",
        "specimen_id": "train-specimen",
        "experiment_id": "train-experiment",
        "synthetic_animal_id": "synthetic-animal-91",
        "section_id": "section-91",
        "split": "development",
    }
    ledger_payload = {
        "step": 0,
        "catalogue_scope": "complete catalogue posterior/inference scope",
        "training_row_ids": [identity["training_row_id"]],
        "training_row_receipt_sha256": [
            identity["training_row_receipt_sha256"]
        ],
        "training_row_identity_sha256": [staged._hash_json(identity)],
        "training_candidate_bank_receipt_sha256": [],
    }
    state["global_step"] = 1
    state["row_identity_records"] = [identity]
    state["training_step_ledger"] = [
        {**ledger_payload, "entry_sha256": staged._hash_json(ledger_payload)}
    ]
    staged_path = tmp_path / "staged-training.pt"
    staged.save_staged_training_checkpoint(state, staged_path)
    training_receipt = staged.make_staged_training_export_receipt_v3(staged_path)
    model = state["model"]
    atlas = torch.arange(2 * 6 * 6 * 6, dtype=torch.float32).reshape(2, 6, 6, 6) / 100.0
    annotation = torch.zeros(6, 6, 6, dtype=torch.long)
    offsets = torch.tensor([-0.5, 0.0, 0.5])
    weights = torch.tensor([0.25, 0.5, 0.25])
    inference_contract = make_inference_contract_v3(
        atlas,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        offsets,
        weights,
        atlas_semantics=_atlas_semantics(),
        annotation_volume_ap_dv_ml=annotation,
    )
    checkpoint = make_arbitrary_plane_joint_checkpoint_v3(
        model,
        _config(),
        catalogue,
        _provenance(),
        training_receipt,
        inference_contract=inference_contract,
    )
    return catalogue, checkpoint, atlas, annotation


def test_checkpoint_round_trip_is_architecture_and_catalogue_bound(tmp_path):
    catalogue, checkpoint, _, _ = _checkpoint(tmp_path)
    path = tmp_path / "joint-v3.pt"
    torch.save(checkpoint, path)
    loaded = load_arbitrary_plane_inference_v3(path, catalogue)
    assert loaded["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert loaded["catalogue_id"] == catalogue["catalogue_id"]
    assert loaded["checkpoint_binding_id"] == checkpoint["checkpoint_binding_id"]
    assert loaded["model_state_sha256"] == checkpoint["model_state_sha256"]
    assert loaded["model"].training is False
    for name, value in checkpoint["state_dict"].items():
        assert torch.equal(loaded["model"].state_dict()[name], value)


def test_checkpoint_loader_rejects_non_i_drive_even_for_an_existing_file(tmp_path):
    catalogue, _, _, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="I: drive"):
        load_arbitrary_plane_inference_v3(r"C:\Windows\win.ini", catalogue)


def test_checkpoint_rejects_tampered_weights_dependencies_and_catalogue(tmp_path):
    catalogue, checkpoint, _, _ = _checkpoint(tmp_path)
    tampered = copy.deepcopy(checkpoint)
    first = next(iter(tampered["state_dict"]))
    tampered["state_dict"][first].view(-1)[0] += 1
    with pytest.raises(ValueError, match="verification"):
        verify_arbitrary_plane_joint_checkpoint_v3(tampered, catalogue)
    tampered = copy.deepcopy(checkpoint)
    tampered["provenance"]["prior_trained_model_dependencies"] = ["old-model"]
    with pytest.raises(ValueError, match="fresh standalone"):
        verify_arbitrary_plane_joint_checkpoint_v3(tampered, catalogue)
    changed_catalogue = copy.deepcopy(catalogue)
    changed_catalogue["arrays"]["cell_log_mass_float64"][0] += 1.0
    with pytest.raises(ValueError, match="catalogue arrays"):
        verify_arbitrary_plane_joint_checkpoint_v3(checkpoint, changed_catalogue)
    model = ArbitraryPlaneJointModel(**_config())
    model.load_state_dict(checkpoint["state_dict"])
    with pytest.raises(ValueError, match="staged-training export receipt"):
        make_arbitrary_plane_joint_checkpoint_v3(
            model,
            _config(),
            catalogue,
            _provenance(),
            {"training_animal_ids": ["train-animal"]},
            inference_contract=checkpoint["inference_contract"],
        )
    model.deformation_decoder.maximum_velocity_gradient = 0.2
    with pytest.raises(ValueError, match="hyperparameters"):
        make_arbitrary_plane_joint_checkpoint_v3(
            model,
            _config(),
            catalogue,
            _provenance(),
            checkpoint["training_receipt"],
            inference_contract=checkpoint["inference_contract"],
        )


def test_checkpoint_embedded_calibration_binds_stable_checkpoint_and_model_digests(tmp_path):
    catalogue, checkpoint, _, _ = _checkpoint(tmp_path)
    receipt = fit_temperature_on_heldout_animals_v3(
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        torch.tensor([0, 1]),
        ["cal-a", "cal-b"],
        ["cal-a", "cal-b"],
        ["final-z"],
        catalogue["catalogue_id"],
        training_animal_ids=["train-animal"],
        checkpoint_binding_id=checkpoint["checkpoint_binding_id"],
        model_state_sha256=checkpoint["model_state_sha256"],
    )
    assert receipt["fully_calibrated"] is False
    assert receipt["refinement_temperature"] == 1.0
    assert receipt["continuous_covariance_scale"] == 1.0
    model = ArbitraryPlaneJointModel(**_config())
    model.load_state_dict(checkpoint["state_dict"])
    calibrated = make_arbitrary_plane_joint_checkpoint_v3(
        model,
        _config(),
        catalogue,
        _provenance(),
        checkpoint["training_receipt"],
        inference_contract=checkpoint["inference_contract"],
        calibration_receipt=receipt,
    )
    assert calibrated["checkpoint_binding_id"] == checkpoint["checkpoint_binding_id"]
    assert calibrated["model_state_sha256"] == checkpoint["model_state_sha256"]
    assert calibrated["checkpoint_id"] != checkpoint["checkpoint_id"]
    assert verify_arbitrary_plane_joint_checkpoint_v3(calibrated, catalogue)
    tampered = copy.deepcopy(receipt)
    tampered["model_state_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="calibration receipt"):
        make_arbitrary_plane_joint_checkpoint_v3(
            model,
            _config(),
            catalogue,
            _provenance(),
            checkpoint["training_receipt"],
            inference_contract=checkpoint["inference_contract"],
            calibration_receipt=tampered,
        )


def test_inference_accepts_exact_three_channel_input_and_returns_probability(tmp_path):
    catalogue, checkpoint, atlas, annotation = _checkpoint(tmp_path)
    path = tmp_path / "joint-v3.pt"
    torch.save(checkpoint, path)
    loaded = load_arbitrary_plane_inference_v3(path, catalogue)
    torch.manual_seed(93)
    input_b3hw = torch.cat(
        (
            torch.rand(1, 1, 8, 8),
            torch.ones(1, 1, 8, 8),
            torch.ones(1, 1, 8, 8),
        ),
        dim=1,
    ).double()
    result = run_arbitrary_plane_inference_v3(
        loaded,
        input_b3hw,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        torch.tensor([-0.5, 0.0, 0.5]),
        torch.tensor([0.25, 0.5, 0.25]),
        animal_ids=["animal-93"],
        specimen_ids=["specimen-93"],
        experiment_ids=["experiment-93"],
        synthetic_animal_ids=["synthetic-animal-93"],
        section_ids=["section-93"],
        synthetic_realization_ids=["realization-93"],
        top_k=1,
        refinement_steps=2,
        pose_only_steps=1,
        retrieval_shape_h_w=(8, 8),
        catalogue_chunk_size=1,
        electrode_points_yx_px=torch.tensor([[2.0, 3.0], [5.0, 4.0]]),
        annotation_volume_ap_dv_ml=annotation,
        raw_prediction_output_path=tmp_path / "raw-joint-output.pt",
    )
    posterior = result["probabilistic_output"]
    assert posterior["probabilities_calibrated"] is False
    assert posterior["complete_retrieval_probability"].sum().item() == pytest.approx(1.0)
    assert posterior["posterior_scope"].startswith("hierarchical/truncated")
    assert result["point_estimate"]["posterior_mean_center_ap_dv_ml_um"].shape == (1, 3)
    assert result["deformation_pullback_yx_px"].shape == (1, 1, 2, 8, 8)
    assert result["trajectory_credible_spatial_volume"][
        "trajectory_sample_points_ap_dv_ml_um"
    ].shape[-2:] == (2, 3)
    assert result["animal_ids"] == ["animal-93"]
    assert result["specimen_ids"] == ["specimen-93"]
    assert result["experiment_ids"] == ["experiment-93"]
    assert result["synthetic_animal_ids"] == ["synthetic-animal-93"]
    assert result["section_ids"] == ["section-93"]
    assert result["synthetic_realization_ids"] == ["realization-93"]
    expected_lineage = {
        "animal_id": "animal-93",
        "specimen_id": "specimen-93",
        "experiment_id": "experiment-93",
        "synthetic_animal_id": "synthetic-animal-93",
        "section_id": "section-93",
        "synthetic_realization_id": "realization-93",
    }
    assert result["lineage"] == [expected_lineage]
    assert result["input_receipt"]["raw_input_receipt"]["dtype"] == "torch.float64"
    assert result["input_receipt"]["model_input_receipt"]["dtype"] == "torch.float32"
    assert result["atlas_receipt"] == checkpoint["inference_contract"]
    assert len(result["configuration_receipt"]["receipt_sha256"]) == 64
    assert len(result["raw_prediction_receipt"]["receipt_sha256"]) == 64
    assert result["raw_prediction_path"].startswith("I:")
    assert len(result["raw_prediction_file_sha256"]) == 64
    raw = torch.load(result["raw_prediction_path"], map_location="cpu", weights_only=True)
    assert raw["raw_prediction_receipt"] == result["raw_prediction_receipt"]
    assert raw["identifiers"]["lineage"] == [expected_lineage]
    assert raw["input_receipt"]["identifiers"] == raw["identifiers"]
    assert raw["raw_prediction"]["lineage"] == [expected_lineage]
    assert "retrieval_cell_log_probability" in raw["raw_prediction"]["pose"]
    assert len(result["inference_receipt_sha256"]) == 64


def test_inference_rejects_nonconstant_availability_channel(tmp_path):
    catalogue, checkpoint, atlas, _ = _checkpoint(tmp_path)
    path = tmp_path / "joint-v3.pt"
    torch.save(checkpoint, path)
    loaded = load_arbitrary_plane_inference_v3(path, catalogue)
    input_b3hw = torch.zeros(1, 3, 8, 8)
    input_b3hw[:, 2, 0, 0] = 1.0
    with pytest.raises(ValueError, match="constant binary"):
        run_arbitrary_plane_inference_v3(
            loaded,
            input_b3hw,
            atlas,
            catalogue,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            torch.tensor([-0.5, 0.0, 0.5]),
            torch.tensor([0.25, 0.5, 0.25]),
            animal_ids=["animal-1"],
            specimen_ids=["specimen-1"],
            experiment_ids=["experiment-1"],
            top_k=1,
            refinement_steps=2,
            pose_only_steps=1,
            retrieval_shape_h_w=(8, 8),
            catalogue_chunk_size=1,
        )


def test_inference_rejects_255_domain_and_mutated_executable_attributes(tmp_path):
    catalogue, checkpoint, atlas, _ = _checkpoint(tmp_path)
    path = tmp_path / "joint-runtime-contract.pt"
    torch.save(checkpoint, path)
    loaded = load_arbitrary_plane_inference_v3(path, catalogue)
    image = torch.cat(
        (torch.full((1, 1, 8, 8), 255.0), torch.ones(1, 2, 8, 8)), dim=1
    )
    with pytest.raises(ValueError, match=r"trained \[0,1\] domain"):
        run_arbitrary_plane_inference_v3(
            loaded,
            image,
            atlas,
            catalogue,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            torch.tensor([-0.5, 0.0, 0.5]),
            torch.tensor([0.25, 0.5, 0.25]),
            animal_ids=["animal-1"],
            specimen_ids=["specimen-1"],
            experiment_ids=["experiment-1"],
        )
    loaded["model"].deformation_decoder.maximum_velocity_gradient = 0.2
    with pytest.raises(ValueError, match="executable model contract"):
        run_arbitrary_plane_inference_v3(
            loaded,
            torch.ones(1, 3, 8, 8),
            atlas,
            catalogue,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            torch.tensor([-0.5, 0.0, 0.5]),
            torch.tensor([0.25, 0.5, 0.25]),
            animal_ids=["animal-1"],
            specimen_ids=["specimen-1"],
            experiment_ids=["experiment-1"],
        )


def test_inference_rejects_runtime_atlas_psf_or_identifier_provenance(tmp_path):
    catalogue, checkpoint, atlas, annotation = _checkpoint(tmp_path)
    path = tmp_path / "joint-v3.pt"
    torch.save(checkpoint, path)
    loaded = load_arbitrary_plane_inference_v3(path, catalogue)
    image = torch.cat(
        (torch.rand(1, 2, 8, 8), torch.ones(1, 1, 8, 8)), dim=1
    )
    with pytest.raises(ValueError, match="runtime atlas assets"):
        run_arbitrary_plane_inference_v3(
            loaded,
            image,
            atlas + 1.0,
            catalogue,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            torch.tensor([-0.5, 0.0, 0.5]),
            torch.tensor([0.25, 0.5, 0.25]),
            animal_ids=["animal-1"],
            specimen_ids=["specimen-1"],
            experiment_ids=["experiment-1"],
            annotation_volume_ap_dv_ml=annotation,
        )
    with pytest.raises(ValueError, match="runtime atlas assets"):
        run_arbitrary_plane_inference_v3(
            loaded,
            image,
            atlas,
            catalogue,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            torch.tensor([-1.0, 0.0, 1.0]),
            torch.tensor([0.25, 0.5, 0.25]),
            animal_ids=["animal-1"],
            specimen_ids=["specimen-1"],
            experiment_ids=["experiment-1"],
            annotation_volume_ap_dv_ml=annotation,
        )
    with pytest.raises(ValueError, match="animal_ids"):
        run_arbitrary_plane_inference_v3(
            loaded,
            image,
            atlas,
            catalogue,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            torch.tensor([-0.5, 0.0, 0.5]),
            torch.tensor([0.25, 0.5, 0.25]),
            animal_ids=[],
            specimen_ids=["specimen-1"],
            experiment_ids=["experiment-1"],
            annotation_volume_ap_dv_ml=annotation,
        )


def test_complete_catalogue_cache_is_bound_complete_and_numerically_equivalent(tmp_path):
    catalogue, checkpoint, atlas, annotation = _checkpoint(tmp_path)
    checkpoint_path = tmp_path / "joint-cache-v3.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = load_arbitrary_plane_inference_v3(checkpoint_path, catalogue)
    cache = make_arbitrary_plane_catalogue_feature_cache_v3(
        loaded,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        torch.tensor([-0.5, 0.0, 0.5]),
        torch.tensor([0.25, 0.5, 0.25]),
        tmp_path / "complete-catalogue-cache.pt",
        retrieval_shape_h_w=(8, 8),
        build_chunk_size=1,
        annotation_volume_ap_dv_ml=annotation,
    )
    reloaded = load_arbitrary_plane_catalogue_feature_cache_v3(
        cache["cache_path"], loaded, catalogue
    )
    receipt = reloaded["cache_receipt"]
    assert receipt["complete_coverage"]["all_cells_exactly_once"] is True
    assert reloaded["cell_id"].tolist() == list(
        range(catalogue["counts"]["cell_count"])
    )
    assert receipt["feature_origin"]["external_or_prior_model_dependencies"] == []
    assert receipt["feature_origin"]["approximate_candidate_pruning"] is False
    assert receipt["numerical_equivalence_contract"] == {
        "absolute_tolerance": CACHE_NUMERICAL_ATOL,
        "relative_tolerance": CACHE_NUMERICAL_RTOL,
        "scope": "complete normalized retrieval log probabilities/probabilities, stable top-K IDs/log probabilities, retained mass, and omitted tail mass",
        "same_dtype_no_compression": True,
    }

    torch.manual_seed(411)
    image = torch.cat(
        (torch.rand(1, 1, 8, 8), torch.ones(1, 2, 8, 8)), dim=1
    )
    arguments = (
        loaded,
        image,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        torch.tensor([-0.5, 0.0, 0.5]),
        torch.tensor([0.25, 0.5, 0.25]),
    )
    keywords = dict(
        animal_ids=["cache-animal"],
        specimen_ids=["cache-specimen"],
        experiment_ids=["cache-experiment"],
        top_k=1,
        refinement_steps=1,
        pose_only_steps=0,
        retrieval_shape_h_w=(8, 8),
        catalogue_chunk_size=1,
        annotation_volume_ap_dv_ml=annotation,
    )
    uncached = run_arbitrary_plane_inference_v3(
        *arguments,
        **keywords,
        raw_prediction_output_path=tmp_path / "uncached-raw.pt",
    )
    cached = run_arbitrary_plane_inference_v3(
        *arguments,
        **keywords,
        raw_prediction_output_path=tmp_path / "cached-raw.pt",
        catalogue_feature_cache=reloaded,
    )
    uncached_artifact = torch.load(
        uncached["raw_prediction_path"], map_location="cpu", weights_only=True
    )
    assert uncached_artifact["identifiers"]["synthetic_animal_ids"] == [None]
    assert uncached_artifact["identifiers"]["section_ids"] == [None]
    assert uncached_artifact["identifiers"]["synthetic_realization_ids"] == [None]
    assert uncached_artifact["raw_prediction"]["lineage"][0]["section_id"] is None
    uncached_raw = uncached_artifact["raw_prediction"]["pose"]
    cached_raw = torch.load(
        cached["raw_prediction_path"], map_location="cpu", weights_only=True
    )["raw_prediction"]["pose"]
    assert cached_raw["retrieval_execution"] == "cached_complete_catalogue_features_exact"
    for name in (
        "retrieval_cell_log_probability",
        "retrieval_cell_probability",
        "retrieval_topk_log_probability",
        "retrieval_topk_retained_probability",
        "retrieval_omitted_probability",
    ):
        assert torch.allclose(
            cached_raw[name],
            uncached_raw[name],
            atol=CACHE_NUMERICAL_ATOL,
            rtol=CACHE_NUMERICAL_RTOL,
        )
    assert torch.equal(
        cached_raw["retrieval_topk_cell_id"], uncached_raw["retrieval_topk_cell_id"]
    )
    assert torch.allclose(
        cached["deformation_pullback_yx_px"],
        uncached["deformation_pullback_yx_px"],
        atol=CACHE_NUMERICAL_ATOL,
        rtol=CACHE_NUMERICAL_RTOL,
    )
    assert cached["configuration_receipt"]["catalogue_feature_cache"]["cache_id"] == receipt["cache_id"]


def test_catalogue_cache_rejects_tampering_misbinding_and_training_use(tmp_path):
    catalogue, checkpoint, atlas, annotation = _checkpoint(tmp_path)
    checkpoint_path = tmp_path / "joint-cache-tamper-v3.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = load_arbitrary_plane_inference_v3(checkpoint_path, catalogue)
    cache = make_arbitrary_plane_catalogue_feature_cache_v3(
        loaded,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        torch.tensor([-0.5, 0.0, 0.5]),
        torch.tensor([0.25, 0.5, 0.25]),
        tmp_path / "tamper-cache.pt",
        retrieval_shape_h_w=(8, 8),
        build_chunk_size=2,
        annotation_volume_ap_dv_ml=annotation,
    )
    tampered = copy.deepcopy(cache)
    tampered["atlas_features"].view(-1)[0] += 1.0
    with pytest.raises(ValueError, match="binding or complete coverage"):
        verify_arbitrary_plane_catalogue_feature_cache_v3(tampered, loaded, catalogue)
    misbound = copy.deepcopy(cache)
    misbound["cache_receipt"]["catalogue_binding"]["catalogue_id"] = "wrong"
    with pytest.raises(ValueError, match="binding or complete coverage"):
        verify_arbitrary_plane_catalogue_feature_cache_v3(misbound, loaded, catalogue)
    loaded["model"].pose_model.train()
    with pytest.raises(ValueError, match="inference-only"):
        with loaded["model"].pose_model.use_complete_catalogue_feature_cache(
            cache["atlas_features"], cache["cell_id"], (8, 8)
        ):
            pass


def test_sealed_session_verifies_large_immutable_dependencies_once(tmp_path, monkeypatch):
    catalogue, checkpoint, atlas, annotation = _checkpoint(tmp_path)
    checkpoint_path = tmp_path / "session-checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = load_arbitrary_plane_inference_v3(checkpoint_path, catalogue)
    cache = make_arbitrary_plane_catalogue_feature_cache_v3(
        loaded,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-0.5, 0.0, 0.5),
        (0.25, 0.5, 0.25),
        tmp_path / "session-features.pt",
        retrieval_shape_h_w=(8, 8),
        build_chunk_size=1,
        annotation_volume_ap_dv_ml=annotation,
    )
    counts = {"state": 0, "features": 0, "contract": 0}
    originals = {
        "state": inference_v3._model_state_sha256,
        "features": inference_v3._large_tensor_receipt,
        "contract": inference_v3.make_inference_contract_v3,
    }

    def counted(name):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return originals[name](*args, **kwargs)

        return wrapped

    monkeypatch.setattr(inference_v3, "_model_state_sha256", counted("state"))
    monkeypatch.setattr(inference_v3, "_large_tensor_receipt", counted("features"))
    monkeypatch.setattr(inference_v3, "make_inference_contract_v3", counted("contract"))
    session = inference_v3.open_arbitrary_plane_inference_session_v3(
        checkpoint_path,
        atlas,
        catalogue,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (-0.5, 0.0, 0.5),
        (0.25, 0.5, 0.25),
        annotation_volume_ap_dv_ml=annotation,
        catalogue_feature_cache_path=cache["cache_path"],
    )
    assert counts == {"state": 1, "features": 1, "contract": 1}
    verified_feature_pointer = session["feature_cache"]["atlas_features"].data_ptr()
    original_isfinite = torch.isfinite

    def no_feature_rescan(value, *args, **kwargs):
        if isinstance(value, torch.Tensor) and value.data_ptr() == verified_feature_pointer:
            raise AssertionError("sealed feature tensor was rescanned")
        return original_isfinite(value, *args, **kwargs)

    monkeypatch.setattr(torch, "isfinite", no_feature_rescan)

    image = torch.cat((torch.rand(1, 1, 8, 8), torch.ones(1, 2, 8, 8)), dim=1)
    keywords = dict(
        animal_ids=["session-animal"],
        specimen_ids=["session-specimen"],
        experiment_ids=["session-experiment"],
        top_k=1,
        refinement_steps=1,
        pose_only_steps=0,
        retrieval_shape_h_w=(8, 8),
        catalogue_chunk_size=1,
    )
    first = inference_v3.run_arbitrary_plane_inference_session_v3(
        session,
        image,
        **keywords,
        raw_prediction_output_path=tmp_path / "session-first.pt",
    )
    second = inference_v3.run_arbitrary_plane_inference_session_v3(
        session,
        image,
        **keywords,
        raw_prediction_output_path=tmp_path / "session-second.pt",
    )
    assert first["raw_prediction_receipt"] == second["raw_prediction_receipt"]
    assert counts == {"state": 1, "features": 1, "contract": 1}

    session["feature_cache"]["atlas_features"].view(-1)[0] += 1.0
    with pytest.raises(ValueError, match="sealed inference session changed"):
        inference_v3.run_arbitrary_plane_inference_session_v3(
            session,
            image,
            **keywords,
            raw_prediction_output_path=tmp_path / "session-tampered.pt",
        )
