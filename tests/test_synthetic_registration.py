import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from source.dense_registration_preprocessing import (
    NATIVE_SHAPE,
    PAD_X,
    numpy_cosine_mask_feather,
)
from training.synthetic_registration import (
    _FINAL_HOLDOUT_CAPABILITY,
    AP_BLOCK_WIDTH,
    BREGMA_AP_INDEX,
    MODEL_SHAPE,
    QUERY_SHA256,
    SyntheticRegistrationGenerator,
    V2_SEMANTICS_VERSION,
    _payload_sha256,
    _query_hierarchy,
    _sample,
    compose_pixel_maps,
    jacobian_determinant,
    make_registration_manifest,
    save_qa_montage,
    split_ap_indices,
)
from training.synthetic_registration import _identity_grid, _sample_labels


ATLAS = Path(__file__).resolve().parents[1] / "data" / "Allen Brain Atlas 25um"

V2_GOLDEN_SOURCE_SHA256 = "5a7274b56cdfa95fdb410b1b441325c4ee9ab3b15eb8076be2781e92c00c1dde"
V2_GOLDEN_CONTRACT_SHA256 = "5ff5a29ccc7f0c554020dfc8e7e07d2af59a662f9420422bffc9a86edcd73872"
V2_GOLDEN_CASES = (
    (
        98_001, 0, -3,
        "7fc74c43e6435036feb8419b0b608bd8af5b733feb5591f912abb13d21e6c4da",
        "b070bb6f782b85057a1dc49a7603dd62a7946be44d8000cfee80d9917d8de0b6",
        "55e70f9a2711f1f544f886fe91e6a931cb8d4451a135b0e29f9dd19a73df56cc",
        "7ead4fa011ee8477ba3376998e5a376b5aa1b5fc5a0db0842664162e195f3e75",
        "d21006267f9edb683183434efaea545b79006b7840927252ca00905713c9bea5",
    ),
    (
        98_002, 1, 3,
        "773ad53ef11ffb34dfccca1d32f3003512c0ff6bdc977a518d370542e716a6a7",
        "fadcfa7c556be32cdd9c8374c2e3960d4b68d404920aea302f0183b775e1305d",
        "4e9f5579e6f4e048098e8fe85086351b50b551408ed9caadf4de797413660942",
        "da22d1ca365625f8ba3510839f3d7bc8d762f257583876cabbd89a0323c7d3c9",
        "2b3618d576631aa1112a9c5be4920d19a689ede3cba760fa937ec66885910a69",
    ),
)


def _v2_exact_pair_sha256(pair: dict) -> str:
    names = (
        "moving_raw_uint8", "fixed_labels", "moving_labels",
        "fixed_mask", "moving_tissue_mask",
        "moving_damage_mask", "moving_visible_mask", "moving_brush_mask",
        "moving_model_mask", "fixed_damage_mask", "fixed_visible_mask",
    )
    digest = hashlib.sha256()
    for name in names:
        values = pair[name].detach().cpu().contiguous().numpy()
        if values.dtype == np.bool_:
            values = values.astype(np.uint8)
        elif name == "moving_raw_uint8":
            values = values.astype(np.uint8)
        else:
            values = values.astype("<i8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _v2_quantized_map_sha256(values: torch.Tensor) -> str:
    quantized = np.rint(
        values.detach().cpu().float().numpy() * 10_000.0
    ).astype("<i4")
    return hashlib.sha256(quantized.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def generator():
    if not (ATLAS / "average_template_25.nrrd").is_file():
        pytest.skip("Local Allen atlas is not installed")
    return SyntheticRegistrationGenerator(ATLAS, "cpu")


def test_ap_block_splits_are_disjoint_and_guarded():
    train = split_ap_indices("train")
    validation = split_ap_indices("validation")
    with pytest.raises(PermissionError, match="one-shot evaluator"):
        split_ap_indices("sealed-test")
    sealed = split_ap_indices(
        "sealed-test", _final_capability=_FINAL_HOLDOUT_CAPABILITY
    )
    assert not set(train) & set(validation)
    assert not set(train) & set(sealed)
    assert not set(validation) & set(sealed)
    assert min(abs(int(value) - train).min() for value in validation) > AP_BLOCK_WIDTH
    assert min(abs(int(value) - train).min() for value in sealed) > AP_BLOCK_WIDTH
    assert int(validation.max() - validation.min()) >= 140
    assert int(sealed.max() - sealed.min()) >= 140


def test_manifest_is_deterministic_hashed_and_split_bounded(generator):
    first = generator.make_manifest(3, "validation", 51, "mild")
    second = generator.make_manifest(3, "validation", 51, "mild")
    regenerated = make_registration_manifest(
        generator.contract, 3, "validation", 51, "mild"
    )
    changed = generator.make_manifest(3, "validation", 52, "mild")
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["manifest_sha256"] == regenerated["manifest_sha256"]
    assert first["manifest_sha256"] != changed["manifest_sha256"]
    assert set(first["ap_index"].astype(int)) <= set(split_ap_indices("validation"))
    assert abs(first["tilt_lr_deg"]).max() <= 8.0
    assert abs(first["tilt_dv_deg"]).max() <= 8.0
    assert "after_upstream_pose_freeze" in generator.contract["plane_sampling"]
    assert len(generator.contract["average_template_sha256"]) == 64
    assert len(generator.contract["annotation_sha256"]) == 64
    assert first["format_version"] == 2


def test_generator_exposes_only_the_frozen_v2_contract(generator):
    assert generator.contract["profile"] == "v2"
    assert generator.contract["generator_version"] == 2
    assert "profile" not in inspect.signature(SyntheticRegistrationGenerator).parameters


@pytest.mark.parametrize(
    "seed,appearance_mode,mask_offset,manifest_sha256,exact_sha256,"
    "forward_sha256,inverse_sha256,velocity_sha256",
    V2_GOLDEN_CASES,
)
def test_v2_contract_and_golden_pairs_are_frozen(
    generator,
    seed,
    appearance_mode,
    mask_offset,
    manifest_sha256,
    exact_sha256,
    forward_sha256,
    inverse_sha256,
    velocity_sha256,
):
    assert V2_SEMANTICS_VERSION == 1
    assert generator.contract["semantics_version"] == V2_SEMANTICS_VERSION
    assert generator.contract["generator_source_sha256"] == (
        V2_GOLDEN_SOURCE_SHA256
    )
    assert generator.contract["contract_sha256"] == V2_GOLDEN_CONTRACT_SHA256

    manifest = generator.make_manifest(1, "validation", seed, "hard")
    manifest["moving_appearance_mode"][:] = appearance_mode
    manifest["mask_offset_px"][:] = mask_offset
    manifest["tear_enabled"][:] = True
    manifest["missing_enabled"][:] = True
    manifest["occlusion_enabled"][:] = True
    manifest["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    pair = generator.batch(manifest, qa=True)

    assert manifest["manifest_sha256"] == manifest_sha256
    assert int(manifest["moving_appearance_mode"][0]) == appearance_mode
    assert int(manifest["mask_offset_px"][0]) == mask_offset
    assert pair["moving_damage_mask"].any()
    assert pair["fixed_damage_mask"].any()
    assert _v2_exact_pair_sha256(pair) == exact_sha256
    assert _v2_quantized_map_sha256(pair["fixed_to_moving"]) == forward_sha256
    assert _v2_quantized_map_sha256(pair["moving_to_fixed"]) == inverse_sha256
    assert _v2_quantized_map_sha256(pair["local_velocity"]) == velocity_sha256


def test_query_hierarchy_uses_exact_integer_ids_and_deepest_fallback(tmp_path):
    query = tmp_path / "query.csv"
    query.write_text(
        "id,structure_id_path\n"
        "997,/997/\n"
        "8,/997/8/\n"
        "614454277,/997/8/567/688/695/315/614454277/\n",
        encoding="utf-8",
    )
    ids, ancestors = _query_hierarchy(query)
    position = int(np.searchsorted(ids, 614454277))
    assert ids[position] == 614454277
    assert ancestors[:, position].tolist() == [695, 315, 614454277, 614454277]
    assert ancestors[:, int(np.searchsorted(ids, 8))].tolist() == [8, 8, 8, 8]


def test_v2_manifest_fields_are_independent_typed_and_split_conditioned(generator):
    train = generator.make_manifest(200, "train", 51, "mild")
    validation = generator.make_manifest(200, "validation", 51, "mild")
    assert train["format_version"] == validation["format_version"] == 2
    assert train["moving_appearance_mode"].dtype == np.uint8
    assert train["label_style_seed"].dtype == np.uint64
    assert train["label_hierarchy_depth"].dtype == np.int8
    assert train["label_blur_sigma_px"].dtype == np.float32
    assert train["mask_offset_px"].dtype == np.int8
    assert set(train["label_hierarchy_depth"].tolist()) <= {4, 5, 6, 7}
    assert set(train["mask_offset_px"].tolist()) <= {-2, -1, 0, 1, 2}
    assert abs(float(train["moving_appearance_mode"].mean()) - 0.70) < 0.08
    assert abs(float(validation["moving_appearance_mode"].mean()) - 0.50) < 0.08
    assert generator.contract["query_sha256"] == QUERY_SHA256


def test_batch_contract_has_no_legacy_aliases_and_qa_is_explicit(generator):
    manifest = generator.make_manifest(1, "validation", 73, "clean")
    production = generator.batch(manifest)
    expected = {
        "fixed", "moving", "fixed_mask", "moving_tissue_mask",
        "moving_damage_mask", "moving_visible_mask", "moving_model_mask",
        "fixed_damage_mask", "fixed_visible_mask", "fixed_labels",
        "moving_labels", "fixed_to_moving", "moving_to_fixed",
        "local_velocity", "similarity_h", "manifest_sha256", "contract",
    }
    assert set(production) == expected
    qa = generator.batch(manifest, qa=True)
    assert set(qa) == expected | {
        "moving_clean", "moving_raw_uint8", "moving_appearance_clean",
        "moving_brush_mask",
    }


def test_zero_tilt_plane_preserves_native_pixels_and_full_allen_ids(generator):
    image, mask, labels = generator.render_planes(
        torch.tensor([BREGMA_AP_INDEX]), torch.zeros(1), torch.zeros(1)
    )
    expected = generator.annotation[int(BREGMA_AP_INDEX)].long()
    left = generator.pad_x
    observed = labels[0, 0, :, left : left + expected.shape[1]]
    assert image.shape == mask.shape == labels.shape == (1, 1, *MODEL_SHAPE)
    assert torch.equal(observed, expected)
    assert labels.unique().numel() > 9
    assert torch.equal(mask, labels > 0)


def test_label_sampling_preserves_ids_above_float32_integer_precision():
    labels = torch.tensor([[[[0, 16_777_217], [614_454_277, 3]]]], dtype=torch.int64)
    identity = _identity_grid(1, 2, 2, "cpu")
    assert torch.equal(_sample_labels(labels, identity), labels)


@pytest.mark.parametrize("stratum", ["clean", "mild", "hard"])
def test_generated_pair_contract_shapes_ranges_and_positive_jacobian(generator, stratum):
    pair = generator.generate(1, "train", 180 + len(stratum), stratum, qa=True)
    for name in ("fixed", "moving", "moving_clean"):
        assert pair[name].shape == (1, 1, *MODEL_SHAPE)
        assert pair[name].dtype == torch.float32
        assert 0.0 <= float(pair[name].min()) <= float(pair[name].max()) <= 1.0
    for name in ("fixed_to_moving", "moving_to_fixed", "local_velocity"):
        assert pair[name].shape == (1, 2, *MODEL_SHAPE)
        assert torch.isfinite(pair[name]).all()
    assert pair["fixed_labels"].dtype == pair["moving_labels"].dtype == torch.int64
    assert set(pair["moving_labels"].unique().tolist()) <= set(pair["fixed_labels"].unique().tolist())
    assert pair["moving_visible_mask"].dtype == pair["fixed_visible_mask"].dtype == torch.bool
    assert torch.all(pair["moving_visible_mask"] <= pair["moving_tissue_mask"])
    cells = (
        pair["fixed_mask"][:, :, :-1, :-1] | pair["fixed_mask"][:, :, :-1, 1:]
        | pair["fixed_mask"][:, :, 1:, :-1] | pair["fixed_mask"][:, :, 1:, 1:]
    )[:, 0]
    assert jacobian_determinant(pair["fixed_to_moving"])[cells].min() > 0.0


def test_adversarial_hard_batch_has_no_forward_or_inverse_folds(generator):
    pair = generator.generate(4, "train", 10022, "hard")
    assert jacobian_determinant(pair["fixed_to_moving"]).min() > 0.0
    assert jacobian_determinant(pair["moving_to_fixed"]).min() > 0.0


def test_known_forward_inverse_maps_close_cycle_inside_valid_tissue(generator):
    pair = generator.generate(1, "train", 907, "hard")
    cycle = compose_pixel_maps(pair["fixed_to_moving"], pair["moving_to_fixed"])
    height, width = MODEL_SHAPE
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    identity = torch.stack((x, y), dim=0)[None].float()
    interior = pair["fixed_visible_mask"].clone()
    interior[:, :, :20] = interior[:, :, -20:] = False
    interior[:, :, :, :20] = interior[:, :, :, -20:] = False
    error = torch.linalg.vector_norm(cycle - identity, dim=1)[interior[:, 0]]
    assert torch.quantile(error, 0.95) < 0.35


def test_optical_damage_does_not_change_geometry_targets(generator):
    manifest = generator.make_manifest(1, "train", 1201, "hard")
    pair = generator.batch(manifest)
    changed = dict(manifest)
    changed["noise"] = changed["noise"] * 0
    changed["blur"] = changed["blur"] * 0
    changed["tile_strength"] = changed["tile_strength"] * 0
    changed["blowout_strength"] = changed["blowout_strength"] * 0
    changed["speck_density"] = changed["speck_density"] * 0
    changed["tear_enabled"] = changed["tear_enabled"] & False
    changed["missing_enabled"] = changed["missing_enabled"] & False
    changed["occlusion_enabled"] = changed["occlusion_enabled"] & False
    changed["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in changed.items() if key != "manifest_sha256"}
    )
    undamaged = generator.batch(changed)
    assert torch.equal(pair["fixed_to_moving"], undamaged["fixed_to_moving"])
    assert torch.equal(pair["moving_to_fixed"], undamaged["moving_to_fixed"])
    assert not torch.equal(pair["moving"], undamaged["moving"])


def test_v2_masks_quantization_and_shared_feather_follow_runtime_contract(generator):
    manifest = generator.make_manifest(1, "train", 1201, "hard")
    manifest["tear_enabled"][:] = True
    manifest["missing_enabled"][:] = True
    manifest["occlusion_enabled"][:] = True
    manifest["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    pair = generator.batch(manifest, qa=True)
    tissue = pair["moving_tissue_mask"]
    tear, missing, occlusion = generator._damage_masks(tissue, manifest)
    damage = (tear | missing | occlusion) & tissue
    visible = tissue & ~damage
    brush = tissue & ~missing
    native_brush = brush[..., : NATIVE_SHAPE[0], PAD_X : PAD_X + NATIVE_SHAPE[1]]
    model = torch.nn.functional.pad(
        generator._offset_mask(native_brush, manifest["mask_offset_px"]),
        (PAD_X, PAD_X),
    )
    assert torch.equal(pair["moving_damage_mask"], damage)
    assert torch.equal(pair["moving_visible_mask"], visible)
    assert torch.equal(pair["moving_brush_mask"], brush)
    assert torch.equal(pair["moving_model_mask"], model)
    fixed_damage = pair["fixed_mask"] & ~pair["fixed_visible_mask"]
    assert torch.equal(pair["fixed_damage_mask"], fixed_damage)
    assert torch.equal(
        pair["fixed_visible_mask"] | pair["fixed_damage_mask"],
        pair["fixed_mask"],
    )
    assert not torch.any(
        pair["fixed_visible_mask"] & pair["fixed_damage_mask"]
    )
    native_alpha = numpy_cosine_mask_feather(
        model[0, 0, :, PAD_X : PAD_X + NATIVE_SHAPE[1]].numpy()
    )
    alpha = torch.nn.functional.pad(
        torch.from_numpy(native_alpha)[None, None], (PAD_X, PAD_X)
    )
    expected = pair["moving_raw_uint8"].float() / 255.0 * alpha
    torch.testing.assert_close(pair["moving"], expected, rtol=0.0, atol=0.0)
    for name in (
        "moving", "moving_raw_uint8", "moving_tissue_mask",
        "moving_damage_mask", "moving_brush_mask", "moving_model_mask",
    ):
        assert not pair[name][..., :PAD_X].any()
        assert not pair[name][..., PAD_X + NATIVE_SHAPE[1] :].any()


def test_v2_label_style_and_mask_fields_cannot_change_geometry(generator):
    template = generator.make_manifest(1, "validation", 4401, "mild")
    template["moving_appearance_mode"][:] = 0
    template["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in template.items() if key != "manifest_sha256"}
    )
    labelled = dict(template)
    labelled["moving_appearance_mode"] = np.ones(1, dtype=np.uint8)
    labelled["label_style_seed"] = np.asarray([np.uint64(0xFEDCBA9876543210)])
    labelled["label_hierarchy_depth"] = np.asarray([7], dtype=np.int8)
    labelled["label_blur_sigma_px"] = np.asarray([1.5], dtype=np.float32)
    labelled["mask_offset_px"] = np.asarray([2], dtype=np.int8)
    labelled["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in labelled.items() if key != "manifest_sha256"}
    )
    first = generator.batch(template, qa=True)
    second = generator.batch(labelled, qa=True)
    for name in (
        "fixed", "fixed_mask", "fixed_labels", "moving_clean", "moving_tissue_mask",
        "moving_labels", "fixed_to_moving", "moving_to_fixed", "local_velocity",
        "similarity_h",
    ):
        assert torch.equal(first[name], second[name])
    assert not torch.equal(first["moving_raw_uint8"], second["moving_raw_uint8"])


def test_same_manifest_recreates_the_same_pair(generator):
    manifest = generator.make_manifest(1, "validation", 4411, "mild")
    first = generator.batch(manifest)
    second = generator.batch(manifest)
    for name in (
        "fixed", "moving", "fixed_labels", "moving_labels",
        "fixed_to_moving", "moving_to_fixed", "moving_visible_mask",
    ):
        assert torch.equal(first[name], second[name])


def test_generator_public_api_cannot_open_the_final_holdout(generator):
    with pytest.raises(PermissionError, match="one-shot evaluator"):
        generator.make_manifest(1, "sealed-test", 4411, "mild")
    with pytest.raises(PermissionError, match="one-shot evaluator"):
        generator.generate(1, "sealed-test", 4411, "mild")


def test_batch_revalidates_manifest_split_membership(generator):
    manifest = generator.make_manifest(1, "validation", 812, "clean")
    manifest["ap_index"] = manifest["ap_index"].copy()
    manifest["ap_index"][0] = split_ap_indices("train")[0]
    manifest["manifest_sha256"] = _payload_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="outside its declared split"):
        generator.batch(manifest)


def test_final_manifest_requires_capability_when_materialized(generator):
    manifest = generator.make_manifest(
        1,
        "sealed-test",
        813,
        "clean",
        _final_capability=_FINAL_HOLDOUT_CAPABILITY,
    )
    with pytest.raises(PermissionError, match="one-shot evaluator"):
        generator.batch(manifest)
    pair = generator.generate(
        1,
        "sealed-test",
        813,
        "clean",
        _final_capability=_FINAL_HOLDOUT_CAPABILITY,
    )
    assert pair["fixed"].shape == (1, 1, *MODEL_SHAPE)


def test_qa_montage_is_written(generator, tmp_path):
    pair = generator.generate(1, "train", 991, "hard", qa=True)
    destination = save_qa_montage(pair, tmp_path / "registration-pairs.png")
    from PIL import Image
    with Image.open(destination) as montage:
        assert montage.size == (MODEL_SHAPE[1] * 8, MODEL_SHAPE[0] + 22)
