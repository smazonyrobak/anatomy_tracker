import numpy as np
import pytest

from training.arbitrary_plane_synthetic_observation import (
    ABSENT_OUTLINE,
    ACCURATE_OUTLINE,
    IMPERFECT_OUTLINE,
    apply_damage,
    mix_template_and_labels,
    robust_clean_normalization,
    sample_damage_masks,
    smart_brush_input,
    synthesize_appearance,
    synthesize_damage,
)


def _arrays(shape=(47, 61)):
    y, x = np.ogrid[: shape[0], : shape[1]]
    tissue = ((x - shape[1] / 2) / 24) ** 2 + ((y - shape[0] / 2) / 18) ** 2 <= 1
    scalar = (3.0 * x + 5.0 * y).astype(np.float32)
    labels = np.zeros(shape, np.uint32)
    labels[tissue & (x < shape[1] / 2)] = 10
    labels[tissue & (x >= shape[1] / 2)] = 900_001
    return scalar, labels, tissue


def _assert_nested_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], dict):
            _assert_nested_equal(first[key], second[key])
        elif isinstance(first[key], np.ndarray):
            assert np.array_equal(first[key], second[key]), key
        else:
            assert first[key] == second[key], key


def test_robust_normalization_handles_outliers_and_constant_tissue():
    scalar, _, tissue = _arrays()
    scalar[20, 30] = 1e9
    normalized = robust_clean_normalization(scalar, tissue, (0.02, 0.98))
    assert normalized.dtype == np.float32
    assert normalized.shape == scalar.shape
    assert normalized[tissue].min() == 0.0
    assert normalized[tissue].max() == 1.0
    assert not normalized[~tissue].any()
    constant = robust_clean_normalization(np.full(scalar.shape, 7), tissue)
    assert np.all(constant[tissue] == 0.5)


def test_normalization_receipt_uses_small_tissue_minmax_and_large_tissue_quantiles():
    small_scalar, small_labels, small_tissue = _arrays((13, 17))
    small = synthesize_appearance(
        small_scalar, small_labels, small_tissue, np.random.default_rng(2),
        clean_path=True,
    )
    assert small["parameters"]["normalization"] == {
        "method": "min-max-fallback",
        "tissue_pixel_count": int(small_tissue.sum()),
        "quantiles": None,
        "lower": float(small_scalar[small_tissue].min()),
        "upper": float(small_scalar[small_tissue].max()),
    }
    large_scalar, large_labels, large_tissue = _arrays()
    large = synthesize_appearance(
        large_scalar, large_labels, large_tissue, np.random.default_rng(2),
        clean_path=True,
    )
    receipt = large["parameters"]["normalization"]
    assert receipt["method"] == "quantile"
    assert receipt["quantiles"] == [0.01, 0.99]
    assert receipt["tissue_pixel_count"] >= 256


def test_label_mixing_never_uses_integer_label_magnitude_as_intensity():
    scalar, labels, tissue = _arrays()
    normalized = robust_clean_normalization(scalar, tissue)
    remapped = labels.copy()
    remapped[labels == 10] = 1
    remapped[labels == 900_001] = np.iinfo(np.uint32).max
    first = mix_template_and_labels(
        normalized, labels, tissue, np.random.default_rng(4),
        template_weight=0.25, label_weight=0.75,
    )
    second = mix_template_and_labels(
        normalized, remapped, tissue, np.random.default_rng(4),
        template_weight=0.25, label_weight=0.75,
    )
    assert np.array_equal(first, second)
    assert 0.0 <= first.min() <= first.max() <= 1.0
    assert labels.dtype == np.uint32 and labels.max() == 900_001


def test_appearance_parameters_bind_label_style_draws_and_offset():
    scalar, labels, tissue = _arrays()
    result = synthesize_appearance(
        scalar, labels, tissue, np.random.default_rng(6),
        template_weight=0.4, label_weight=0.6, offset_range=(0.07, 0.07),
    )
    style = result["parameters"]["label_style"]
    assert style["region_ids"] == [10, 900_001]
    assert len(style["region_levels"]) == 2
    assert result["parameters"]["offset"] == 0.07


def test_clean_appearance_path_preserves_normalized_tissue_and_arbitrary_shape():
    scalar, labels, tissue = _arrays((31, 73))
    result = synthesize_appearance(
        scalar,
        labels,
        tissue,
        np.random.default_rng(8),
        clean_path=True,
        background_texture_range=(0.0, 0.0),
    )
    expected = robust_clean_normalization(scalar, tissue)
    assert np.array_equal(result["clean_grayscale"], expected)
    assert np.array_equal(result["tissue_appearance"][tissue], expected[tissue])
    assert not result["appearance_artifact_mask"].any()
    assert result["pre_damage_image"].shape == (31, 73)


def test_augmented_appearance_is_deterministic_only_from_explicit_rng():
    scalar, labels, tissue = _arrays()
    kwargs = dict(
        artifact_density_range=(0.02, 0.02),
        background_texture_range=(0.1, 0.1),
        noise_std_range=(0.02, 0.02),
        downsample_factor_range=(1.7, 1.7),
    )
    first = synthesize_appearance(
        scalar, labels, tissue, np.random.default_rng(123), **kwargs
    )
    repeated = synthesize_appearance(
        scalar, labels, tissue, np.random.default_rng(123), **kwargs
    )
    changed = synthesize_appearance(
        scalar, labels, tissue, np.random.default_rng(124), **kwargs
    )
    _assert_nested_equal(first, repeated)
    assert not np.array_equal(first["pre_damage_image"], changed["pre_damage_image"])
    assert first["appearance_artifact_mask"].dtype == bool
    assert not first["appearance_artifact_mask"][~tissue].any()


def test_damage_is_applied_after_appearance_and_mask_algebra_is_exact():
    scalar, labels, tissue = _arrays()
    appearance = synthesize_appearance(
        scalar, labels, tissue, np.random.default_rng(3), clean_path=True
    )
    physical = np.zeros(tissue.shape, bool)
    occlusion = np.zeros(tissue.shape, bool)
    artifact = np.zeros(tissue.shape, bool)
    physical[20:23, 20:24] = tissue[20:23, 20:24]
    occlusion[25:28, 30:34] = tissue[25:28, 30:34]
    artifact[15:17, 27:29] = tissue[15:17, 27:29]
    map_valid = np.ones(tissue.shape, bool)
    map_valid[:, -2:] = False
    result = apply_damage(
        appearance["pre_damage_image"],
        appearance["acquired_background"],
        tissue,
        physical,
        occlusion,
        artifact,
        occlusion_value=0.07,
        map_domain_valid_mask=map_valid,
    )
    assert np.array_equal(result["damage_mask"], physical | occlusion)
    assert np.array_equal(result["observable_footprint_mask"], tissue & ~physical)
    assert np.array_equal(
        result["observation_invalid_mask"], tissue & (physical | occlusion | artifact)
    )
    assert np.array_equal(
        result["valid_correspondence_mask"],
        map_valid & tissue & ~(physical | occlusion | artifact),
    )
    assert np.array_equal(
        result["damaged_acquired_image"][physical],
        appearance["acquired_background"][physical],
    )
    assert np.all(result["damaged_acquired_image"][occlusion] == np.float32(0.07))


def test_sampled_damage_is_deterministic_disjoint_and_tissue_bounded():
    _, _, tissue = _arrays()
    first = sample_damage_masks(
        tissue, np.random.default_rng(19),
        physical_loss_probability=1.0, occlusion_probability=1.0,
    )
    repeated = sample_damage_masks(
        tissue, np.random.default_rng(19),
        physical_loss_probability=1.0, occlusion_probability=1.0,
    )
    _assert_nested_equal(first, repeated)
    assert first["physical_loss_mask"].any() and first["occlusion_mask"].any()
    assert not np.any(first["physical_loss_mask"] & first["occlusion_mask"])
    assert not np.any((first["physical_loss_mask"] | first["occlusion_mask"]) & ~tissue)


def test_synthesize_damage_does_not_mutate_appearance_or_tissue_truth():
    scalar, labels, tissue = _arrays()
    appearance = synthesize_appearance(
        scalar, labels, tissue, np.random.default_rng(22)
    )
    image_before = appearance["pre_damage_image"].copy()
    tissue_before = tissue.copy()
    first = synthesize_damage(
        appearance["pre_damage_image"], appearance["acquired_background"], tissue,
        appearance["appearance_artifact_mask"], np.random.default_rng(23),
        physical_loss_probability=1.0, occlusion_probability=1.0,
    )
    repeated = synthesize_damage(
        appearance["pre_damage_image"], appearance["acquired_background"], tissue,
        appearance["appearance_artifact_mask"], np.random.default_rng(23),
        physical_loss_probability=1.0, occlusion_probability=1.0,
    )
    _assert_nested_equal(first, repeated)
    assert np.array_equal(appearance["pre_damage_image"], image_before)
    assert np.array_equal(tissue, tissue_before)


def test_accurate_outline_uses_footprint_not_validity_and_is_strictly_black_outside():
    scalar, labels, tissue = _arrays()
    image = synthesize_appearance(
        scalar, labels, tissue, np.random.default_rng(28)
    )["pre_damage_image"]
    footprint = tissue.copy()
    validity = tissue.copy()
    validity[18:27, 25:36] = False
    result = smart_brush_input(
        image, footprint, np.random.default_rng(29), ACCURATE_OUTLINE
    )
    assert np.array_equal(result["input_outline_mask"], footprint)
    assert result["input_outline_mask"][~validity].any()
    assert not result["input_image"][~result["input_outline_mask"]].any()
    assert result["outline_available"] is True


def test_imperfect_outline_is_deterministic_perturbed_and_strictly_black_outside():
    scalar, labels, tissue = _arrays()
    image = synthesize_appearance(
        scalar, labels, tissue, np.random.default_rng(31)
    )["pre_damage_image"]
    first = smart_brush_input(
        image, tissue, np.random.default_rng(32), IMPERFECT_OUTLINE
    )
    repeated = smart_brush_input(
        image, tissue, np.random.default_rng(32), IMPERFECT_OUTLINE
    )
    _assert_nested_equal(first, repeated)
    assert not np.array_equal(first["input_outline_mask"], tissue)
    assert not first["input_image"][~first["input_outline_mask"]].any()
    assert first["parameters"]["gap_applied"]
    assert first["parameters"]["island_applied"]


def test_absent_outline_retains_acquired_background_and_returns_no_outline():
    scalar, labels, tissue = _arrays()
    image = synthesize_appearance(
        scalar, labels, tissue, np.random.default_rng(40),
        background_level_range=(0.3, 0.3), background_texture_range=(0.0, 0.0),
    )["pre_damage_image"]
    result = smart_brush_input(
        image, tissue, np.random.default_rng(41), ABSENT_OUTLINE
    )
    assert np.array_equal(result["input_image"], image)
    assert not result["input_outline_mask"].any()
    assert result["input_image"][~tissue].any()
    assert result["outline_available"] is False
    assert result["outline_quality_iou"] is None


def test_global_random_state_is_not_an_accepted_rng():
    scalar, labels, tissue = _arrays()
    with pytest.raises(TypeError, match="explicit numpy.random.Generator"):
        synthesize_appearance(scalar, labels, tissue, np.random)  # type: ignore[arg-type]
