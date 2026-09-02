import copy
import json

import numpy as np
import pytest
import torch

import training.arbitrary_plane_finite_slab_v4 as slab
import training.arbitrary_plane_psf_v4 as psf_v4
import training.arbitrary_plane_synthetic_generator as synthetic
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components,
    render_finite_thickness_plane,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_rendered_generator import (
    component_interval_union,
    finite_plane_raster_geometry,
    finite_render_receipt,
    make_finite_arbitrary_plane_render_from_context,
    prepare_finite_render_context,
    render_finite_arbitrary_plane,
)
from training.arbitrary_plane_support import build_annotation_support_index
from training.arbitrary_plane_synthetic_generator_v2 import (
    finite_boxcar_kernel,
    reduce_v2_slab_samples,
)


def _context(*, spacing=(9.0, 21.0, 37.0), full_support=False):
    shape = (17, 15, 13)
    if full_support:
        annotation = np.ones(shape, dtype=np.uint16)
    else:
        annotation = np.zeros(shape, dtype=np.uint16)
        annotation[2:15, 2:13, 1:12] = 7
        annotation[6:11, 5:10, 4:9] = 19
    ap, dv, ml = np.indices(shape)
    scalar = (0.2 + 0.011 * ap + 0.017 * dv + 0.023 * ml).astype(np.float32)
    support = build_annotation_support_index(
        annotation,
        atlas_id="fixture-ccf",
        atlas_version="fixture-v4",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="a" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=spacing,
        origin_um=(-71.0, 23.0, 107.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    context = prepare_finite_render_context(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="b" * 64,
        template_decoder="fixture",
        annotation_decoder="fixture",
    )
    return scalar, annotation, support, context


def _parent(context, *, seed=9191, sample_index=7):
    return make_finite_arbitrary_plane_render_from_context(
        context,
        "development",
        seed,
        (25, 27),
        sample_index=sample_index,
        margin_um=(3.0, 5.0),
        animal_id="animal-4",
        specimen_id="specimen-4a",
        experiment_id="experiment-41",
    )


@pytest.mark.parametrize("thickness_um", [25.0, 55.0, 100.0])
def test_fixed_s9_psf_and_reducer_are_exact_v2_parity(thickness_um):
    selection = slab._resolve_thickness_selection(slab.FINITE_BOXCAR, None, thickness_um)
    actual = slab.finite_psf_v4(
        slab.FINITE_BOXCAR,
        thickness_um,
        thickness_selection_sha256=selection["thickness_selection_sha256"],
    )
    assert actual == psf_v4.make_finite_psf_schedule_v4(
        slab.FINITE_BOXCAR,
        thickness_um,
        thickness_selection_sha256=selection["thickness_selection_sha256"],
    )
    expected = finite_boxcar_kernel(
        "finite_boxcar",
        thickness_um,
        axial_step_um_max=thickness_um / 8.0,
    )
    assert actual["axial_sample_count"] == 9
    assert actual["axial_offsets_um"] == expected["optical_kernel_offsets_um"]
    assert actual["axial_integer_masses"] == expected["optical_kernel_integer_masses"]
    assert actual["axial_weights"] == expected["optical_kernel_weights"]
    assert actual["axial_step_um"] == pytest.approx(thickness_um / 8.0)

    rng = np.random.default_rng(31)
    scalar = rng.normal(size=(9, 7, 8)).astype(np.float32)
    labels = rng.integers(0, 5, size=(9, 7, 8), dtype=np.int64)
    masses = np.asarray(actual["axial_integer_masses"], dtype=np.int64)
    observed = slab._reduce_samples(scalar, labels, masses, 4)
    v2 = reduce_v2_slab_samples(scalar, labels, masses, 4)
    mapping = {
        "observed_scalar_float32": "scalar",
        "centre_plane_annotation_int64": "centre_plane_annotation",
        "centre_plane_support_mask": "centre_plane_support_mask",
        "slab_brain_occupancy_float32": "slab_brain_occupancy",
        "slab_observable_support_mask": "slab_observable_support_mask",
        "slab_modal_annotation_int64": "slab_modal_annotation",
        "slab_modal_purity_float32": "slab_label_purity",
        "centre_label_psf_mass_float32": "centre_label_support_weight",
    }
    assert all(np.array_equal(observed[key], v2[value]) for key, value in mapping.items())
    assert np.array_equal(
        observed["dense_correspondence_weight_float32"],
        v2["slab_supervision_weight_or_abstention"]["dense_correspondence_weight"],
    )
    assert np.array_equal(
        observed["dense_correspondence_abstention_mask"],
        v2["slab_supervision_weight_or_abstention"]["abstention_mask"],
    )


def test_capability_hash_is_shared_exact_random_init_contract():
    capability = slab.finite_psf_capability_v4()
    assert capability == psf_v4.finite_psf_model_capability_v4()
    assert capability["receipt_sha256"] == (
        "bcd6441a685e902fb5b59e85bb7003ef3261207d906a0b9390d4a219c3ae3d3e"
    )
    assert capability["prior_model_weight_dependencies"] == []
    assert capability["prior_feature_dependencies"] == []
    assert capability["prior_pseudolabel_dependencies"] == []
    assert capability["unknown_thickness_policy"] == "reject"


def test_bool_array_receipt_has_frozen_raw_c_byte_digest():
    mask = np.asarray([[True, False, True], [False, False, True]], dtype=bool)
    assert slab._array_receipt(mask) == {
        "dtype": "|b1",
        "shape": [2, 3],
        "array_sha256": "a97b5094b21f045007fe5eebc101947d15ce1c9e09fad0e8ae68dfebd4a1eee8",
    }


def test_adapter_replays_binds_provenance_and_never_mutates_or_redraws_parent():
    _, _, support, context = _context()
    parent = _parent(context)
    frozen_parent_receipt = finite_render_receipt(parent)
    result = slab.make_finite_slab_render_v4(
        parent,
        context,
        nominal_cut_thickness_um=55.0,
    )
    replayed = slab.replay_finite_slab_render_v4(result, parent, context)
    artifact = result["artifact"]
    block = artifact["slab_observation_v4"]
    targets = artifact["centre_plane_targets"]

    slab.verify_finite_slab_render_v4(result, parent, context)
    assert finite_render_receipt(parent) == frozen_parent_receipt
    assert set(result) == {"artifact"}
    assert block["finite_plane_render_id"] == parent["finite_plane_render_id"]
    assert block["plane_realization_id"] == parent["plane_realization_id"]
    assert block["finite_render_receipt_sha256"] == parent["finite_render_receipt_sha256"]
    assert block["animal_id"] == "animal-4"
    assert block["specimen_id"] == "specimen-4a"
    assert block["experiment_id"] == "experiment-41"
    assert artifact["diagnostics"]["pose_draw_count"] == 0
    assert artifact["diagnostics"]["pose_or_tissue_conditioned_rejection_count"] == 0
    assert np.array_equal(targets["centre_plane_annotation_int64"], parent["raster"]["annotation"])
    assert np.array_equal(targets["centre_plane_support_mask"], parent["raster"]["brain_mask"])
    assert block["centre_plane_targets_receipt_sha256"] == targets["receipt_sha256"]
    assert artifact["generator"]["learned_checkpoint_dependencies"] == []
    assert artifact["generator"]["previous_model_dependencies"] == []
    assert artifact["generator"]["pretrained_feature_dependencies"] == []
    assert result["artifact"]["receipt_sha256"] == replayed["artifact"]["receipt_sha256"]
    json.dumps(slab.finite_slab_render_receipt_v4(result), allow_nan=False)

    tampered = copy.deepcopy(result)
    tampered["artifact"]["slab_observation_v4"]["observed_scalar_float32"][0, 0] += 1.0
    with pytest.raises(ValueError, match="live array receipts"):
        slab.verify_finite_slab_render_v4(tampered, parent, context)
    tampered = copy.deepcopy(result)
    tampered["artifact"]["slab_observation_v4"]["finite_psf"]["axial_offsets_um"][0] += 1.0
    with pytest.raises(ValueError, match="PSF contract"):
        slab.verify_finite_slab_render_v4(tampered, parent, context)
    tampered = copy.deepcopy(result)
    tampered["artifact"]["centre_plane_targets"]["centre_plane_ccf_um_float32"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="target live array receipts"):
        slab.verify_finite_slab_render_v4(tampered, parent, context)
    wrong_parent = _parent(context, seed=31337, sample_index=17)
    with pytest.raises(ValueError, match="parent reference"):
        slab.verify_finite_slab_render_v4(result, wrong_parent, context)


def test_real_slab_producer_block_is_accepted_unchanged_by_synthetic_generator():
    _, _, support, context = _context()
    parent = _parent(context)
    adapter = slab.make_finite_slab_render_v4(
        parent,
        context,
        nominal_cut_thickness_um=55.0,
    )
    block = adapter["artifact"]["slab_observation_v4"]
    realization = synthetic.make_arbitrary_plane_synthetic_realization(
        parent,
        support,
        slab_observation_v4=block,
        root_seed=91917,
        sample_index=7,
        outline_mode=synthetic.ABSENT_OUTLINE,
    )

    copied_block = realization["slab_observation_v4"]
    assert synthetic.slab_observation_v4_receipt(copied_block) == (
        synthetic.slab_observation_v4_receipt(block)
    )
    assert all(
        np.array_equal(copied_block[name], block[name])
        for name in synthetic.SLAB_OBSERVATION_V4_ARRAY_NAMES
    )
    assert realization["slab_observation_v4_identity"]["receipt_sha256"] == block[
        "receipt_sha256"
    ]
    assert realization["slab_observation_v4_identity"]["finite_psf_sha256"] == block[
        "finite_psf"
    ]["finite_psf_sha256"]
    synthetic.verify_arbitrary_plane_synthetic_realization(realization, support)


@pytest.mark.parametrize("tensor_name", ["scalar_tensor", "annotation_tensor"])
def test_numpy_alias_byte_mutation_is_rejected_before_rendering(monkeypatch, tensor_name):
    _, _, _, context = _context()
    parent = _parent(context)
    tensor = context[tensor_name]
    original_version = tensor._version
    tensor.numpy().reshape(-1)[0] += 1
    assert tensor._version == original_version

    monkeypatch.setattr(
        slab,
        "_render_samples",
        lambda *args, **kwargs: pytest.fail("renderer ran before live asset authentication"),
    )
    with pytest.raises(ValueError, match="authenticated bytes changed"):
        slab.make_finite_slab_render_v4(
            parent,
            context,
            nominal_cut_thickness_um=55.0,
        )


def test_numpy_alias_byte_mutation_is_rejected_before_replay_rendering(monkeypatch):
    _, _, _, context = _context()
    parent = _parent(context)
    result = slab.make_finite_slab_render_v4(
        parent,
        context,
        nominal_cut_thickness_um=55.0,
    )
    tensor = context["scalar_tensor"]
    original_version = tensor._version
    tensor.numpy().reshape(-1)[0] += 1.0
    assert tensor._version == original_version

    monkeypatch.setattr(
        slab,
        "_render_samples",
        lambda *args, **kwargs: pytest.fail("replay renderer ran before live asset authentication"),
    )
    with pytest.raises(ValueError, match="authenticated bytes changed"):
        slab.replay_finite_slab_render_v4(result, parent, context)


def test_thickness_seed_is_independent_of_parent_pose_and_changes_only_descendant():
    _, _, _, context = _context()
    parent_a = _parent(context, seed=111, sample_index=3)
    parent_b = _parent(context, seed=222, sample_index=9)
    first = slab.make_finite_slab_render_v4(parent_a, context, thickness_seed=123456)
    repeat = slab.make_finite_slab_render_v4(parent_b, context, thickness_seed=123456)
    changed = slab.make_finite_slab_render_v4(parent_a, context, thickness_seed=123457)
    a = first["artifact"]["slab_observation_v4"]
    b = repeat["artifact"]["slab_observation_v4"]
    c = changed["artifact"]["slab_observation_v4"]

    assert parent_a["plane_realization_id"] != parent_b["plane_realization_id"]
    assert a["thickness_selection"]["nominal_cut_thickness_um"] == (
        b["thickness_selection"]["nominal_cut_thickness_um"]
    )
    assert a["thickness_selection"]["nominal_cut_thickness_um"] != (
        c["thickness_selection"]["nominal_cut_thickness_um"]
    )
    assert a["plane_realization_id"] == c["plane_realization_id"] == parent_a["plane_realization_id"]
    assert (
        first["artifact"]["centre_plane_targets"]["receipt_sha256"]
        == changed["artifact"]["centre_plane_targets"]["receipt_sha256"]
    )
    assert a["slab_observation_id"] != c["slab_observation_id"]


def test_zero_thickness_is_byte_exact_centre_plane_ablation():
    _, _, _, context = _context()
    parent = _parent(context)
    result = slab.make_finite_slab_render_v4(
        parent,
        context,
        render_mode=slab.CENTRE_PLANE_ABLATION,
        nominal_cut_thickness_um=0.0,
    )
    block = result["artifact"]["slab_observation_v4"]
    targets = result["artifact"]["centre_plane_targets"]
    brain = parent["raster"]["brain_mask"]

    slab.verify_finite_slab_render_v4(result, parent, context)
    assert block["finite_psf"]["axial_offsets_um"] == [0.0]
    assert block["finite_psf"]["axial_weights"] == [1.0]
    assert np.array_equal(block["observed_scalar_float32"], parent["raster"]["scalar"])
    assert np.array_equal(targets["centre_plane_annotation_int64"], parent["raster"]["annotation"])
    assert np.array_equal(block["slab_brain_occupancy_float32"], brain.astype(np.float32))
    assert np.array_equal(block["slab_observable_support_mask"], brain)
    assert np.array_equal(block["slab_modal_annotation_int64"], parent["raster"]["annotation"])
    assert np.array_equal(block["slab_modal_purity_float32"], np.ones(brain.shape, np.float32))
    assert np.array_equal(block["centre_label_psf_mass_float32"], np.ones(brain.shape, np.float32))
    assert np.array_equal(block["dense_correspondence_weight_float32"], brain.astype(np.float32))
    assert np.array_equal(block["dense_correspondence_abstention_mask"], ~brain)


def test_nonzero_offsets_use_one_batch_and_are_byte_exact_to_serial_rendering(monkeypatch):
    _, _, _, context = _context()
    parent = _parent(context)
    selection = slab._resolve_thickness_selection(slab.FINITE_BOXCAR, None, 55.0)
    finite_psf = slab.finite_psf_v4(
        slab.FINITE_BOXCAR,
        55.0,
        thickness_selection_sha256=selection["thickness_selection_sha256"],
    )
    batched_arrays, _, diagnostics = slab._render_samples(parent, context, finite_psf)
    original = slab.render_arbitrary_plane
    calls = []

    def serial_emulation(volume, centers, frames, bases, output_shape, labels, **kwargs):
        calls.append(int(centers.shape[0]))
        images = []
        sampled_labels = []
        for index in range(centers.shape[0]):
            image, sampled = original(
                volume,
                centers[index],
                frames[index],
                bases[index],
                output_shape,
                labels,
                **kwargs,
            )
            images.append(image)
            sampled_labels.append(sampled)
        return torch.cat(images), torch.cat(sampled_labels)

    monkeypatch.setattr(slab, "render_arbitrary_plane", serial_emulation)
    serial_arrays, _, _ = slab._render_samples(parent, context, finite_psf)

    assert calls == [8]
    assert diagnostics["renderer_call_count"] == 1
    assert diagnostics["nonzero_offsets_rendered_in_one_batch"] is True
    assert all(np.array_equal(array, serial_arrays[name]) for name, array in batched_arrays.items())


@pytest.mark.parametrize("normal", ([1.0, 0.0, 0.0], [1.0, -2.0, 3.0]))
def test_scalar_render_matches_differentiable_renderer_for_axis_and_oblique_anisotropic(normal):
    scalar, annotation, support, context = _context(spacing=(9.0, 21.0, 37.0))
    interval = component_interval_union(normal, support)[0]
    geometry = finite_plane_raster_geometry(
        normal,
        float(interval.mean()),
        0.713,
        support,
        (23, 29),
        (2.0, 4.0),
    )
    centre = render_finite_arbitrary_plane(scalar, annotation, geometry)
    fake_parent = {"geometry": geometry, "raster": centre}
    selection = slab._resolve_thickness_selection(slab.FINITE_BOXCAR, None, 55.0)
    finite_psf = slab.finite_psf_v4(
        slab.FINITE_BOXCAR,
        55.0,
        thickness_selection_sha256=selection["thickness_selection_sha256"],
    )
    arrays, _, diagnostics = slab._render_samples(fake_parent, context, finite_psf)

    physical_ouv = torch.as_tensor(
        geometry["effective_physical_ouv_ap_dv_ml_um"], dtype=torch.float32
    )
    state = full_frame_state_from_components(*physical_ouv_to_frame(physical_ouv))
    differentiable = render_finite_thickness_plane(
        context["scalar_tensor"][None],
        state,
        tuple(geometry["output_shape_h_w"]),
        tuple(support["origin_um"]),
        tuple(support["voxel_size_um"]),
        torch.tensor(finite_psf["axial_offsets_um"], dtype=torch.float32),
        torch.tensor(finite_psf["axial_weights"], dtype=torch.float32),
    )[0, 0].detach().numpy()

    assert np.allclose(arrays["observed_scalar_float32"], differentiable, atol=2e-5, rtol=2e-5)
    assert diagnostics["maximum_coordinate_raster_tangential_displacement_error_um"] < 0.01
    assert diagnostics["maximum_coordinate_raster_axial_displacement_error_um"] < 0.01


def test_global_zero_padding_attenuates_atlas_edge_without_per_pixel_renormalization():
    scalar, annotation, support, context = _context(
        spacing=(10.0, 20.0, 30.0), full_support=True
    )
    scalar = np.ones_like(scalar, dtype=np.float32)
    context = prepare_finite_render_context(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/ones.nrrd",
        scalar_source_sha256="c" * 64,
        template_decoder="fixture",
        annotation_decoder="fixture",
    )
    normal = [1.0, 0.0, 0.0]
    interval = component_interval_union(normal, support)[0]
    geometry = finite_plane_raster_geometry(
        normal,
        float(interval[0]),
        0.0,
        support,
        (19, 21),
    )
    centre = render_finite_arbitrary_plane(scalar, annotation, geometry)
    fake_parent = {"geometry": geometry, "raster": centre}
    selection = slab._resolve_thickness_selection(slab.FINITE_BOXCAR, None, 100.0)
    finite_psf = slab.finite_psf_v4(
        slab.FINITE_BOXCAR,
        100.0,
        thickness_selection_sha256=selection["thickness_selection_sha256"],
    )
    arrays, _, _ = slab._render_samples(fake_parent, context, finite_psf)
    centre_support = arrays["centre_plane_support_mask"]
    observed = arrays["observed_scalar_float32"][centre_support]

    assert observed.size > 0
    assert np.all(observed < 1.0)
    assert np.any(arrays["slab_brain_occupancy_float32"][centre_support] < 1.0)
    assert np.all(observed > 0.0)
