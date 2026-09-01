import copy
import json

import numpy as np
import pytest
import torch

import training.arbitrary_plane_acquisition_v2 as acquisition
from training.arbitrary_plane_acquisition_v2 import (
    V2_PLANE_SCHEMA,
    V2_PLANE_STRATA,
    derive_v2_field_seed,
    global_reference_plane_geometry,
    global_reference_support_geometry,
    make_v2_smoke_global_reference_centre_render,
    prepare_arbitrary_plane_acquisition_context_v2,
    replay_v2_smoke_global_reference_centre_render,
    sample_v2_smoke_plane_pose,
    shifted_component_interval_union,
    v2_centre_render_receipt,
    verify_v2_smoke_global_reference_centre_render,
)
from training.arbitrary_plane_rendered_generator import (
    effective_renderer_sampling_arrays,
    render_finite_arbitrary_plane,
)
from training.arbitrary_plane_support import build_annotation_support_index


def _volumes(projection_origin_um=None):
    annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    annotation[2:15, 3:13, 1:11] = 7
    annotation[7:11, 4:8, 7:11] = 19
    ap, dv, ml = np.indices(annotation.shape)
    scalar = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.float32)
    support = build_annotation_support_index(
        annotation,
        atlas_id="fixture-ccf",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(11.0, 17.0, 29.0),
        origin_um=(-71.0, 23.0, 107.0),
        projection_origin_um=projection_origin_um,
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return scalar, annotation, support


@pytest.fixture(scope="module")
def prepared():
    scalar, annotation, support = _volumes()
    context = prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="fixture decoder",
        annotation_decoder="fixture decoder",
    )
    return scalar, annotation, support, context


def test_v2_rng_has_frozen_known_vectors_and_separates_every_domain():
    root = 0x415154564F320001
    cases = [
        ("development", 0, "pose", "axis-cosine", 0),
        ("train", 0, "pose", "axis-cosine", 0),
        ("development", 1, "pose", "axis-cosine", 0),
        ("development", 0, "pose", "axis-cosine", 1),
        ("development", 0, "window", "axis-cosine", 0),
        ("development", 0, "pose", "roll", 0),
    ]
    expected = [
        "311ee7a6519bb08e",
        "229110201f5466bc",
        "dede0384bd3c9183",
        "d85829c27f057d12",
        "7fa177c5d361a913",
        "09f2022f051fe944",
    ]
    observed = [f"{derive_v2_field_seed(root, *case):016x}" for case in cases]
    assert observed == expected
    assert len(set(observed)) == len(observed)


def test_global_fov_uses_closed_voxel_faces_and_is_pose_independent(prepared):
    _, _, support, _ = prepared
    fov = global_reference_support_geometry(support)
    spacing = np.asarray(support["voxel_size_um"])
    origin = np.asarray(support["origin_um"])
    expected_lower = origin + np.asarray([2, 3, 1]) * spacing
    expected_upper = origin + np.asarray([15, 13, 11]) * spacing
    expected_center = (expected_lower + expected_upper) / 2
    expected_radius = np.linalg.norm((expected_upper - expected_lower) / 2)

    assert np.array_equal(fov["closed_face_lower_ap_dv_ml_um"], expected_lower)
    assert np.array_equal(fov["closed_face_upper_ap_dv_ml_um"], expected_upper)
    assert np.array_equal(fov["support_origin_ap_dv_ml_um"], expected_center)
    assert fov["diameter_um"] == pytest.approx(
        2 * (expected_radius + np.linalg.norm(spacing)), abs=1e-12
    )
    assert fov["parent_shape_h_w"] == [256, 256]


def test_projection_intervals_are_shifted_to_support_origin_without_filling_gaps():
    annotation = np.zeros((19, 5, 5), dtype=np.uint16)
    annotation[1:3, 1:4, 1:4] = 1
    annotation[13:18, 1:4, 1:4] = 2
    support = build_annotation_support_index(
        annotation,
        atlas_id="fixture-disconnected",
        atlas_version="v1",
        source_uri="file:///fixture/disconnected.nrrd",
        source_sha256="5" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(7.0, 11.0, 13.0),
        origin_um=(10.0, 20.0, 30.0),
        projection_origin_um=(500.0, -200.0, 50.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    fov = global_reference_support_geometry(support)
    shifted = shifted_component_interval_union(
        np.asarray([1.0, 0.0, 0.0]), support
    )
    intervals = np.asarray(shifted["support_origin_interval_union_um"])

    assert np.allclose(intervals, [[-59.5, -45.5], [24.5, 59.5]], atol=1e-12)
    assert intervals[0, 1] < intervals[1, 0]
    for fraction in (0.01, 0.25, 0.50, 0.75, 0.99):
        offset, _ = acquisition._offset_at_measure_fraction(intervals, fraction)
        assert np.any((offset >= intervals[:, 0]) & (offset <= intervals[:, 1]))


def test_global_plane_grid_has_exact_endpoints_antipodal_identity_and_bound_float32_grid(prepared):
    _, _, support, _ = prepared
    normal = np.asarray([1.0, -2.0, 3.0])
    fov = global_reference_support_geometry(support)
    intervals = shifted_component_interval_union(
        normal, support
    )
    offset = float(np.asarray(intervals["support_origin_interval_union_um"])[0].mean())
    geometry = global_reference_plane_geometry(normal, offset, 1.137, support)
    antipodal = global_reference_plane_geometry(-normal, -offset, 1.137, support)
    physical_ouv = np.asarray(geometry["physical_ouv_ap_dv_ml_um"])
    origin, edge_u, edge_v = physical_ouv[:3], physical_ouv[3:6], physical_ouv[6:9]
    center = np.asarray(geometry["plane_center_ap_dv_ml_um"])

    assert geometry["global_reference_grid_id"] == antipodal["global_reference_grid_id"]
    assert np.allclose(origin, center - 0.5 * edge_u * 255 / 256 - 0.5 * edge_v * 255 / 256)
    assert np.allclose(
        np.asarray(geometry["raster_endpoint_semantics"]["last_sample_ap_dv_ml_um"]),
        center + 0.5 * edge_u * 255 / 256 + 0.5 * edge_v * 255 / 256,
        atol=1e-9,
    )
    allen_ouv = np.asarray(geometry["allen_index_ouv_ap_dv_ml"])
    spacing = np.asarray(support["voxel_size_um"])
    atlas_origin = np.asarray(support["origin_um"])
    assert np.allclose(atlas_origin + (allen_ouv[:3] + 0.5) * spacing, origin, atol=1e-9)
    assert np.allclose(allen_ouv[3:6] * spacing, edge_u, atol=1e-9)
    assert np.allclose(allen_ouv[6:9] * spacing, edge_v, atol=1e-9)
    effective = effective_renderer_sampling_arrays(
        geometry,
        tuple(support["annotation_shape"]),
        origin_ap_dv_ml_um=tuple(support["origin_um"]),
        voxel_size_ap_dv_ml_um=tuple(support["voxel_size_um"]),
    )
    for name, values in effective.items():
        assert acquisition._array_receipt(values) == geometry["array_receipts"][name]
    assert geometry["diagnostics"]["physical_index_roundtrip_max_abs_um"] <= 1e-9
    assert geometry["diagnostics"]["plane_residual_max_abs_um"] <= 1e-9
    assert geometry["diagnostics"]["support_corner_minimum_inplane_clearance_um"] >= (
        fov["margin_um"] - 1e-9
    )


def test_exact_twenty_case_sampler_covers_all_plane_strata_without_rejection(prepared):
    _, _, support, _ = prepared
    root = "0x415154564f320001"
    observed = []
    for sample_index in range(20):
        stratum = V2_PLANE_STRATA[sample_index // 4]
        sample = sample_v2_smoke_plane_pose(
            support, "development", root, sample_index, stratum
        )
        normal = np.abs(np.asarray(sample["normal_rp2_ap_dv_ml"]))
        intervals = np.asarray(sample["shifted_intervals"]["support_origin_interval_union_um"])
        offset = sample["signed_offset_um_about_support_origin"]
        assert np.any((offset >= intervals[:, 0]) & (offset <= intervals[:, 1]))
        assert sample["rejection_attempts"] == []
        assert set(sample["field_stream_attempt_index"].values()) == {0}
        if stratum.startswith("near_"):
            axis = {"near_AP": 0, "near_DV": 1, "near_ML": 2}[stratum]
            assert 0.90 <= normal[axis] < 0.985
        elif stratum == "general_oblique":
            assert normal.max() < 0.90
        else:
            if sample_index % 2 == 0:
                assert 0.01 <= sample["offset_measure_fraction"] < 0.03
            else:
                assert 0.97 < sample["offset_measure_fraction"] <= 0.99
        observed.append(stratum)
    assert {stratum: observed.count(stratum) for stratum in V2_PLANE_STRATA} == {
        stratum: 4 for stratum in V2_PLANE_STRATA
    }


def test_centre_render_matches_frozen_primitive_and_linear_phantom(prepared):
    scalar, annotation, support, context = prepared
    artifact = make_v2_smoke_global_reference_centre_render(
        context,
        "development",
        "0x415154564f320001",
        0,
        "near_AP",
        animal_id="animal-1",
        specimen_id="specimen-1a",
        experiment_id="experiment-11",
    )
    direct = render_finite_arbitrary_plane(scalar, annotation, artifact["geometry"])
    effective = effective_renderer_sampling_arrays(
        artifact["geometry"], tuple(annotation.shape)
    )["coordinate_raster_allen_index_float32"]
    valid = np.ones(effective.shape[:2], dtype=bool)
    for axis, size in enumerate(annotation.shape):
        valid &= (effective[..., axis] >= 0) & (effective[..., axis] <= size - 1)
    expected = 100 + 3 * effective[..., 0] + 5 * effective[..., 1] + 7 * effective[..., 2]
    scale = float(scalar.max() - scalar.min())

    assert artifact["schema_version"] == V2_PLANE_SCHEMA
    assert all(
        np.array_equal(artifact["raster"][name], direct[name])
        for name in ("scalar", "annotation", "brain_mask")
    )
    assert np.max(np.abs(artifact["raster"]["scalar"][valid] - expected[valid])) / scale <= 1e-5
    assert artifact["raster"]["annotation"].dtype == np.int64
    assert set(np.unique(artifact["raster"]["annotation"])) <= {0, 7, 19}
    assert artifact["generator"]["resolved_config"]["previous_model_dependencies"] == []
    assert artifact["generator"]["resolved_config"]["preflight"]["receipt_id"] == (
        "arbitrary-plane-acquisition-hardening-2026-09-01"
    )
    assert artifact["generator"]["resolved_config_sha256"] == acquisition._payload_sha256(
        artifact["generator"]["resolved_config"]
    )
    assert artifact["provenance"]["animal_id"] == "animal-1"
    assert artifact["smoke_case_assignment"] == {
        "sample_index": 0,
        "plane_stratum": "near_AP",
        "window_plan_severity": "standard",
        "reflection": "none",
        "render_mode": "centre_plane_ablation",
        "nominal_cut_thickness_um": 50.0,
        "thickness_class": None,
        "effective_optical_support_um": 0.0,
    }
    assert not {
        "slab_render_id",
        "acquisition_window_realization_id",
        "reflection_realization_id",
        "v2_acquisition_realization_id",
    } & artifact.keys()
    json.dumps(v2_centre_render_receipt(artifact), allow_nan=False)


def test_all_twenty_centre_renders_replay_byte_exactly(prepared):
    _, _, _, context = prepared
    identities = []
    observed = []
    expected = [
        ("db55361f85c73be8a902d5bb856ebe7664b696614e1bcbe52b473a2de6cf0466", "ceaff85f49cc634d4c9c7c24aec0b791de8392248f45105bc331036472759320", "a92329eaa4225219a995d8269ce07307de224dd43870b36f60dc431d8219e3d7"),
        ("30e5906ba14ffe85cb770b47f96b064ea4cd4878c4fa7f2c966762830b9fe5e7", "08ab9a01c98066258eb2c7d3643234fcfa44d7126232b82594bade77293dfc81", "2fabb826e7dc34c7a84bf678a67346c315837519d1f15451cd6476a1cf5d18bc"),
        ("42d967a1c4459728257bddad33de42105a81a0120a3a106cf505b164014f8338", "a38a76c08942804f2c8ba02ffeedc846f8d4ddfe22d019f8bbeb81215c7d5360", "78026ba249e8c360494b13cb42be02709de0bdea0c315a4ac51b99f93f140074"),
        ("0fc6b57deaf921c982d9bc1bb62161c91f5405b7378af7830564ddf3dd159797", "9f8908a5cb722cf23afef66d481a46455d3b22ae913d1d97124566c762216924", "3f09b938100a2448341b0b72ef176ee902e66aed73eb5bce67a87afa26cb268d"),
        ("f40aff1d3e51fbd5f124ac94a6de66a99eafa0c56f20b4366dbbe46d5d7ca5d9", "14a1ac9a6ae166b310fc1c48360fb149e6d7e543e09e8b4db6724958309be9d6", "be027634e70afca91344ddd6a3529318415002c5a399e85c1762582491a68e9a"),
        ("a432b8829141476af175846a14f3c6ca6e99df5d67786b25478e5b43dc84860c", "f8d5f146e5cbe373cd25070e12bd3c0d2c181995711c606946706a8e386c3e5b", "fc51dee0fafd045de00503308304e0ebca0809695caa329bc17d73403843b285"),
        ("703896bf0315678929edb6d54d574932e6f24d1838bb382f8ac53a2ae51a337f", "6ed7ac9ec398a90b387b8e19db27ac1b8a85e8c373f6a30049145d91de8c8a38", "7b15394ab0658202656058d7632bacfe07e8f8985dc7952807e3894a842797c3"),
        ("c2bc0763862c27f511f3f6496b7628acbc6815c8872ce3116226935377b73de2", "04c02cacd4fae0c99ec78f49c7c9c7291444aa2f5b2b3a2c50cded264993cfcd", "fa80ab100933bd44ed799f06dfdc315e4ff624db9846cfaaea064182533bfab4"),
        ("fc96bc890f1e5a0a7b65c8a0a777c698ebec13730782aa42bf6e03c9409d8640", "dd8150a76280484ff55d77fac6f6cb82b443a74d52ed45a3955ec1cb8189d577", "b775fbbea425c94459ae3fa63faf88f4e1cdcf498cd6b6cfddafa95f2a349c04"),
        ("fb2e6cb98a6de6aad20ba70fb68bfc9731fe5c9d315ecf87a9b915ea7baf6e33", "e1a2eec3f6761a08e0204150ff0405b24277a789c1264d40aa9689e222b3e12f", "03c73c21013b21c0590212857b8275bf8dc1b44fa46285e75ff77e57f125d554"),
        ("28b248a8ad99cb5bcb9d3d49a984678b228f2ba462133892b18486a6f6064ff4", "092724e4c404ee297d1b5fdb085f1aed13eec167e71ff5900b7f0b094e914672", "eb1d73172d3082f253889fcc90e4ed88b107a837b81c6b556418fa0648062cb8"),
        ("3e200b12861cd0a3baef070700031706acc95d68f00956408ed839a743dd0995", "0d37d52dd13d9be4da60dacf3b442bddfd049c7c3963e4a38b56daf463d7647f", "bdc5fa6286e4aab46dea66c088e8ca6c06ebd04986b4f4ffbd14d58f2ac4f6de"),
        ("eb9161cbfe01443483b65dfc028f0d9f92892113c746db98aa0145978257712b", "0bcdf6d0a5c280a248fcadc2159a176067588738f6796ee0f1c2827e044e287d", "01eadda8e3fda6947f92e8cf3a613b2e3b638a89b9a0daf4e1a3b3978e8bdc5c"),
        ("48ce37b0ad84603b3f003d90fcd61454351fb6d2d2fb84a9db2be24eec3406e1", "82aa8c5daab12ee839ce12ff69fd3be1b3c33eca83f2d772be73a472d6b13d65", "f1ef4f7510fb45e8b2c4c6d8faeb0a710f1df1d74094736f1818a254e44ae6aa"),
        ("1b500a3586f9331a74925659cf1b344dabf9c686924e4e8ad058941619b50dc3", "780c395eac92c6953080a251875329fed67b0e017c3981d9a9de73f4ea29a486", "3599bb449b86246a15ee63ef34ea4499cda6df9540f5a0fd616d1660bf1f89e8"),
        ("aaf74693fc427c38d4fcfa1ec6a3215de40f156f329bfcf7adf55da75d662158", "d0d39d2286c65b8e5df88326f71c636f40c7ce9084236731048db7fe5f78045b", "b0ea3f32b858e549bb02040a6d85dcc0c9427871f18bed89fa05d7309891e31c"),
        ("b5635476135a236aa50b2fc8615c7a5c29cfcb1f315126b047b11b6ba81fd86e", "e498ae409f8ddc6329206d272e90e37801b2c6536839e111d1c1aa3080c9dfaa", "d2aa873759fd0af8c4899af1a6b9e2b4a56184804ea38246c79c682a0b4a34c1"),
        ("35927dfd6567cc86c5f01d6226a96a59d8e44b43bc53475210eae2e7e478abab", "9e56821e011eed89b627c3b39946fc91d526eaf0d95cb32b302a2a50573bee50", "008f366b560630fba8af3bba739042bca029fa5234f96f4f609341cb733da1ec"),
        ("47d452516813090f26c20b48efcb25aa73b34084c04a937d155f056aaf3c4677", "00e07033e5c286abea25decf6892cc882f56e3e4f546d3461408bf78716618ec", "6f589125334f478d3d351061f42fc2f2f9b80c8dc0ba4d5edbe51917cd4db200"),
        ("3bd3e3e959ec68bb5200f77afcae2c233e95d2eaeb8ee1495f127b93765c5164", "5d9344e6378f35f74d2f3609a1299c9a1ee2c4840c929b8abf8bed484c64310a", "570bc6cde638c2e7166bf6cf4d4cc3c0c633daacecb2e4c9b4457cdf6e016af0"),
    ]
    for sample_index in range(20):
        stratum = V2_PLANE_STRATA[sample_index // 4]
        artifact = make_v2_smoke_global_reference_centre_render(
            context, "development", "0x415154564f320001", sample_index, stratum
        )
        replayed = replay_v2_smoke_global_reference_centre_render(artifact, context)
        verify_v2_smoke_global_reference_centre_render(artifact, context)
        assert all(
            np.array_equal(artifact["raster"][name], replayed["raster"][name])
            for name in ("scalar", "annotation", "brain_mask")
        )
        identities.append(artifact["v2_plane_realization_id"])
        observed.append(
            (
                artifact["sampling"]["plane_sampler_receipt_sha256"],
                artifact["geometry"]["global_reference_grid_id"],
                artifact["raster"]["combined_sha256"],
            )
        )
    assert len(set(identities)) == 20
    assert observed == expected


@pytest.mark.parametrize("target", ["geometry", "raster", "source"])
def test_coherent_receipt_and_array_tampering_is_rejected(prepared, target):
    _, _, _, context = prepared
    artifact = make_v2_smoke_global_reference_centre_render(
        context, "development", "0x415154564f320001", 0, "near_AP"
    )
    tampered = copy.deepcopy(artifact)
    if target == "geometry":
        tampered["geometry"]["physical_ouv_ap_dv_ml_um"][0] += 1.0
    elif target == "raster":
        tampered["raster"]["scalar"][0, 0] += 1.0
    else:
        tampered["generator"]["resolved_config"]["source_sha256"][
            "arbitrary_plane_geometry.py"
        ] = "0" * 64
    with pytest.raises(ValueError, match="receipt|arrays"):
        verify_v2_smoke_global_reference_centre_render(tampered, context)


def test_context_change_cannot_replay_an_existing_plane(prepared):
    scalar, annotation, support, context = prepared
    artifact = make_v2_smoke_global_reference_centre_render(
        context, "development", "0x415154564f320001", 0, "near_AP"
    )
    other = prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="6" * 64,
    )
    with pytest.raises(ValueError, match="replay receipt"):
        verify_v2_smoke_global_reference_centre_render(artifact, other)


@pytest.mark.parametrize("mutation", ["numpy-scalar", "numpy-annotation", "torch"])
def test_context_tensor_mutation_is_detected_even_through_numpy_aliases(mutation):
    scalar, annotation, support = _volumes()
    context = prepare_arbitrary_plane_acquisition_context_v2(
        scalar,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
    )
    parent = context["opaque_v1_context"]
    if mutation == "numpy-scalar":
        parent["scalar_tensor"].numpy()[0, 0, 0] += 1
    elif mutation == "numpy-annotation":
        parent["annotation_tensor"].numpy()[0, 0, 0] = 1
    else:
        with torch.no_grad():
            parent["scalar_tensor"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="tensors changed"):
        make_v2_smoke_global_reference_centre_render(
            context, "development", "0x415154564f320001", 0, "near_AP"
        )


def test_context_receipt_and_support_are_recursively_immutable(prepared):
    _, _, _, context = prepared
    with pytest.raises(TypeError):
        context["receipt"]["support_index_sha256"] = "0" * 64
    with pytest.raises(TypeError):
        context["opaque_v1_context"]["support_index"]["components"][0][
            "anchor_index"
        ] = (0, 0, 0)
    with pytest.raises(ValueError):
        context["opaque_v1_context"]["support_index"]["component_hull_indices"][0][0, 0] = 0


@pytest.mark.parametrize("array_name,dtype", [("scalar", np.float64), ("annotation", np.float64), ("brain_mask", np.uint8)])
def test_dtype_only_raster_tampering_is_rejected(prepared, array_name, dtype):
    _, _, _, context = prepared
    artifact = make_v2_smoke_global_reference_centre_render(
        context, "development", "0x415154564f320001", 0, "near_AP"
    )
    tampered = copy.deepcopy(artifact)
    tampered["raster"][array_name] = tampered["raster"][array_name].astype(dtype)
    with pytest.raises(ValueError, match="dtype|shape|support"):
        verify_v2_smoke_global_reference_centre_render(tampered, context)


def test_coherently_re_receipted_raster_tamper_fails_replay(prepared):
    _, _, _, context = prepared
    artifact = make_v2_smoke_global_reference_centre_render(
        context, "development", "0x415154564f320001", 0, "near_AP"
    )
    tampered = copy.deepcopy(artifact)
    tampered["raster"]["scalar"][0, 0] += 1
    tampered["raster"].update(
        acquisition._v2_raster_metadata(
            tampered["raster"]["scalar"],
            tampered["raster"]["annotation"],
            tampered["raster"]["brain_mask"],
        )
    )
    tampered["receipt_sha256"] = acquisition._payload_sha256(
        v2_centre_render_receipt(tampered)
    )
    with pytest.raises(ValueError, match="replay receipt"):
        verify_v2_smoke_global_reference_centre_render(tampered, context)


def test_smoke_root_shape_and_full_assignment_are_frozen(prepared):
    _, _, support, context = prepared
    with pytest.raises(ValueError, match="smoke"):
        sample_v2_smoke_plane_pose(support, "development", "0x0000000000000001", 0, "near_AP")
    with pytest.raises(ValueError, match="256x256"):
        global_reference_support_geometry(support, (64, 64))
    with pytest.raises(ValueError, match="parent shape"):
        make_v2_smoke_global_reference_centre_render(
            context,
            "development",
            "0x415154564f320001",
            0,
            "near_AP",
            parent_shape_h_w=(64, 64),
        )
    assert len(acquisition.V2_SMOKE_ASSIGNMENTS) == 20
    assert [acquisition._smoke_assignment(index)["effective_optical_support_um"] for index in (0, 5, 10)] == [
        0.0,
        0.0,
        0.0,
    ]


@pytest.mark.parametrize("location", ["top", "raster"])
def test_unauthenticated_extra_fields_or_arrays_are_rejected(prepared, location):
    _, _, _, context = prepared
    artifact = make_v2_smoke_global_reference_centre_render(
        context, "development", "0x415154564f320001", 0, "near_AP"
    )
    if location == "top":
        artifact["unexpected"] = "unbound"
    else:
        artifact["raster"]["unexpected_array"] = artifact["raster"]["scalar"].copy()
    with pytest.raises(ValueError, match="extra fields"):
        verify_v2_smoke_global_reference_centre_render(artifact, context)


def test_plane_sampling_and_identity_do_not_contain_downstream_smoke_nuisance(prepared):
    _, _, _, context = prepared
    artifact = make_v2_smoke_global_reference_centre_render(
        context, "development", "0x415154564f320001", 0, "near_AP"
    )
    assert "fixed_smoke_assignment" not in artifact["sampling"]
    assert not {
        "window_plan_severity",
        "reflection",
        "render_mode",
        "nominal_cut_thickness_um",
        "effective_optical_support_um",
    } & artifact["sampling"].keys()
