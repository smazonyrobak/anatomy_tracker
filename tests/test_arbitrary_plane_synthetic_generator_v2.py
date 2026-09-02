import copy
import math

import numpy as np
import pytest

import training.arbitrary_plane_synthetic_generator_v2 as slab
import training.run_arbitrary_plane_slab_qualification as slab_runner
import training.slab_refinement_gate_status_v2 as slab_gate
from training.arbitrary_plane_acquisition_v2 import (
    V2_PLANE_STRATA,
    V2_SMOKE_ASSIGNMENTS,
    make_v2_smoke_global_reference_centre_render,
    prepare_arbitrary_plane_acquisition_context_v2,
)
from training.arbitrary_plane_support import build_annotation_support_index


# Refrozen after authenticated finite-render verifier source changed; raster hashes are unchanged.
GOLDEN_SLAB_IDENTITIES = (
    ("49083b9a8f9574aba27c3d25ff4855b17c766c446e810f55a765188e0d9dcfb5", "84ddf91ee7cb4a615a01cf628b3bee842cd969621ea1c7b4b8654c2f6f4cf77d", "f9d1979d1b6fbecb071ec4f90d8c7ada6b61bffda54ce07f7f1904fba18a0096", "f647359f5ff87e57d314903b2d4ed0bacc3cfa7e556c8a887bd4d573cf981c78"),
    ("0e6cc990bcebd5e97d77ddbb0be0cf5eee111aaff23589ea5da01ff057876906", "eb509ee2c4934eb06a0f61bfb7b15877a1fe0e570ce81f911590c54d70dc61b2", "77782be7361a39bd36669614bbf25f6d7e5404cb04e3019914bb491062ece931", "2dbedad9463da9a349c750eae846809d07934437d5e03cd36961d6b40ae1181f"),
    ("4297adfc3da0126b7eb763da0377222f608426d525fbf8f23477dcd0dc571bdc", "8adb92f479571ec85f6ec2f8c77dd37c833c3143fb2ee75666efb0ba5c20a13e", "13eb28277bc0491c88dcc481cda149b3f31d3d61cfda3eedf9ade78c99cdccca", "4cedfafd01629810278c56f71f7b58adbac5bd603a5884ed75470efa5b7c74bc"),
    ("387ab4fd76df987b8b005e9d95aa25f6dbf04cbf217c3a8b22dadb0a28e5abee", "bda61a64b2fdd14d66afd4cb701bd85b4fa475c9f779cc652a33dc8f03e46e08", "6edb9b7023e5fd7aa853639cf07985273c9056d954c2db9fc17cf32c969767f9", "202b13053aee6ccf28b68995bf8db267650bf026e8e4efab595dbadc530070f6"),
    ("741ede312c26e959f326d987d038fc42a8e1127bb84d8b042b11df391b16d2dd", "a21499c59ed054f302f7df8c5c75b5fa3ec0a35d67aba226a6b0d0a03291ecfd", "1a385d5e6eff41d8fe3725666654974f4e95ec114a6721b0c23da4bb0fc37011", "6464537398e41c3a5917c561982ce9b4094888884117ac1c8fb7a7299bc1ba0a"),
    ("57eb552172a9ec47c08967d0bb399052629a4eb15e1b076b5019be2b9fa98cd3", "1f56c184b950c0a6bf5f26f0846e988c46c5ba183b1f63182a059559e56e02fd", "939be644383e64cc75283b3c1694d484962df9e97907592a9c0b3ba1e18bb502", "d3844aa3d8fcbb47fc7e61f58e3005b9a339e65577f825a64de596d91192d264"),
    ("b3b28c1d238fbd0289b36cbcff5121be1abf5f1c7c5e5462fe34d10e59696b0f", "5dac0d5b8f07de15bb95fcfa91fc76594924eb2e38564462ee45706c019cb684", "bfe0b0a5dbd381832d5febbc853565faf2df5131834317f5ca4894da3f28798c", "f2cc04f8c765e6f37bf3b329a522db11dde56d1faf39454e7f79044781ef3182"),
    ("3f359dc75ccb55262d4d697e484ac891e13de6a7ea922a665198634b262a4014", "688a6c7c7f1d4bad2eff757418680ae649e3027b2332c3ad72df34232db88f91", "fc2d6a4763dd192a6fcc9e2f004c784c5372bd0ae685aa1dc5ab72c01574a000", "e1dcfefb4f475f3eed8c1782c09bfbc9a1e8accaa9b5dd9f31f51797a8d27ced"),
    ("91aea2e2ccb34728f28b19c0dd5e71290032c6347f9547952264bead4619f196", "fbc7ba001a71d44efde28c6477cd919388382bb990ef515f03610c21445478b1", "eaf5d3183d643a310e71b63c84937571fd3886d21faadaf7b613481880dcee19", "1bb9a4addcff8c78bea88368a2609f2087384569e71534d6980ca64f2c7f0f6b"),
    ("681d4be6654802270353c516eedf601ee24cd805cc500bbb3e69fa10cf46e439", "4177138ee77934768b9f30935cdde0582630c8570b1452b164328bc8cee9c59f", "49e83830b1f83015f176297f36c866c4c753e856c0fe53c6a76741e154edd626", "3dad31d63bd4e62ad32e4e600a6f42b4b6ec5bf0f540e67b7c3780c817509dd8"),
    ("c391350e46abbb5dbe08eb826172a9842866a1518df22c1f125f099a8c249484", "05546121bb68d0b5c1e3464fda12637ee69c502241c7d85624df1f746a074a55", "e1b3e6ac7ca24fa0ffdd3378d1d23f9260a31ccfc2928068d83b498f6367ca11", "a9ccafda1198838d2558ab6a1d53c57fb30f6abf7a001a1885e5c531ec3fee2d"),
    ("b34328e2298c53b2e7e46b81d22b1265d944b69f5abe09e4056e12872cc5fc2a", "d445b93fed1e9276fe1b6ce76c3d5e6fc1a3a829377c17c26b658988f67740d0", "edc7576f91fd1a935d176c9358f220329f5ff4e712e6b8819071f75b1d2cc4f8", "41d4b97cb2509084a7df748e233e6173ac2432f8761a371a9a144bd9257c6118"),
    ("a50c8fd8e0e09a681f8ea3907f2ada51c40004d95014465c1643f4d1833a69eb", "e71488db1de7292e3f77416cf5d60b549fc830d1a4a63275be40cf6826723265", "22341f0c888ed9da7b45fbbe7d8c7f01c45e654c1550d50e17ed6396a170600e", "ee0f7a12fa54e8c0e2cedc60b60c587c68ba610289194b6b743bb524e61dcfc5"),
    ("0307785fb150512a9c5d2cebe99b51eb9674aa8134056ebd7f783ca8ef36a738", "8f849db023c365c2976852cbb23f541e0e64b879952feba56db384a272eef15a", "229caf827dc4e5fabb31eeb4c0ebfbfa2e1417adae7db4ecf074caa02e2150f0", "405990cb7e6b25e188feff8e1a0c48c94f90dc5fbdd5fcb64c64f78ff1dbca23"),
    ("7d3a01d7757de435aca325ab5fd9189730a1217fb464333234e7452a71959bf1", "265cfe1cea59f3663284d4582bba5f1c07f82497e83c968c3d2d4c9e39d433cd", "02590ebea318c0f4ee998f0150cd1490fe2e8ca17201dd6b2d32f818efdb1585", "ae7d3179e9d7eef73fb7d60d09cee5f7e8e827292ebf3fce4a1f47a86010b3b3"),
    ("a6eff2e1e71d9f32e9cda5d0f0bafbeb58623ba5b5505713b59646f66a3a7705", "a135f8554c04887d8194101376be0f3c2e56226a52365abba593921a418e8723", "5ca40d9a91f484dc71cf3def9d428a6bd0659a37366683b3002b4322039ca39a", "6e36b250287fb35f7168f6ee19ae8c877fd36e6698bf4eb1a9fb18414aaf83e7"),
    ("5397556e106229966092af5d591207bf0c95d70421dbd19beb4a4096195a431a", "c744bcaead1aa0db3d5ccb67ca2a6caa385323657e6a150ffb6fa6303d22d570", "07d43bf80d02a9e357774c633298520ee64196940dbabefe8e92f306cbd91cef", "2b46b3e43bea56259a94deccb0be2aa4c3295a7f5e80dafd0f165bcb5c73b362"),
    ("ce4230bb2c7a99664e1a18008cbc08616e5d3950953a7686d7e79e00c1f68705", "34fce6f554702011faeecb41c3c312f6e887984218d8e2968b69d56edf367b43", "64528c148ee28a22d655283015dd7b121343a234dfa3f08619f987d1fb021300", "97b1738b5fce34b532cf1e0e291796b65d8cc765f6e369f27e9fb1bd05afe521"),
    ("be82090e93c842439f04b1da7a746c2327b6ad4869cda63f93825e717e044270", "0be0bded9cda68f62a8c55f3d13974ca8537f01e657868a2d2cf234bf292d021", "81cb41a2eaa8f34433902a3acf8aab2f3d6567280407cdbe8cb24bfc3601fe3e", "364f06e5f39a927cf38f678c2f90a83277556d154cb44fd7c06e7545e0a3bf54"),
    ("04c734bb36d0dd06fe087fa5b33a2a4f87f679313d1f4c8ed2ee838741342101", "ebd0cbf9e88bca898f5e4013d92465570fd917599acfae38018d8d92dcb9a56a", "3d3037e0479dd5ee2f6ffa18f32b2045b6136930b873d9af272e76a87c810a17", "1921545f36dbd47cb9e6b1d620166ecbdec087cb845f1270f90a019c2a5d7819"),
)


def _prepared_context():
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
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return prepare_arbitrary_plane_acquisition_context_v2(
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


def _make(context, sample_index):
    return slab.make_v2_smoke_global_reference_slab_render(
        context,
        "development",
        "0x415154564f320001",
        sample_index,
        V2_PLANE_STRATA[sample_index // 4],
        animal_id="animal-fixture",
        animal_index=7,
        specimen_id="specimen-fixture",
        experiment_id="experiment-fixture",
    )


def test_all_fixed_smoke_kernels_have_the_predeclared_schedule():
    counts = [1, 3, 5, 7, 9, 1, 5, 7, 9, 5, 1, 5, 7, 7, 9, 5, 7, 9, 5, 7]
    for index, row in enumerate(V2_SMOKE_ASSIGNMENTS):
        _, _, _, mode, thickness, _, declared_support = row
        kernel = slab.finite_boxcar_kernel(mode, thickness)
        offsets = np.asarray(kernel["optical_kernel_offsets_um"])
        masses = np.asarray(kernel["optical_kernel_integer_masses"])
        weights = np.asarray(kernel["optical_kernel_weights"])
        assert kernel["axial_sample_count"] == counts[index]
        assert np.array_equal(offsets, -offsets[::-1])
        assert np.array_equal(masses, masses[::-1])
        assert math.fsum(weights.tolist()) == 1.0
        assert offsets[len(offsets) // 2] == 0.0
        assert kernel["material_thickness_um"] == thickness
        if mode == "centre_plane_ablation":
            assert offsets.tolist() == [0.0]
            assert weights.tolist() == [1.0]
            assert kernel["effective_optical_support_um"] == declared_support == 0.0
        else:
            assert np.max(np.diff(offsets)) <= 12.5
            assert kernel["effective_optical_support_um"] == thickness


def test_qualification_runner_refuses_a_different_pynrrd_decoder(monkeypatch):
    monkeypatch.setattr(slab_runner.nrrd, "__version__", "different")
    with pytest.raises(ValueError, match="frozen decoder version"):
        slab_runner.main()


def test_exact_categorical_ties_include_zero_and_supervision_threshold_is_frozen():
    offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    base = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    slope = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
    scalar = np.stack([base + offset * slope for offset in offsets])[:, None, :]
    labels = np.asarray([[[0, 9, 7]], [[7, 7, 7]], [[0, 9, 9]]], dtype=np.int64)
    reduced = slab.reduce_v2_slab_samples(
        scalar, labels, np.asarray([1, 2, 1], dtype=np.int64), 1
    )
    assert np.array_equal(reduced["scalar"], base[None])
    assert np.array_equal(reduced["slab_modal_annotation"], [[0, 7, 7]])
    assert np.array_equal(reduced["slab_label_purity"], [[0.5, 0.5, 0.75]])
    assert np.array_equal(reduced["centre_label_support_weight"], [[0.5, 0.5, 0.75]])
    supervision = reduced["slab_supervision_weight_or_abstention"]
    assert np.array_equal(supervision["abstention_mask"], [[True, True, False]])
    assert np.allclose(supervision["dense_correspondence_weight"], [[0, 0, 5 / 6]], atol=1e-7)


def test_all_background_centre_pixels_are_never_dense_correspondence_targets():
    reduced = slab.reduce_v2_slab_samples(
        np.zeros((3, 2, 2), dtype=np.float32),
        np.zeros((3, 2, 2), dtype=np.int64),
        np.asarray([1, 2, 1], dtype=np.int64),
        1,
    )
    supervision = reduced["slab_supervision_weight_or_abstention"]
    assert not reduced["centre_plane_support_mask"].any()
    assert np.array_equal(reduced["centre_label_support_weight"], np.ones((2, 2), np.float32))
    assert np.array_equal(
        supervision["dense_correspondence_weight"], np.zeros((2, 2), np.float32)
    )
    assert supervision["abstention_mask"].all()
    slab._slab_raster_metadata(reduced)


@pytest.mark.parametrize("sample_index", [0, 5, 10])
def test_ablation_is_byte_identical_to_the_single_centre_render(prepared, sample_index):
    stratum = V2_PLANE_STRATA[sample_index // 4]
    centre = make_v2_smoke_global_reference_centre_render(
        prepared,
        "development",
        "0x415154564f320001",
        sample_index,
        stratum,
        animal_id="animal-fixture",
        animal_index=7,
        specimen_id="specimen-fixture",
        experiment_id="experiment-fixture",
    )
    artifact = _make(prepared, sample_index)
    assert artifact["v2_plane_realization_id"] == centre["v2_plane_realization_id"]
    assert artifact["centre_plane_render_id"] == centre["centre_plane_render_id"]
    assert artifact["raster"]["scalar"].tobytes() == centre["raster"]["scalar"].tobytes()
    assert np.array_equal(
        artifact["raster"]["centre_plane_annotation"], centre["raster"]["annotation"]
    )
    assert np.array_equal(
        artifact["raster"]["slab_brain_occupancy"], centre["raster"]["brain_mask"]
    )
    supervision = artifact["raster"]["slab_supervision_weight_or_abstention"]
    assert np.array_equal(
        supervision["dense_correspondence_weight"],
        centre["raster"]["brain_mask"].astype(np.float32),
    )
    assert np.array_equal(supervision["abstention_mask"], ~centre["raster"]["brain_mask"])
    assert artifact["diagnostics"]["nonzero_offset_render_count"] == 0
    slab.verify_v2_smoke_global_reference_slab_render(artifact, prepared)


def test_ablation_bypasses_numeric_reduction_and_preserves_signed_zero(prepared, monkeypatch):
    precursor = make_v2_smoke_global_reference_centre_render(
        prepared,
        "development",
        "0x415154564f320001",
        0,
        "near_AP",
        animal_id="animal-fixture",
        animal_index=7,
        specimen_id="specimen-fixture",
        experiment_id="experiment-fixture",
    )
    precursor["raster"]["scalar"][0, 0] = np.float32(-0.0)
    monkeypatch.setattr(
        slab,
        "reduce_v2_slab_samples",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("numeric reduction used")),
    )
    reduced, _, _ = slab._render_offset_samples(
        prepared,
        precursor,
        slab.finite_boxcar_kernel("centre_plane_ablation", 50.0),
    )
    assert reduced["scalar"].tobytes() == precursor["raster"]["scalar"].tobytes()


def test_finite_slab_moves_only_along_the_physical_normal(prepared):
    artifact = _make(prepared, 1)
    normal = np.asarray(artifact["geometry"]["normal_rp2_ap_dv_ml"])
    for receipt in artifact["offset_render_receipts"]:
        expected = receipt["offset_um"] * normal
        assert np.array_equal(receipt["design_physical_displacement_ap_dv_ml_um"], expected)
        assert np.allclose(
            receipt["effective_physical_displacement_ap_dv_ml_um"], expected, atol=4e-5
        )
        assert max(
            receipt["centre_displacement_axial_error_um"],
            receipt["centre_displacement_tangential_error_um"],
            receipt["coordinate_raster_displacement_axial_error_um"],
            receipt["coordinate_raster_displacement_tangential_error_um"],
        ) <= 0.01
    assert [item["reused_authenticated_centre_plane_render"] for item in artifact[
        "offset_render_receipts"
    ]] == [False, True, False]


def test_oblique_physical_normal_linear_ramp_has_the_symmetric_slab_expectation(prepared):
    centre = make_v2_smoke_global_reference_centre_render(
        prepared,
        "development",
        "0x415154564f320001",
        12,
        "general_oblique",
        animal_id="animal-fixture",
        animal_index=7,
        specimen_id="specimen-fixture",
        experiment_id="experiment-fixture",
    )
    artifact = _make(prepared, 12)
    assert np.max(np.abs(artifact["geometry"]["normal_rp2_ap_dv_ml"])) < 0.9
    observed = artifact["raster"]["slab_brain_occupancy"] == 1.0
    assert observed.any()
    assert np.max(
        np.abs(artifact["raster"]["scalar"][observed] - centre["raster"]["scalar"][observed])
    ) < 5e-5


def test_all_twenty_slab_renders_have_golden_ids_and_replay_byte_exactly(prepared):
    for sample_index in range(20):
        artifact = _make(prepared, sample_index)
        replayed = slab.replay_v2_smoke_global_reference_slab_render(artifact, prepared)
        slab.verify_v2_smoke_global_reference_slab_render(artifact, prepared)
        assert (
            artifact["slab_recipe_id"],
            artifact["slab_render_id"],
            artifact["raster"]["combined_sha256"],
            artifact["receipt_sha256"],
        ) == GOLDEN_SLAB_IDENTITIES[sample_index]
        assert artifact["slab_recipe"][
            "implementation_source_sha256_canonicalization"
        ] == slab.acquisition.V2_SOURCE_SHA256_CANONICALIZATION
        assert slab.v2_slab_render_receipt(artifact) == slab.v2_slab_render_receipt(replayed)
        for name, array in slab._slab_arrays(artifact["raster"]).items():
            assert np.array_equal(array, slab._slab_arrays(replayed["raster"])[name])
        diagnostics = artifact["diagnostics"]
        assert max(
            diagnostics["maximum_centre_axial_displacement_error_um"],
            diagnostics["maximum_centre_tangential_displacement_error_um"],
            diagnostics["maximum_coordinate_raster_axial_displacement_error_um"],
            diagnostics["maximum_coordinate_raster_tangential_displacement_error_um"],
        ) <= diagnostics["physical_displacement_tolerance_um"] == 0.01
        assert not {
            "acquisition_window_realization_id",
            "reflection_transform_id",
            "reflection_realization_id",
            "v2_acquisition_realization_id",
            "synthetic_realization_id",
        } & artifact.keys()


@pytest.mark.parametrize("target", ["array", "recipe", "extra"])
def test_coherently_re_receipted_tampering_or_extra_fields_are_rejected(prepared, target):
    tampered = copy.deepcopy(_make(prepared, 1))
    if target == "array":
        tampered["raster"]["scalar"][0, 0] += np.float32(1.0)
        tampered["raster"].update(slab._slab_raster_metadata(tampered["raster"]))
    elif target == "recipe":
        tampered["slab_recipe"]["nominal_cut_thickness_um"] = 26.0
        tampered["slab_recipe_id"] = slab.acquisition._payload_sha256(tampered["slab_recipe"])
    else:
        tampered["unexpected"] = "unbound"
    if target != "extra":
        tampered["slab_render_id"] = slab.acquisition._payload_sha256(
            slab._slab_render_identity_payload(tampered)
        )
        tampered["receipt_sha256"] = slab.acquisition._payload_sha256(
            slab.v2_slab_render_receipt(tampered)
        )
    with pytest.raises(ValueError):
        slab.verify_v2_smoke_global_reference_slab_render(tampered, prepared)


def test_refinement_diagnostic_binds_raw_refined_receipts(prepared):
    report = slab.compare_v2_slab_axial_refinement(_make(prepared, 1), prepared)
    assert report["coarse_axial_step_um_max"] == 12.5
    assert report["refined_axial_step_um_max"] == 6.25
    assert report["union_nonzero_support_pixel_count"] > 0
    assert set(report["metrics"]) == {
        "normalized_scalar_mae",
        "normalized_scalar_absolute_error_p99",
        "support_mass_mae",
        "support_mass_absolute_error_p99",
        "slab_label_purity_mae",
        "slab_label_purity_absolute_error_p99",
        "centre_label_support_weight_mae",
        "centre_label_support_weight_absolute_error_p99",
        "dense_correspondence_weight_mae",
        "dense_correspondence_weight_absolute_error_p99",
        "slab_modal_annotation_disagreement_fraction",
        "slab_observable_support_mask_disagreement_fraction",
        "dense_correspondence_abstention_disagreement_fraction",
    }
    assert len(report["refined_offset_render_receipts"]) > 3
    assert report["case_receipt_sha256"] == slab.acquisition._payload_sha256(
        {key: value for key, value in report.items() if key != "case_receipt_sha256"}
    )
    assert isinstance(report["passed"], bool)


@pytest.fixture(scope="module")
def qualification_report(prepared):
    return slab.evaluate_v2_slab_refinement_smoke(prepared)


def _rereceipt_qualification(report):
    for case in report["cases"]:
        case["case_receipt_sha256"] = slab.acquisition._payload_sha256(
            {key: value for key, value in case.items() if key != "case_receipt_sha256"}
        )
    report["qualification_receipt_sha256"] = slab.acquisition._payload_sha256(
        {key: value for key, value in report.items() if key != "qualification_receipt_sha256"}
    )


def test_legacy_universal_gate_is_authenticated_but_explicitly_rejected(
    qualification_report, prepared
):
    assert qualification_report["subject_deformed_qualification_status"] == (
        "pending; not evaluated by this report"
    )
    assert qualification_report["source_sha256_canonicalization"] == (
        slab.acquisition.V2_SOURCE_SHA256_CANONICALIZATION
    )
    assert qualification_report["runner_source_sha256"] == (
        slab.acquisition._normalized_text_sha256(
            slab._SOURCE_ROOT / "run_arbitrary_plane_slab_qualification.py"
        )
    )
    assert qualification_report["all_cases_passed"] is False
    decision = slab_gate.assess_rejected_legacy_gate(qualification_report, prepared)
    assert decision["gate_contract"]["decision"] == "reject_legacy_universal_gate"
    assert decision["gate_contract"]["qualification_eligible"] is False
    assert decision["gate_contract"]["legacy_comparison"] == {
        "axial_steps_um_max": [12.5, 6.25],
        "mean_absolute_error_max": 0.02,
        "absolute_error_p99_max": 0.10,
        "thresholds_changed": False,
    }
    assert decision["gate_contract"]["replacement_experiment"][
        "axial_steps_um_max"
    ] == [12.5, 6.25, 3.125, 1.5625]
    assert decision["legacy_numerical_outcome"]["failing_case_indices"]
    assert decision["decision_receipt_sha256"] == slab.acquisition._payload_sha256(
        {
            key: value
            for key, value in decision.items()
            if key != "decision_receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="did not pass all 17 finite cases"):
        slab.verify_v2_slab_refinement_qualification(qualification_report, prepared)


@pytest.mark.parametrize(
    "tamper", ["fabricated", "swapped", "metric", "refined_receipt", "failed"]
)
def test_coherently_rereceipted_qualification_tampering_is_rejected(
    qualification_report, prepared, tamper, monkeypatch
):
    monkeypatch.setattr(
        slab_gate.slab,
        "evaluate_v2_slab_refinement_smoke",
        lambda context: qualification_report,
    )
    report = copy.deepcopy(qualification_report)
    if tamper == "fabricated":
        report["cases"] = [
            {"sample_index": index, "passed": True} for index in report["finite_case_indices"]
        ]
    elif tamper == "swapped":
        report["cases"][0], report["cases"][1] = report["cases"][1], report["cases"][0]
    elif tamper == "metric":
        report["cases"][0]["metrics"]["normalized_scalar_mae"] += 1e-9
    elif tamper == "refined_receipt":
        report["cases"][0]["refined_offset_render_receipts"][0]["offset_um"] += 1e-9
    else:
        report["cases"][0]["metrics"]["normalized_scalar_mae"] = 1.0
        report["cases"][0]["passed"] = False
        report["all_cases_passed"] = False
    _rereceipt_qualification(report)
    with pytest.raises(ValueError):
        slab_gate.assess_rejected_legacy_gate(report, prepared)


def test_qualification_is_bound_to_the_prepared_context(qualification_report):
    other = _prepared_context()
    other["opaque_v1_context"]["scalar_tensor"].numpy()[0, 0, 0] += 1.0
    with pytest.raises(ValueError):
        slab_gate.assess_rejected_legacy_gate(qualification_report, other)
