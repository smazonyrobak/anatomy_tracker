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


# Refrozen after dense supervision was masked to centre support instead of background.
GOLDEN_SLAB_IDENTITIES = (
    ("fd2f7b960f2f3d4d77e43349499911e48ecf3fe135583fc1e00cdd19717d6a4e", "d2f9b9479a3450eaf3bc37af1515d3ee34cab95055003e51ae1e6c6893f0120d", "f9d1979d1b6fbecb071ec4f90d8c7ada6b61bffda54ce07f7f1904fba18a0096", "87c2be961b4040310fe91fa0f1efc227dd60399c69f3b477d10811fd7db7e2e8"),
    ("d53edd4019fda77bfdf66f602c4ffd4beb7a6e2e8f736479e1419161667510f6", "1cce2e6911aba2ead66795a9d7105fdfc4f2a1288fd2189491275cae23e67120", "77782be7361a39bd36669614bbf25f6d7e5404cb04e3019914bb491062ece931", "633d768052cb32742b1da64d675d83325fc6e3aa2637f5115fe66e06b8dc235c"),
    ("1cef888563b1bb269e4b9d3350c5e03e3a7eebe21b62e2b277fba2fc54db0324", "03fe1b2121b6e9a6fa7d85a311ce8a2f2ce0f87e953a68d2d566678cf4771d18", "13eb28277bc0491c88dcc481cda149b3f31d3d61cfda3eedf9ade78c99cdccca", "b394c91c70ecfc2719db11f0a44233bb1de1dcf06f404a8a912041122bc8a2f4"),
    ("72d08b3c6cd3ddd0a8776ab649f77258298ef08ba8d14a2a16561c98656e59f4", "61745cd485cfce2c750e9ab94d5aa234cb5b8fbb8e38613cd21467c01bfcd6c3", "6edb9b7023e5fd7aa853639cf07985273c9056d954c2db9fc17cf32c969767f9", "8c6ee9cfa7565b49d65620216b148f057290c81712f8f38ed63adc21f35920c8"),
    ("fa666ac74dda41e9088b4e56ea41638b0938748786efc51a5294e7fb14183f90", "0f16fe56b0dd24345dcd5298bce809bd74bf8b7a0d9edcbf57bc7965cd3d08ae", "1a385d5e6eff41d8fe3725666654974f4e95ec114a6721b0c23da4bb0fc37011", "d18a2e18e4af4439190cf625170666acadee9509861fca230b5e7dcaa2c0b56d"),
    ("9dfc78da3faf319d4cc3cc0b20fc1d018e34f92a2aae15f3871f6effc6020403", "df3a1c889abd28a958d0e2b443b980997acc96804c8f59f24897ba3035271019", "939be644383e64cc75283b3c1694d484962df9e97907592a9c0b3ba1e18bb502", "151a5e82046a772c6f7807fa97f7745a978bb61a06636ccf9bedd8509dbc6051"),
    ("adae56bdb1a48a84219376ec32fb66b6f05484be74f9db21cdac49f7a10198f6", "0ae44b70b1219498a6c8cf23b223e7d1ab6da5e87e22234be77d19f19204a5a2", "bfe0b0a5dbd381832d5febbc853565faf2df5131834317f5ca4894da3f28798c", "3b2880e83922650b62d64483fc3c64d93307d775e6321fea157e1410a45406d6"),
    ("6797a2a1e4571c8f196697aa58cfe88b87f84b625cef37be3bf55db593075ef1", "584c83fa3f4c82a50bef7b7e9be6018e4ae9afb0ac69810452f034a75150c75c", "fc2d6a4763dd192a6fcc9e2f004c784c5372bd0ae685aa1dc5ab72c01574a000", "7090906be01995e5c90844d20df81f0a582350b5d0ce92cbb5fbeaaa80c30406"),
    ("3af2c400ed56613a9a0d8798d098cd2784d7ce1af0eb285515b8c5f45df20ede", "23dd66f654bbae53f9e471da047206787b65877e43f40281adefd8ea0507c449", "eaf5d3183d643a310e71b63c84937571fd3886d21faadaf7b613481880dcee19", "7e90f13d8e5ba7f38b0194a7d113578a93275d25c844418b6ea905b7d463f272"),
    ("97cf62ef3dca1aa64e33670fe5e908b3067484f4fb7106a56bfb2da48ce2e3e0", "448c4b00350fe66b1cf6fa92624b8f9beb53f6ebb2e3ea17c57c12622bee8edc", "49e83830b1f83015f176297f36c866c4c753e856c0fe53c6a76741e154edd626", "46c668a75a5a7d8eddb35c30bf0524f813b35a085e59f2ae5d0c14b36179c96b"),
    ("e05d8e1bff78fa90f81d76cc88f50381ad20b1a038aaa490bbb751e7485c567a", "94184c0f09bf8120297a976634b3126a8967b009c006257a1e3a4eab114f1577", "e1b3e6ac7ca24fa0ffdd3378d1d23f9260a31ccfc2928068d83b498f6367ca11", "6dd4cbce95ac62e50d655fb002a2e510cfbf8fd0d03ac4d002bfeeb0fa52e16b"),
    ("2118c0369bd27e9410144db1d636179364530670b98166f16143776526179722", "7095b75a2e9c2c19a79d6c9be8a64a711be39f571b9d07eeff5588cb2b065e39", "edc7576f91fd1a935d176c9358f220329f5ff4e712e6b8819071f75b1d2cc4f8", "227431c477d4b64fe64a2b4df8387644a9635de603ea6bd9c2ab6fb91fb6baa1"),
    ("397d8b1d671d78599a0306a7ffe0707bcfd9d5043b69ed47871b12b84bfc1fd9", "29591c3b3a062309b95c30abbd38f10d1aae692f071a63f20c0b19dc71b2c72a", "22341f0c888ed9da7b45fbbe7d8c7f01c45e654c1550d50e17ed6396a170600e", "e99e648ce723170a0b3f1e0f78ca591da252b8866fe7a496d1cc3ceec87e3fe2"),
    ("51bcc5ecd4631347fbb26ab685a01307db3e1314fef88245a2bfd94a54f8e691", "90b3335b5c9223d4436581f670ce7853f1695f9066552dcfb1e2f546994eb121", "229caf827dc4e5fabb31eeb4c0ebfbfa2e1417adae7db4ecf074caa02e2150f0", "4f841c90a13723d1274364a18e6336b53cea97252459b1fa5c5dc56f17217392"),
    ("0aaf4386a3e3cffec2abb9bc75cc795366abe42461c81d86974c0c4b8c50ff73", "4b42aeefd31f839f055fa0df919cf9d705b5950c5a8f66fce324f0af2aa3d2d8", "02590ebea318c0f4ee998f0150cd1490fe2e8ca17201dd6b2d32f818efdb1585", "47b82e5523005df1b1bad42f66ab2e968ba0ef8fb7056c2039fa2e1d369fb43f"),
    ("39542e948decfc0975602ec19ef0458e8fc6e9813e99eaf2539e4601e543c882", "131908a8af17591d8b33a02a580eed8371a36b75cd9fa72aca1168a607388d0e", "5ca40d9a91f484dc71cf3def9d428a6bd0659a37366683b3002b4322039ca39a", "8a1a2222ef93e50f22c50ef875b0e0c401f969abd6078816db341f8d8f11089f"),
    ("c0f5f8b798ca89dcc5df3b4bd7665bdbd1a52cbddfe40f202df32d68bb4fa2b4", "9aad80fe9d1d0702c1f4ea903af89a02e902f5d84acca85f09e692a7cca3d458", "07d43bf80d02a9e357774c633298520ee64196940dbabefe8e92f306cbd91cef", "f3d96b6a187e8ffab1240088c90b12ffef161817c9490ec69d55a154977ac82b"),
    ("23791ac3b98687fce18e8d1a366240111dd1c872de83b0618e6a90d2db3b1b2f", "e8b45936a8876c3696f9a20d16e2b041f3a8f7979ac5d24a1578f0b68be26ef4", "64528c148ee28a22d655283015dd7b121343a234dfa3f08619f987d1fb021300", "64e9d51bc21575733deb663bfed513a7962a47d3c2de683fddc8b700ea3b2cbf"),
    ("a4bb43567886d242f8a270b3a813bd3bc985f4346c5ddb5ef46a2677bb09483e", "4f204a23f0c86c695eb86637eca3bb7be124cc284c2660da92f32cbe9748850f", "81cb41a2eaa8f34433902a3acf8aab2f3d6567280407cdbe8cb24bfc3601fe3e", "d7c7678cafe4cf8b171117321a500f319dad120c0244554a6c8b6cc38790afc8"),
    ("29006c0ecc0d956763eabb2b5e3cd0b7c66b6f12b1ad5e58443f213d93f20dec", "1167216cf3e399d394154db81e1ecfb2bf34a7dbce1acc06d8a184ff3af77c18", "3d3037e0479dd5ee2f6ffa18f32b2045b6136930b873d9af272e76a87c810a17", "f9d4e0051157cfc2f529ebd521acd5616fac275fe1b8d7c81d6d05253cdbbe2a"),
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
