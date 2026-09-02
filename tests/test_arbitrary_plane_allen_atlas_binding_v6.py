from __future__ import annotations

import ast
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import training.arbitrary_plane_allen_atlas_binding_v6 as allen_v6


@pytest.fixture(scope="module")
def bound_96():
    bundle = allen_v6.prepare_bound_allen_atlas_v6(raster_shape_h_w=(96, 96))
    yield bundle
    del bundle
    gc.collect()


def test_source_imports_only_deterministic_atlas_geometry_dependencies():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "training"
        / "arbitrary_plane_allen_atlas_binding_v6.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert {
        "training.arbitrary_plane_support",
        "training.arbitrary_plane_catalogue_v3",
        "training.arbitrary_plane_catalogue_binding_v3",
    }.issubset(imports)
    assert not any(
        token in name
        for name in imports
        for token in (
            "model",
            "checkpoint",
            "feature",
            "candidate_bank",
            "training_bank",
            "prediction",
            "pseudolabel",
            "inference",
        )
    )


def test_clean_import_transitive_training_boundary_is_explicit():
    root = Path(__file__).resolve().parents[1]
    script = (
        "import json,sys; "
        "import training.arbitrary_plane_allen_atlas_binding_v6; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.startswith('training.arbitrary_plane'))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(completed.stdout))
    expected = {
        "training." + Path(path).stem
        for path in allen_v6.DETERMINISTIC_SOURCE_FILES_V6
    }
    assert loaded == expected
    assert not any(
        token in name
        for name in loaded
        for token in (
            "model",
            "checkpoint",
            "feature",
            "candidate_bank",
            "training_bank",
            "prediction",
            "pseudolabel",
            "inference",
        )
    )


def test_unsupported_raster_is_rejected_before_raw_io(monkeypatch):
    monkeypatch.setattr(
        allen_v6,
        "_decode_and_preprocess_allen_v6",
        lambda: pytest.fail("unsupported profiles must fail before NRRD I/O"),
    )
    with pytest.raises(ValueError, match="96x96 or 160x160"):
        allen_v6.prepare_bound_allen_atlas_v6(raster_shape_h_w=(128, 128))


def test_pinned_96_bundle_has_exact_raw_decode_support_atlas_and_catalogue(bound_96):
    assert allen_v6.verify_bound_allen_atlas_v6(bound_96)
    resolved = allen_v6.resolve_bound_allen_atlas_v6(bound_96)
    atlas = resolved["atlas_volume_float32"]
    support = resolved["support_index"]
    catalogue = resolved["catalogue"]
    binding = resolved["binding"]

    assert resolved["schema_version"] == allen_v6.ALLEN_ATLAS_BUNDLE_V6_SCHEMA
    assert atlas.shape == (2, 528, 320, 456)
    assert atlas.dtype == np.float32
    assert atlas.flags.c_contiguous
    assert binding["decoder"]["version"] == "1.1.3"
    assert binding["decoder"]["index_order"] == "F"
    assert binding["raw_sources"]["template"]["raw_sha256"] == (
        allen_v6.TEMPLATE_RAW_SHA256_V6
    )
    assert binding["raw_sources"]["annotation"]["raw_sha256"] == (
        allen_v6.ANNOTATION_RAW_SHA256_V6
    )
    assert allen_v6._plain(binding["decoded_atlas_receipt"]) == (
        allen_v6.ATLAS_FLOAT32_RECEIPT_V6
    )
    assert binding["preprocessing"]["intensity_channel"]["observed_q01"] == 9.0
    assert binding["preprocessing"]["intensity_channel"]["observed_q99"] == 273.0
    assert binding["preprocessing"]["channel_order"] == (
        "intensity",
        "annotation-support",
    )
    assert support["support_index_sha256"] == allen_v6.SUPPORT_INDEX_SHA256_V6
    assert support["foreground_voxel_count"] == 32_387_385
    assert catalogue["counts"]["cell_count"] == 98_304
    assert catalogue["receipt_sha256"] == (
        allen_v6.CATALOGUE_PROFILE_V6[(96, 96)]["receipt_sha256"]
    )
    assert binding["receipt_sha256"] == (
        allen_v6.ALLEN_ATLAS_BINDING_RECEIPT_BY_RASTER_V6[(96, 96)]
    )
    mask = atlas[1] != 0.0
    assert np.all(atlas[:, ~mask] == 0.0)


def test_restore_uses_cheap_frozen_objects_without_raw_io(bound_96, monkeypatch):
    monkeypatch.setattr(
        allen_v6,
        "_decode_and_preprocess_allen_v6",
        lambda: pytest.fail("cheap restore must not decode NRRDs"),
    )
    restored = allen_v6.restore_bound_allen_atlas_v6(
        atlas_volume_float32=bound_96.atlas_volume_float32,
        support_index=bound_96.support_index,
        catalogue=bound_96.catalogue,
        binding=bound_96.binding,
    )
    assert allen_v6.verify_bound_allen_atlas_v6(restored)


def test_resume_can_separately_rehash_raw_sources_without_decoding(bound_96, monkeypatch):
    monkeypatch.setattr(
        allen_v6,
        "_decode_and_preprocess_allen_v6",
        lambda: pytest.fail("raw-source verification must not decode NRRDs"),
    )
    assert allen_v6.verify_pinned_allen_raw_sources_v6(bound_96.binding)


def test_re_receipted_raw_to_decoded_lineage_tamper_is_rejected(bound_96):
    binding = allen_v6._plain(bound_96.binding)
    binding["raw_sources"]["template"]["raw_sha256"] = "0" * 64
    binding = allen_v6._with_receipt(allen_v6._payload(binding))
    with pytest.raises(ValueError, match="provenance binding differs"):
        allen_v6.restore_bound_allen_atlas_v6(
            atlas_volume_float32=bound_96.atlas_volume_float32,
            support_index=bound_96.support_index,
            catalogue=bound_96.catalogue,
            binding=binding,
        )


def test_wrong_support_or_catalogue_cannot_be_restored(bound_96):
    support = dict(bound_96.support_index)
    support["support_index_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Support-index metadata"):
        allen_v6.restore_bound_allen_atlas_v6(
            atlas_volume_float32=bound_96.atlas_volume_float32,
            support_index=support,
            catalogue=bound_96.catalogue,
            binding=bound_96.binding,
        )

    catalogue = dict(bound_96.catalogue)
    catalogue["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="catalogue arrays or immutable receipt"):
        allen_v6.restore_bound_allen_atlas_v6(
            atlas_volume_float32=bound_96.atlas_volume_float32,
            support_index=bound_96.support_index,
            catalogue=catalogue,
            binding=bound_96.binding,
        )


def test_one_voxel_atlas_tamper_is_detected_and_reversible(bound_96):
    atlas = bound_96.atlas_volume_float32
    original = float(atlas[0, 0, 0, 0])
    atlas[0, 0, 0, 0] = np.float32(0.5 if original == 0.0 else 0.0)
    try:
        with pytest.raises(ValueError, match="decoded atlas is not exact"):
            allen_v6.verify_bound_allen_atlas_v6(bound_96)
    finally:
        atlas[0, 0, 0, 0] = np.float32(original)
    assert allen_v6.verify_bound_allen_atlas_v6(bound_96)


def test_slow_raw_to_catalogue_replay_matches_the_complete_96_binding(bound_96):
    replay = allen_v6.replay_allen_atlas_binding_v6(bound_96.binding)
    try:
        assert allen_v6.verify_bound_allen_atlas_v6(replay)
        assert allen_v6._plain(replay.binding) == allen_v6._plain(bound_96.binding)
    finally:
        del replay
        gc.collect()


def test_pinned_160_profile_reproduces_its_exact_full_catalogue_receipt():
    bundle = allen_v6.prepare_bound_allen_atlas_v6(raster_shape_h_w=(160, 160))
    try:
        assert allen_v6.verify_bound_allen_atlas_v6(bundle)
        assert bundle.catalogue["counts"]["cell_count"] == 98_304
        assert bundle.catalogue["receipt_sha256"] == (
            "ab189d84a9c397eafbc82e4b4245f57e3c4b82d617b53562d7271b331637a143"
        )
        assert bundle.binding["receipt_sha256"] == (
            "8963e58824721e7748ac448996da02170247c070d477ef1d8fd23ca8d5185ebc"
        )
    finally:
        del bundle
        gc.collect()
