import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
FROZEN_AT_COMMIT = "bbb2b45c272188bfe8583ae34f7cafcd6daaced3"
LEGACY_SHA256 = {
    "arbitrary_plane_finite_training_runner_v4.py": "0c7163ab33ec2c5e3bb028d26a6981affeeb5d759857dc1c557d05e9ea5bf5c1",
    "arbitrary_plane_staged_training.py": "a9a146ed2db4a42dfb9bf5932d502554978704cd4fdedd9de7b061d1535f1a05",
    "arbitrary_plane_joint_model.py": "e8982696523a467ccaecd7b1047d539f31212fc610195ac2c1f73dd696ff5694",
    "arbitrary_plane_joint_loss.py": "5d36549802b91ecb8d98bb9b4bbf9e0554e81a4fa33250a35c9d55d90101c041",
    "arbitrary_plane_recurrent_model.py": "d5e3b259979c34512742dcb7ec8f27c2293c83a0f59e3e79c3ac40b3b01e1e5e",
    "arbitrary_plane_inference_v3.py": "4ea8ef2fecd16fb41c9441ab849940c12d462e7730deda847da54afc72443c38",
    "arbitrary_plane_catalogue_capture_audit_v3.py": "49be41d6a1039bf75e0df183a5bcde40437447021328ee2611d52c946dca55a8",
    "run_arbitrary_plane_authentic_development_v3.py": "2ed4407ab767ce5cc98190439b865864b4846ddd296358623f9b5c84cc13e818",
    "run_arbitrary_plane_authentic_finite_development_v4.py": "15ae850fa72c55e8ca96e5beb88dfa355d4e6ab2d29ccdf1a0f3847f696fa636",
}
V4_RUNNERS = {
    "arbitrary_plane_finite_training_runner_v4.py",
    "run_arbitrary_plane_authentic_finite_development_v4.py",
}


def test_receipt_sensitive_legacy_sources_match_the_frozen_baseline():
    observed = {
        name: hashlib.sha256((TRAINING / name).read_bytes()).hexdigest()
        for name in LEGACY_SHA256
    }
    assert observed == LEGACY_SHA256, f"legacy namespace changed after {FROZEN_AT_COMMIT}"


def test_v6_modules_are_siblings_outside_the_legacy_namespace():
    paths = {
        path
        for path in ROOT.rglob("*_v6.py")
        if path.parent != ROOT / "tests"
    }
    assert paths
    assert all(path.is_file() and path.parent == TRAINING for path in paths)
    assert paths.isdisjoint({TRAINING / name for name in LEGACY_SHA256})


def test_v4_runners_do_not_import_v6_modules():
    v6_module_names = {
        path.stem
        for path in ROOT.rglob("*_v6.py")
        if path.parent != ROOT / "tests"
    }
    imported = set()
    for name in V4_RUNNERS:
        tree = ast.parse((TRAINING / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    imported_names = {module.rsplit(".", 1)[-1] for module in imported}
    assert imported_names.isdisjoint(v6_module_names)
