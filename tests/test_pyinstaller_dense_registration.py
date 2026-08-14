import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).parents[1]
SPEC = ROOT / "TrajectoryTracker.spec"
DENSE_MODEL_DIR = ROOT / "models" / "DiffeomorphicRegistration"
DENSE_FILENAMES = {
    "dense_registration.onnx",
    "dense_registration.metadata.json",
}


class AnalysisStub:
    def __init__(self, *args, **kwargs):
        self.pure = []
        self.scripts = []
        self.binaries = []
        self.datas = kwargs["datas"]


def execute_spec(monkeypatch, present_dense_files):
    verified = []
    atlas_runtime = ModuleType("atlas_pose_runtime")
    atlas_runtime.verify_atlas_pose_evaluated_bundle = lambda path: None
    atlas_runtime.verify_atlas_pose_model_bundle = lambda path: None
    dense_runtime = ModuleType("dense_registration_runtime")
    dense_runtime.verify_dense_registration_bundle = (
        lambda model, metadata: verified.append((Path(model), Path(metadata)))
    )
    monkeypatch.setitem(sys.modules, "atlas_pose_runtime", atlas_runtime)
    monkeypatch.setitem(sys.modules, "dense_registration_runtime", dense_runtime)

    original_is_file = Path.is_file

    def is_file(path):
        if path.parent == DENSE_MODEL_DIR:
            return path.name in present_dense_files
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", is_file)
    namespace = runpy.run_path(
        str(SPEC),
        init_globals={
            "SPECPATH": str(ROOT),
            "Analysis": AnalysisStub,
            "PYZ": lambda *args, **kwargs: None,
            "EXE": lambda *args, **kwargs: None,
            "COLLECT": lambda *args, **kwargs: None,
        },
    )
    return namespace["a"].datas, verified


def test_spec_omits_uninstalled_dense_registration_bundle(monkeypatch):
    datas, verified = execute_spec(monkeypatch, set())

    assert verified == []
    assert not any(destination == "models/DiffeomorphicRegistration" for _, destination in datas)


def test_dense_registration_resource_lookup_matches_spec_destination():
    from source import proprietary_trajectory_tool as tracker

    assert tracker.RESOURCE_DIR == ROOT
    assert tracker.DENSE_REGISTRATION_MODEL_PATH == (
        ROOT
        / "models"
        / "DiffeomorphicRegistration"
        / "dense_registration.onnx"
    )
    assert tracker.DENSE_REGISTRATION_METADATA_PATH == (
        ROOT
        / "models"
        / "DiffeomorphicRegistration"
        / "dense_registration.metadata.json"
    )
    tracker_source = (ROOT / "source" / "proprietary_trajectory_tool.py").read_text(
        encoding="utf-8"
    )
    assert "RESOURCE_DIR = Path(sys._MEIPASS)" in tracker_source


def test_dense_registration_standalone_dependency_is_runtime_pinned():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    spec = SPEC.read_text(encoding="utf-8")

    assert "onnxruntime-directml==1.24.4" in requirements
    assert '"onnxruntime"' not in spec
    assert '"torch"' in spec


def test_spec_verifies_and_bundles_complete_dense_registration_bundle(monkeypatch):
    datas, verified = execute_spec(monkeypatch, DENSE_FILENAMES)

    expected_files = [
        DENSE_MODEL_DIR / "dense_registration.onnx",
        DENSE_MODEL_DIR / "dense_registration.metadata.json",
    ]
    assert verified == [tuple(expected_files)]
    assert [
        (Path(source), destination)
        for source, destination in datas
        if destination == "models/DiffeomorphicRegistration"
    ] == [(path, "models/DiffeomorphicRegistration") for path in expected_files]


@pytest.mark.parametrize("filename", sorted(DENSE_FILENAMES))
def test_spec_rejects_partial_dense_registration_bundle(monkeypatch, filename):
    with pytest.raises(RuntimeError, match="dense-registration bundle is incomplete"):
        execute_spec(monkeypatch, {filename})
