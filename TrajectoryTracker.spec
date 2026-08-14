# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import json
import sys


ROOT = Path(SPECPATH)
DEEPSLICE_MODELS = ROOT / "models" / "DeepSlice"
ATLAS_POSE_MODELS = ROOT / "models" / "AtlasPose"
DENSE_REGISTRATION_MODELS = ROOT / "models" / "DiffeomorphicRegistration"
sys.path.insert(0, str(ROOT / "source"))
from atlas_pose_runtime import verify_atlas_pose_evaluated_bundle, verify_atlas_pose_model_bundle
from dense_registration_runtime import verify_dense_registration_bundle
ATLAS_POSE_NAMES = [
    "atlas_pose.onnx",
    "atlas_pose.json",
    "provenance.json",
    "RELEASE_REPORT.json",
]
atlas_pose_release = ATLAS_POSE_MODELS / "RELEASE_REPORT.json"
atlas_pose_evidence = (
    json.loads(atlas_pose_release.read_text(encoding="utf-8"))
    if atlas_pose_release.is_file()
    else {}
)
if atlas_pose_evidence.get("release_approved") is True:
    ATLAS_POSE_NAMES.extend(
        (
            "SEALED_metrics.json",
            "SEALED_predictions.csv",
            "PRESEALED_COMMITMENT.json",
            "SEALED_CLAIM.json",
            "SEALED_CONSUMPTION_RECEIPT.json",
        )
    )
    if atlas_pose_evidence.get("release_report_version") == 4:
        ATLAS_POSE_NAMES.extend(
            ("SEALED_RECOVERY_COMMITMENT.json", "FAILED_ATTEMPT_CLAIM.json", "FAILED_ATTEMPT_RECEIPT.json")
        )
ATLAS_POSE_FILES = [
    ATLAS_POSE_MODELS / name
    for name in ATLAS_POSE_NAMES
]
ATLAS_POSE_DATAS = []
if all(path.is_file() for path in ATLAS_POSE_FILES):
    verifier = (
        verify_atlas_pose_model_bundle
        if atlas_pose_evidence.get("release_approved") is True
        else verify_atlas_pose_evaluated_bundle
    )
    verifier(ATLAS_POSE_FILES[0])
    ATLAS_POSE_DATAS = [(str(path), "models/AtlasPose") for path in ATLAS_POSE_FILES]

DENSE_REGISTRATION_FILES = [
    DENSE_REGISTRATION_MODELS / "dense_registration.onnx",
    DENSE_REGISTRATION_MODELS / "dense_registration.metadata.json",
]
DENSE_REGISTRATION_PRESENT = [path.is_file() for path in DENSE_REGISTRATION_FILES]
if any(DENSE_REGISTRATION_PRESENT) and not all(DENSE_REGISTRATION_PRESENT):
    raise RuntimeError("The dense-registration bundle is incomplete")
DENSE_REGISTRATION_DATAS = []
if all(DENSE_REGISTRATION_PRESENT):
    verify_dense_registration_bundle(*DENSE_REGISTRATION_FILES)
    DENSE_REGISTRATION_DATAS = [
        (str(path), "models/DiffeomorphicRegistration")
        for path in DENSE_REGISTRATION_FILES
    ]

a = Analysis(
    [str(ROOT / "source" / "proprietary_trajectory_tool.py")],
    pathex=[str(ROOT / "source")],
    binaries=[],
    datas=[
        (str(DEEPSLICE_MODELS / "deepslice_mouse_primary_opset18.onnx"), "models/DeepSlice"),
        (str(DEEPSLICE_MODELS / "deepslice_mouse_secondary_opset18.onnx"), "models/DeepSlice"),
        (str(ROOT / "licenses" / "DeepSlice-LICENSE.txt"), "licenses"),
    ] + ATLAS_POSE_DATAS + DENSE_REGISTRATION_DATAS,
    hiddenimports=["imagecodecs._imcd", "imagecodecs._shared_cython"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "matplotlib",
        "IPython",
        "pytest",
        "dask",
        "tkinter",
        "torch",
        "torchvision",
        "torchaudio",
        "DeepSlice",
        "tensorflow",
        "keras",
        "h5py",
        "sklearn",
        "skimage",
        "pyarrow",
        "polars",
        "_polars_runtime_32",
        "numba",
        "llvmlite",
        "tables",
        "zmq",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TrajectoryTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(ROOT / "anatomy.ico")],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TrajectoryTracker",
)
