# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys


ROOT = Path(SPECPATH)
DEEPSLICE_MODELS = ROOT / "models" / "DeepSlice"
ATLAS_POSE_MODELS = ROOT / "models" / "AtlasPose"
NONLINEAR_MODEL = ROOT / "models" / "DiffeomorphicRegistration" / "diffeomorphic.onnx"
sys.path.insert(0, str(ROOT / "source"))
from diffeomorphic_registration_runtime import verify_diffeomorphic_model_bundle
from atlas_pose_runtime import verify_atlas_pose_model_bundle

verify_diffeomorphic_model_bundle(NONLINEAR_MODEL)
NONLINEAR_DATAS = [
    (str(NONLINEAR_MODEL), "models/DiffeomorphicRegistration"),
    (str(NONLINEAR_MODEL.with_suffix(".manifest.json")), "models/DiffeomorphicRegistration"),
]
ATLAS_POSE_FILES = [
    ATLAS_POSE_MODELS / name
    for name in ("atlas_pose.onnx", "atlas_pose.json", "RELEASE_REPORT.json", "SEALED_metrics.json")
]
ATLAS_POSE_DATAS = []
if all(path.is_file() for path in ATLAS_POSE_FILES):
    verify_atlas_pose_model_bundle(ATLAS_POSE_FILES[0])
    ATLAS_POSE_DATAS = [(str(path), "models/AtlasPose") for path in ATLAS_POSE_FILES]

a = Analysis(
    [str(ROOT / "source" / "proprietary_trajectory_tool.py")],
    pathex=[str(ROOT / "source")],
    binaries=[],
    datas=[
        (str(DEEPSLICE_MODELS / "deepslice_mouse_primary_opset18.onnx"), "models/DeepSlice"),
        (str(DEEPSLICE_MODELS / "deepslice_mouse_secondary_opset18.onnx"), "models/DeepSlice"),
        (str(ROOT / "licenses" / "DeepSlice-LICENSE.txt"), "licenses"),
    ] + ATLAS_POSE_DATAS + NONLINEAR_DATAS,
    hiddenimports=[],
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
