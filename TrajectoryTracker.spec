# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH)
DEEPSLICE_MODELS = ROOT / "models" / "DeepSlice"
ATLAS_POSE_MODELS = ROOT / "models" / "AtlasPose"

a = Analysis(
    [str(ROOT / "source" / "proprietary_trajectory_tool.py")],
    pathex=[str(ROOT / "source")],
    binaries=[],
    datas=[
        (str(DEEPSLICE_MODELS / "deepslice_mouse_primary_opset18.onnx"), "models/DeepSlice"),
        (str(DEEPSLICE_MODELS / "deepslice_mouse_secondary_opset18.onnx"), "models/DeepSlice"),
        (str(ATLAS_POSE_MODELS / "atlas_pose.onnx"), "models/AtlasPose"),
        (str(ATLAS_POSE_MODELS / "atlas_pose.json"), "models/AtlasPose"),
        (str(ROOT / "licenses" / "DeepSlice-LICENSE.txt"), "licenses"),
    ],
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
