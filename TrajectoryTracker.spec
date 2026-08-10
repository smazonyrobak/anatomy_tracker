# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / "source" / "proprietary_trajectory_tool.py")],
    pathex=[str(ROOT / "source")],
    binaries=[],
    datas=[],
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
        "tensorflow",
        "pyarrow",
        "numba",
        "llvmlite",
        "tables",
        "h5py",
        "zmq",
        "rich",
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name="TrajectoryTracker",
)
