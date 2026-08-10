# Anatomy Tracker

Standalone PySide6 application for registering histological slices to the Allen CCF, tracing Neuropixels probe trajectories, and assigning atlas structures to `channels.csv` and `units.csv`.

## Installation

```powershell
python -m pip install -r requirements.txt
```

Place the Allen 25 µm atlas files in:

```text
data/Allen Brain Atlas 25um/
```

The folder must contain `average_template_25.nrrd` and `annotation_25.nrrd`. `query.csv` and `atlas_meshdata.pkl` add region names and the 3D brain mesh.

## Run

```powershell
python source/proprietary_trajectory_tool.py
```

Select a preprocessing output folder containing `channels.csv` and `units.csv`, either directly or inside its `preprocessed_data` subfolder.

Atlas positions and exported coordinates use a bregma-centred stereotaxic convention: AP 0 is bregma, anterior AP is positive, and posterior AP is negative.

Use **Add slices** to select many TIFF, PNG, JPEG, or BMP images at once. Browse them with the slice selector, the previous/next buttons, or `Ctrl+Left` and `Ctrl+Right`; points and adjustments are retained separately for each slice.

The desktop launcher source is `source/proprietary_tracker_launcher.pyw`. Generated executables, bundled runtimes, and atlas data are intentionally excluded from Git because they exceed normal GitHub repository limits.

## Build the bundled tracker

```powershell
python -m PyInstaller --noconfirm --clean TrajectoryTracker.spec
```

Deploy the complete generated `dist/TrajectoryTracker` folder. Do not combine a newly built executable with an older `_internal` runtime; the executable and runtime must come from the same build.
