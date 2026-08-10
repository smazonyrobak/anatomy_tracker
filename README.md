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

The **Manual alignment** tab is the direct landmark workflow: choose AP and cutting tilts, add corresponding atlas/slice landmarks, then transform the slice. The **Automatic alignment** tab is independent: enable **Smart surface brush** and paint over the intended brain object. The app uses the stroke as foreground evidence, selects the contrast-defined object, and converts its outer boundary into the exact number of evenly spaced points selected in **Auto-selection** (50 by default); changing that value resamples an existing automatically selected outline. Hold Shift while painting to subtract. Manual trustworthy surface arcs remain available across folds, tears, or missing boundary. Surface geometry establishes scale; brightness-invariant internal anatomy drives the atlas match.

To constrain automatic registration, enable **Limit AP search** and enter a stereotaxic interval such as `From +1200 um` and `To +1800 um`; bregma is 0, anterior is positive, and posterior is negative. A cancellable progress window shows the AP scan, shared-tilt estimation, and 25 µm/1° refinement. The custom brightness curve changes only display contrast and cannot change automatic detection or alignment.

After transforming or auto-aligning a slice, use the **Slice — Atlas** blend slider directly below the atlas comparison image to inspect either image alone or any intermediate opacity.

For a joint run, check the slices to include, drag them into explicit most-anterior-to-most-posterior order, and use **Auto-align selected A→P sequence**. The solver enforces that order while optimizing one shared L-R/D-V cutting tilt and a separate AP position/2D transform for every selected slice. Saved overlays and per-slice costs update while browsing.

The desktop launcher source is `source/proprietary_tracker_launcher.pyw`. Generated executables, bundled runtimes, and atlas data are intentionally excluded from Git because they exceed normal GitHub repository limits.

## Build the bundled tracker

```powershell
python -m PyInstaller --noconfirm --clean TrajectoryTracker.spec
```

Deploy the complete generated `dist/TrajectoryTracker` folder. Do not combine a newly built executable with an older `_internal` runtime; the executable and runtime must come from the same build.
