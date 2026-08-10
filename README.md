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

The Atlas/Slices setup row, Alignment/Probe mapping row, controls/image workspace, Atlas/Slice/3D views, and slice/contrast editor are separated by draggable splitter handles. Pull the controls/image handle upward to give the brain-slice view more room. The manual-alignment contrast editor shows a continuous smoothed intensity distribution; drag curve points, double-click to add one, right-click an interior point to remove it, or use **Reset linear**.

For a joint run, use **Auto-align all outlined slices**. Every slice with at least 8 surface points participates automatically, and the solver estimates one shared L-R/D-V cutting tilt plus a separate AP position/2D transform for each. AP order is optional: leave every slice unchecked for a fully unconstrained search, or drag a subset into known most-anterior-to-most-posterior order and check only that subset. The solver enforces the checked partial-order chain without imposing an order on unchecked slices. Saved overlays and per-slice costs update while browsing.

The desktop launcher source is `source/proprietary_tracker_launcher.pyw`. Generated executables, bundled runtimes, and atlas data are intentionally excluded from Git because they exceed normal GitHub repository limits.

## Build the bundled tracker

```powershell
python -m PyInstaller --noconfirm --clean TrajectoryTracker.spec
```

Deploy the complete generated `dist/TrajectoryTracker` folder. Do not combine a newly built executable with an older `_internal` runtime; the executable and runtime must come from the same build.
