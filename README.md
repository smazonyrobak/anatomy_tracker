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

The 3D plane is the histology slice plane. It follows the current slice's saved AP and L-R/D-V tilt while browsing. Enable **All slice planes** below the 3D view to compare every slice at once; each plane is labelled with its slice number and filename. A joint auto-alignment therefore shows the slices at their individual AP positions with one shared tilt, while independent auto-alignments retain their separate tilts.

The **Manual alignment** tab is the direct landmark workflow: choose AP and cutting tilts, add corresponding atlas/slice landmarks, then transform the slice. The **Automatic alignment** tab runs the published DeepSlice 1.2.8 two-model mouse ensemble on the selected coronal brain object while preserving all internal anatomy. It predicts QuickNII anchoring vectors from internal anatomy; the tracker then uses the trusted outer-surface points to robustly calibrate residual isotropic scale and translation against the predicted Allen CCF plane. Rotation and horizontal/vertical flips are applied before inference. The custom brightness curve, zoom, and opacity are never used by automatic alignment, so display customization cannot change the result.

DeepSlice supports whole coronal mouse-brain sections. Automatic alignment requires at least eight trusted surface points. **Auto-align current** estimates one outlined section independently, including its own L-R and D-V tilt. **Auto-align all** uses only outlined slices, applies DeepSlice angle integration, and stores one exactly shared L-R/D-V tilt across the batch while retaining a separate AP position and in-plane affine for each slice. Unoutlined slices cannot influence that result. When possible, use a representative multi-section batch from the same tissue block because all participating sections inform the shared-angle estimate. Review every result using the atlas blend and 3D planes; the model provides an affine initialization, not nonlinear correction for folded or distorted tissue.

Enable **Smart brush** and paint over the intended brain object to create an editable boundary with the selected number of points (50 by default). In **Add / edit points**, add, drag, or delete individual points; use **Erase points** over folds, tears, or missing tissue. Deleted gaps are excluded from surface calibration rather than filled. Editing a surface invalidates its previous automatic alignment. The selected object is cropped with a background margin so other tissue on the camera frame cannot confuse DeepSlice, but it is not masked or recolored: all internal anatomy remains visible while the trusted points remove residual scale and position bias.

To use prior anatomical knowledge, enable **Constrain AP estimate** and enter a stereotaxic interval such as `From -1100 um` and `To -2200 um`; bregma is 0, anterior is positive, and posterior is negative. DeepSlice is a direct regressor rather than an atlas-screenshot search, so this is applied as a hard output prior: an estimate outside the interval is projected to the nearest permitted value and marked for review. The non-modal progress window reports model loading, two-model inference, shared-angle integration, coordinate conversion, and surface calibration. Cancel discards the pending result without mutating saved alignments; a native ONNX inference call already in flight may finish in the worker before cleanup.

After transforming or auto-aligning a slice, use the **Slice — Atlas** blend slider directly below the atlas comparison image to inspect either image alone or any intermediate opacity.

The Atlas/Slices setup row, Alignment/Probe mapping row, controls/image workspace, Atlas/Slice/3D views, and slice/contrast editor are separated by draggable splitter handles. Pull the controls/image handle upward to give the brain-slice view more room. The manual-alignment contrast editor shows a continuous smoothed intensity distribution; drag curve points, double-click to add one, right-click an interior point to remove it, or use **Reset linear**.

AP order is optional: leave every slice unchecked to preserve the model's independent AP estimates, or drag any known subset into most-anterior-to-most-posterior order and check only that subset. A least-change isotonic constraint makes only that checked subset monotonic; unchecked slices remain independent apart from an optional common AP range. The result reports pre-constraint versus constrained AP, surface-fit residuals, runtime device, and primary/secondary-network disagreement. It marks **REVIEW** at conservative 400 µm AP / 5° tilt guardrails; this is a review signal, not a calibrated confidence probability. The raw ensemble OUV, any shared-angle OUV, constraints, diagnostics, DeepSlice version, model hashes, atlas hashes, and region-lookup hash are saved separately in the trajectory manifest.

The dedicated desktop source folder and bundled desktop build include checksum-locked ONNX conversions of the official DeepSlice 1.2.8 primary and secondary mouse models for offline inference. The application uses ONNX Runtime DirectML on the first available GPU by default and automatically falls back to the CPU if DirectML is unavailable; the alignment result is numerically equivalent on either backend. DeepSlice should be cited as Carey et al., *Nature Communications* 14, 5884 (2023). Its repository LICENSE is included in `licenses/DeepSlice-LICENSE.txt`; note that current repository/PyPI license metadata are contradictory, so obtain maintainer clarification before redistributing a closed-source bundle outside this workstation.

Generated executables, bundled runtimes, atlas data, and the two approximately 93 MB ONNX binaries are intentionally excluded from Git. The required model filenames and SHA-256 hashes remain documented in `models/DeepSlice/README.md`; local builds require those validated files to be present. The desktop shortcut should target the bundled `tools/TrajectoryTracker/TrajectoryTracker.exe` directly; the application resolves the atlas from this dedicated desktop folder without a wrapper process.

## Build the bundled tracker

```powershell
python -m PyInstaller --noconfirm --clean TrajectoryTracker.spec
```

Deploy the complete generated `dist/TrajectoryTracker` folder. Do not combine a newly built executable with an older `_internal` runtime; the executable and runtime must come from the same build.
