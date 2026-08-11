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

The **Manual alignment** tab is the direct landmark workflow: choose AP and cutting tilts, add corresponding atlas/slice landmarks, then transform the slice. The **Automatic alignment** tab first runs the published DeepSlice 1.2.8 two-model mouse ensemble, then performs a coarse-to-fine modality-independent search against Allen atlas candidates. Internal anatomy drives the MIND image score, the selected brain surface contributes a modest shape term, and trusted surface points finally calibrate isotropic scale and translation. Rotation and horizontal/vertical flips are applied before inference. The custom brightness curve, zoom, and opacity are never used, so display customization cannot change an automatic result.

DeepSlice supports whole coronal mouse-brain sections. Automatic alignment requires at least eight trusted surface points. **Auto-align current** estimates one outlined section independently, including its own L-R and D-V tilt. **Auto-align all** uses only outlined slices and solves one exact shared L-R/D-V tilt while retaining a separate AP position and in-plane affine for each slice. Unoutlined slices cannot influence that result. When possible, use a representative multi-section batch from the same tissue block because all participating sections inform the shared-angle estimate. Review every result using the atlas blend and 3D planes; the method is affine and does not nonlinearly repair folded or distorted tissue.

Enable **Smart brush** and paint over the intended brain object to create an editable boundary with the selected number of points (50 by default). In **Add / edit points**, add, drag, or delete individual points; use **Erase points** over folds, tears, or missing tissue. Deleted gaps are excluded from final surface calibration rather than filled. Editing a surface invalidates its previous automatic alignment. Any trusted surface selection, including discontinuous arcs, defines a background-margin crop for DeepSlice while preserving the internal anatomy used for matching.

To use prior anatomical knowledge, enable **Limit AP search** and enter a stereotaxic interval such as `From -1100 um` and `To -2200 um`; bregma is 0, anterior is positive, and posterior is negative. Only atlas sections inside that interval are evaluated—results are not clipped to a boundary after prediction. Without an explicit interval, refinement searches locally around DeepSlice's estimate and warns if the best match reaches that local edge. The non-modal progress window reports model inference and AP/tilt search. Cancel preserves previous alignments, and the rest of the interface remains usable while alignment runs.

After transforming or auto-aligning a slice, use the **Slice — Atlas** blend slider directly below the atlas comparison image to inspect either image alone or any intermediate opacity.

The Atlas/Slices setup row, Alignment/Probe mapping row, controls/image workspace, Atlas/Slice/3D views, and slice/contrast editor are separated by draggable splitter handles. Pull the controls/image handle upward to give the brain-slice view more room. The manual-alignment contrast editor shows a continuous smoothed intensity distribution; drag curve points, double-click to add one, right-click an interior point to remove it, or use **Reset linear**.

AP order is optional: leave every slice unchecked to impose no order, or drag any known subset into most-anterior-to-most-posterior order and check only that subset. The candidate solver enforces strict AP order only for checked slices; unchecked slices keep their own best candidate. Results report raw DeepSlice versus refined AP, search boundary and ambiguity warnings, surface residuals, runtime device, candidate counts, and primary/secondary-network disagreement. **REVIEW** is a conservative audit signal, not a calibrated confidence probability. The raw ensemble OUV, shared-angle OUV, constraints, search diagnostics, model hashes, atlas hashes, and region-lookup hash are retained in trajectory manifests.

Choose the probe being edited with **Active probe (edit / map)**. Every probe keeps its own points, weighted 3D regression, endpoint, and export; all probe fits remain visible together in distinct colors while the active one is emphasized. Mapping one probe updates only that probe's rows in `channels.csv` and `units.csv`.

The dedicated desktop source folder and bundled desktop build include checksum-locked ONNX conversions of the official DeepSlice 1.2.8 primary and secondary mouse models for offline inference. The application uses ONNX Runtime DirectML on the first available GPU by default and automatically falls back to the CPU if DirectML is unavailable; the alignment result is numerically equivalent on either backend. DeepSlice should be cited as Carey et al., *Nature Communications* 14, 5884 (2023). Its repository LICENSE is included in `licenses/DeepSlice-LICENSE.txt`; note that current repository/PyPI license metadata are contradictory, so obtain maintainer clarification before redistributing a closed-source bundle outside this workstation.

Generated executables, bundled runtimes, atlas data, and the two approximately 93 MB ONNX binaries are intentionally excluded from Git. The required model filenames and SHA-256 hashes remain documented in `models/DeepSlice/README.md`; local builds require those validated files to be present. The desktop shortcut targets the deployed bundled executable directly; the application resolves the atlas from this dedicated desktop folder without a wrapper process.

## Build the bundled tracker

```powershell
python -m PyInstaller --noconfirm --clean TrajectoryTracker.spec
```

Deploy the complete generated `dist/TrajectoryTracker` folder. Do not combine a newly built executable with an older `_internal` runtime; the executable and runtime must come from the same build.
