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

The **Manual alignment** tab is the direct landmark workflow: choose AP and cutting tilts, add corresponding atlas/slice landmarks, then transform the slice. In **Automatic alignment**, choose the published DeepSlice 1.2.8 mouse ensemble, the locally trained AtlasPose CNN, or a weighted vote between them. DeepSlice remains the default. The selected predictor supplies the initial AP and cutting tilts; the application then performs the same coarse-to-fine modality-independent search against Allen atlas candidates. Internal anatomy drives the MIND image score, the selected brain surface contributes a modest shape term, and trusted surface points calibrate isotropic scale and translation. The custom brightness curve, zoom, and opacity are never used, so display customization cannot change an automatic result.

AtlasPose was trained with arbitrary in-plane rotation and 0.5–1.5× scale augmentation. At inference, the trusted brain mask supplies roll and scale canonicalization. The user's displayed A-to-P orientation is authoritative: the automatic slice-to-atlas transform is orientation-preserving, and only the horizontal/vertical controls may reflect a slice and its trusted or probe points. DeepSlice receives the corresponding preprocessed crop.

DeepSlice supports whole coronal mouse-brain sections. Automatic alignment requires at least eight trusted surface points. **Auto-align current** estimates one outlined section independently, including its own L-R and D-V tilt. **Auto-align all** uses only outlined slices and solves one exact shared L-R/D-V tilt while retaining a separate AP position and in-plane affine for each slice. Unoutlined slices cannot influence that result. When possible, use a representative multi-section batch from the same tissue block because all participating sections inform the shared-angle estimate.

Automatic registration is deliberately two-stage. **Auto-align** first freezes AP, L-R/D-V tilt, and the surface-calibrated affine transform. After reviewing that pose, **Fit current slice to atlas (nonlinear)** applies a bounded in-plane B-spline refinement to internal anatomy on that exact atlas plane. It uses Mattes mutual information with deterministic dense sampling and line-search optimization, preserves bounded residual in-plane proportional differences, exponentiates the residual field into inverse-consistent maps, and accepts it only when geometry, MIND correspondence, surface overlap, and tissue-coverage gates pass. The fit has its own non-modal progress/cancel flow and cannot alter AP or either tilt. It cannot repair a wrong pose, folded or torn tissue, or missing anatomy; rejected pairs retain their affine alignment. Review every accepted result using the atlas blend and 3D planes. A learned diffeomorphic model can replace this backend only after its independently locked internal-landmark benchmark and source-pinned release evidence pass.

Enable **Smart brush** and paint over the intended brain object to create an editable boundary with the selected number of points (50 by default). In **Add / edit points**, add, drag, or delete individual points; use **Erase points** over folds, tears, or missing tissue. Deleted gaps are excluded from final surface calibration rather than filled. Editing a surface invalidates its previous automatic alignment. Any trusted surface selection, including discontinuous arcs, defines a background-margin crop for DeepSlice while preserving the internal anatomy used for matching.

To use prior anatomical knowledge, enable **Limit AP search** and enter a stereotaxic interval such as `From -1100 µm` and `To -2200 µm`; bregma is 0, anterior is positive, and posterior is negative. Only atlas sections inside that interval are evaluated—results are not clipped to a boundary after prediction. Without an explicit interval, refinement searches locally around the selected predictor's estimate and warns if the best match reaches that local edge. The non-modal progress window reports model inference and AP/tilt search. Cancel preserves previous alignments, and the rest of the interface remains usable while alignment runs.

After transforming or auto-aligning a slice, use the **Slice — Atlas** blend slider directly below the atlas comparison image to inspect either image alone or any intermediate opacity.

The Atlas/Slices setup row, Alignment/Probe mapping row, controls/image workspace, Atlas/Slice/3D views, and slice/contrast editor are separated by draggable splitter handles. Pull the controls/image handle upward to give the brain-slice view more room. The manual-alignment contrast editor shows a continuous smoothed intensity distribution; drag curve points, double-click to add one, right-click an interior point to remove it, or use **Reset linear**.

AP order is optional: leave every slice unchecked to impose no order, or drag any known subset into most-anterior-to-most-posterior order and check only that subset. The candidate solver enforces strict AP order only for checked slices; unchecked slices keep their own best candidate. Results report raw DeepSlice versus refined AP, search boundary and ambiguity warnings, surface residuals, runtime device, candidate counts, and primary/secondary-network disagreement. **REVIEW** is a conservative audit signal, not a calibrated confidence probability. The raw ensemble OUV, final shared tilt, constraints, search diagnostics, model hashes, atlas hashes, and region-lookup hash are retained in trajectory manifests.

Choose the probe being edited with **Active probe (edit / map)**. Every probe keeps its own points, weighted 3D regression, endpoint, and export; all probe fits remain visible together in distinct colors while the active one is emphasized. Mapping one probe updates only that probe's rows in `channels.csv` and `units.csv`.

Use **File > Save session** (`Ctrl+S`) to create one portable `.attracker` file. It embeds the loaded source images and preserves all slice orientation, surface selection, transforms, probe marks, alignment metadata, optional AP/order/surgical constraints, endpoint modes and display settings. **File > Open session** (`Ctrl+O`) restores that complete state without needing the original image paths.

Optional surgical constraints are stored independently for each probe. Before alignment, select **Mark probe on slice** and place at least two observations directly on the raw histology; atlas and 3-D coordinates deliberately remain absent until a pose is solved. Enter the planned bregma-centred AP/ML coordinate and uncertainty radius, plus attack angle and tolerance; **0 degrees is horizontal and 90 degrees is vertical**. Then run **Auto-align current** or **Auto-align all**. The solver jointly evaluates image correspondence and only those poses satisfying the raw probe observations, optional AP bounds, optional partial anterior-to-posterior order, and—for Auto-align all—the exact shared cutting tilt. Candidate results are accepted only when the ordinary post-alignment trajectory displayed by the UI itself lies inside the AP/ML disk, attack-angle interval, insertion-depth and physical-shank bounds. Editing constraints never changes an existing result; it marks it pending until auto-alignment is rerun. When constraints are disabled, alignment follows the image-only path exactly.

The dedicated desktop source folder and bundled desktop build include checksum-locked ONNX conversions of the official DeepSlice 1.2.8 primary and secondary mouse models. The optional AtlasPose model is also checksum-pinned: it passed every absolute-quality gate on the sealed benchmark and beat DeepSlice on AP and L-R, while the D-V difference was statistically inconclusive, so DeepSlice remains the default and AtlasPose is not described as globally superior. ONNX Runtime uses the first supported GPU provider by default and falls back to CPU; export checks compare CPU with every available application accelerator provider. DeepSlice should be cited as Carey et al., *Nature Communications* 14, 5884 (2023). Its repository LICENSE is included in `licenses/DeepSlice-LICENSE.txt`; current repository and PyPI license metadata are contradictory, so maintainer clarification is required before redistributing it in a closed-source bundle. AtlasPose uses Allen-derived data and permissively licensed pretrained timm weights, but this project does not grant redistribution or commercial-use clearance for Allen data or derived weights.

Generated executables, bundled runtimes, atlas data, the two approximately 93 MB DeepSlice ONNX binaries, the approximately 119 MB AtlasPose ONNX model, and raw sealed rows are intentionally excluded from Git. Tracked metadata retain model hashes and the sealed summary; the complete immutable evidence remains beside the local model and in the `F:` training workspace. Local builds include the optional AtlasPose candidate only when its source-pinned model, metadata, provenance, and integrity-checked evaluation report agree. The desktop shortcut targets the bundled executable directly, and the application resolves the atlas from this dedicated desktop folder without a wrapper process.

## AtlasPose validation status

The historical local ConvNeXt-Tiny v6 artifact reached sealed synthetic-test MAE of 58.72 µm AP, 0.934° L–R, and 1.052° D–V. On the 148-section published DeepSlice set, which was later used as development feedback and is therefore not an untouched holdout, v6 reached 245.20 µm, 1.639°, and 3.996° MAE versus 174.26 µm, 1.463°, and 1.268° for the published DeepSlice outputs. It remains experimental and is not packaged as an approved AtlasPose release. The pending v7 pipeline compares three heads and three commercially permissive backbones on synthetic and registered histology, then requires a one-shot sealed release gate; no v7 accuracy is claimed yet. Full protocol and provenance details are in `training/README.md`.

## Build the bundled tracker

```powershell
python -m PyInstaller --noconfirm --clean TrajectoryTracker.spec
```

Deploy the complete generated `dist/TrajectoryTracker` folder. Do not combine a newly built executable with an older `_internal` runtime; the executable and runtime must come from the same build.
