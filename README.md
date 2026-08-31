# Anatomy Tracker

Windows PySide6 application for registering histological slices to the Allen CCF, tracing Neuropixels probe trajectories, and assigning atlas structures to `channels.csv` and `units.csv`.

## Requirements

- 64-bit Windows 10 or 11. Run the application in native Windows, not WSL.
- 64-bit Python 3.13. The pinned package set is validated with Python 3.13.
- A current graphics driver with ONNX Runtime DirectML support. DeepSlice can fall back to CPU, but the automatic dense-registration warp requires `DmlExecutionProvider`.
- Internet access during setup to download approximately 349 MB of checksum-pinned models and atlas data from the GitHub release.
- Histology images in TIFF, PNG, JPEG, or BMP format. Channel/unit mapping additionally requires `channels.csv` and `units.csv` from the preprocessing pipeline.

Git is convenient but not mandatory: either clone the repository or use **Code > Download ZIP** on GitHub and extract it before continuing.

## Clean installation

Open PowerShell in the folder where the repository should be stored:

```powershell
git clone https://github.com/smazonyrobak/anatomy_tracker.git
cd anatomy_tracker
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_runtime.py
```

If you downloaded the ZIP, skip the `git clone` and `cd` lines, open PowerShell inside the extracted `anatomy_tracker-main` folder, and run the remaining commands there.

If the `py` command is unavailable, install 64-bit Python 3.13 from Python.org or replace `py -3.13` with the full path to a Python 3.13 executable. Activation of the virtual environment is optional because every command above addresses its Python executable directly.

`setup_runtime.py` installs and verifies exactly the runtime assets used by the validated desktop copy:

- DeepSlice 1.2.8 primary and secondary ONNX networks;
- the evaluated AtlasPose ONNX network;
- the dense-registration ONNX network;
- the Allen CCFv3 25 µm template, annotation, structure lookup, and 3-D mesh.

The command is safe to rerun. A valid existing file is retained; a missing or incorrect file is downloaded again and accepted only when its SHA-256 checksum matches. See [`RUNTIME_ASSETS.md`](RUNTIME_ASSETS.md) for provenance, licensing, and citations.

## Verify and run

Run the setup command once more. A complete installation prints eight `ready:` lines followed by the verification message:

```powershell
.\.venv\Scripts\python.exe setup_runtime.py
```

Check that DirectML is available:

```powershell
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The output should include both `DmlExecutionProvider` and `CPUExecutionProvider`. Then start the application from the repository root:

```powershell
.\.venv\Scripts\python.exe source\proprietary_trajectory_tool.py
```

On first use, choose the bundled `data\Allen Brain Atlas 25um` directory if an atlas-folder dialog appears. Add one or more histology images with **Add slices**. For channel/unit mapping, select a preprocessing output folder containing `channels.csv` and `units.csv`, either directly or inside its `preprocessed_data` subfolder.

## Troubleshooting setup

- **`No matching distribution found`**: confirm that the environment uses 64-bit Python 3.13 with `.\.venv\Scripts\python.exe --version`.
- **A model or atlas file is reported missing**: run `.\.venv\Scripts\python.exe setup_runtime.py` from the repository root. Do not download or rename the files manually.
- **`DmlExecutionProvider` is absent**: update the Windows graphics driver and confirm that the app is running in native Windows. Manual alignment remains available, but the automatic dense-registration warp cannot run without DirectML.
- **A download was interrupted**: rerun `setup_runtime.py`. Partial `.part` files are not accepted as runtime assets.
- **GitHub downloads are blocked**: allow access to `github.com` and `release-assets.githubusercontent.com`, then rerun the setup command.
- **The GUI starts but automatic matching fails**: confirm all eight assets print as `ready:` and that the complete histological brain section has a trusted surface outline with at least eight points.

Do not copy a virtual environment from another computer. Recreate `.venv` with the commands above; only the repository and the release assets are portable.

The automatic-matching controls described below are the deployed legacy
coronal workflow. Its AP/L--R/D--V chart, AP interval/order and shared-tilt rule
do not define the arbitrary-plane model now under development; that future path
uses full QuickNII O/U/V/frame geometry, a full-plane domain, ordered signed
offsets along a declared stack normal and a shared 3-D plane normal.

Atlas positions and exported coordinates use a bregma-centred stereotaxic convention: AP 0 is bregma, anterior AP is positive, and posterior AP is negative.

Use **Add slices** to select many TIFF, PNG, JPEG, or BMP images at once. Browse them with the slice selector, the previous/next buttons, or `Ctrl+Left` and `Ctrl+Right`; points and adjustments are retained separately for each slice.

The 3D plane is the histology slice plane. It follows the current slice's saved AP and L-R/D-V tilt while browsing. Enable **All slice planes** below the 3D view to compare every slice at once; each plane is labelled with its slice number and filename. A joint auto-alignment therefore shows the slices at their individual AP positions with one shared tilt, while independent auto-alignments retain their separate tilts.

The **Landmark registration** tab maps histology coordinates onto the selected atlas plane: choose AP and cutting tilts yourself, or obtain them first with **Automatic section matching**, then add corresponding atlas/slice landmarks and apply the nonlinear landmark warp. Automatic section matching offers the published DeepSlice 1.2.8 mouse ensemble, the locally trained AtlasPose CNN, or a weighted vote between them. DeepSlice remains the default. The selected predictor supplies the initial AP and cutting tilts; the application then performs the same coarse-to-fine modality-independent search against Allen atlas candidates. Internal anatomy drives the MIND image score, the selected brain surface contributes a modest shape term, and trusted surface points calibrate isotropic scale and translation. The custom brightness curve, zoom, and opacity are never used, so display customization cannot change an automatic result.

AtlasPose was trained with arbitrary in-plane rotation and 0.5–1.5× scale augmentation. At inference, the trusted brain mask supplies roll and scale canonicalization. The user's displayed A-to-P orientation is authoritative: the automatic slice-to-atlas transform is orientation-preserving, and only the horizontal/vertical controls may reflect a slice and its trusted or probe points. DeepSlice receives the corresponding preprocessed crop.

DeepSlice supports whole coronal mouse-brain sections. Automatic alignment requires at least eight trusted surface points. **Auto-align current** estimates one outlined section independently, including its own L-R and D-V tilt. **Auto-align all** uses only outlined slices and solves one exact shared L-R/D-V tilt while retaining a separate AP position and in-plane affine for each slice. Unoutlined slices cannot influence that result. When possible, use a representative multi-section batch from the same tissue block because all participating sections inform the shared-angle estimate.

The workflow is deliberately two-stage. **Automatic section matching** estimates AP and L-R/D-V cutting tilt. Then either **Apply landmark warp** fits an in-plane thin-plate spline through corresponding landmarks, or **Apply automatic warp** uses the trained dense-registration model on that fixed atlas plane. **Apply to all aligned slices** runs the automatic warp sequentially in the background for every non-stale automatically matched slice and installs the whole batch atomically. The same nonlinear coordinate map renders the histology overlay and translates every marked electrode point before 3-D trajectory fitting and anatomical export.

Enable **Smart brush** and paint over the intended brain object to create an editable boundary with the selected number of points (50 by default). In **Add / edit points**, add, drag, or delete individual points; use **Erase points** over folds, tears, or missing tissue. Deleted gaps are excluded from final surface calibration rather than filled. Editing a surface invalidates its previous automatic alignment. Any trusted surface selection, including discontinuous arcs, defines a background-margin crop for DeepSlice while preserving the internal anatomy used for matching.

To use prior anatomical knowledge, enable **Limit AP search** and enter a stereotaxic interval such as `From -1100 µm` and `To -2200 µm`; bregma is 0, anterior is positive, and posterior is negative. Only atlas sections inside that interval are evaluated—results are not clipped to a boundary after prediction. Without an explicit interval, refinement searches locally around the selected predictor's estimate and warns if the best match reaches that local edge. The non-modal progress window reports model inference and AP/tilt search. Cancel preserves previous alignments, and the rest of the interface remains usable while alignment runs.

After applying either coordinate warp, use the **Slice — Atlas** blend slider directly below the atlas comparison image to inspect either image alone or any intermediate opacity.

The Atlas/Slices setup row, Alignment/Probe mapping row, controls/image workspace, Atlas/Slice/3D views, and slice/contrast editor are separated by draggable splitter handles. Pull the controls/image handle upward to give the brain-slice view more room. The manual-alignment contrast editor shows a continuous smoothed intensity distribution; drag curve points, double-click to add one, right-click an interior point to remove it, or use **Reset linear**.

AP order is optional: leave every slice unchecked to impose no order, or drag any known subset into most-anterior-to-most-posterior order and check only that subset. The candidate solver enforces strict AP order only for checked slices; unchecked slices keep their own best candidate. Results report raw DeepSlice versus refined AP, search boundary and ambiguity warnings, surface residuals, runtime device, candidate counts, and primary/secondary-network disagreement. **REVIEW** is a conservative audit signal, not a calibrated confidence probability. The raw ensemble OUV, final shared tilt, constraints, search diagnostics, model hashes, atlas hashes, and region-lookup hash are retained in trajectory manifests.

Choose the probe being edited with **Active probe (edit / map)**. Every probe keeps its own points, weighted 3D regression, endpoint, and export; all probe fits remain visible together in distinct colors while the active one is emphasized. Mapping one probe updates only that probe's rows in `channels.csv` and `units.csv`.

After **Map channels/units** succeeds, the app opens an Allen-colored physical probe view. Every recording site is shown at its real shank position, contiguous depth bands identify the structures traversed, hovering reports the channel and assigned region, and the adjacent table reports channel/unit counts per structure. A probe-aligned vertical atlas section shows the fitted trajectory in plane over the Allen template and translucent structure map; hover reports adjacent regions and stereotaxic coordinates, and clicking pins a region highlight. **View mapped anatomy** reopens the saved result for the active probe at any time.

Use **File > Save session** (`Ctrl+S`) to create one portable `.attracker` file. It embeds the loaded source images and preserves all slice orientation, surface selection, transforms, probe marks, alignment metadata, optional AP/order/surgical constraints, endpoint modes and display settings. **File > Open session** (`Ctrl+O`) restores that complete state without needing the original image paths.

Optional surgical constraints are stored independently for each probe. Before alignment, select **Mark probe on slice** and place at least two observations directly on the raw histology; atlas and 3-D coordinates deliberately remain absent until a pose is solved. Enter the planned bregma-centred AP/ML coordinate and uncertainty radius, plus attack angle and tolerance; **0 degrees is horizontal and 90 degrees is vertical**. Then run **Auto-align current** or **Auto-align all**. The solver jointly evaluates image correspondence and only those poses satisfying the raw probe observations, optional AP bounds, optional partial anterior-to-posterior order, and—for Auto-align all—the exact shared cutting tilt. Once atlas coordinates exist, the displayed and exported trajectory is itself the robust best fit constrained to enter inside the selected cortical disk and remain inside the angle and physical-depth bounds; disagreement among marked dots changes the best fit but never relaxes those hard bounds. The UI also reports signed horizontal roll relative to the bregma-to-lambda axis (parallel is 0°, perpendicular is ±90°). Changing a constraint immediately refits the displayed trajectory and marks automatic slice matching for rerun so the same information can update slice pose. When constraints are disabled, alignment and trajectory fitting follow their unconstrained paths.

The runtime release includes checksum-locked ONNX conversions of the official DeepSlice 1.2.8 primary and secondary mouse models. The optional AtlasPose model is also checksum-pinned: it passed every absolute-quality gate on the sealed benchmark and beat DeepSlice on AP and L-R, while the D-V difference was statistically inconclusive, so DeepSlice remains the default and AtlasPose is not described as globally superior. ONNX Runtime uses the first supported GPU provider by default and falls back to CPU; export checks compare CPU with every available application accelerator provider. DeepSlice should be cited as Carey et al., *Nature Communications* 14, 5884 (2023). Its MIT license is included in `licenses/DeepSlice-LICENSE.txt`. AtlasPose and the dense-registration model use Allen-derived data and are distributed for noncommercial research subject to the upstream Allen Institute terms described in `RUNTIME_ASSETS.md`.

Generated executables, Python environments, bundled runtimes, training workspaces, and raw sealed rows remain excluded from Git. Large inference and atlas files are published as versioned GitHub release assets and installed by `setup_runtime.py`; tracked metadata retain their hashes and validation summaries. Local builds include the optional AtlasPose candidate only when its source-pinned model, metadata, provenance, and integrity-checked evaluation report agree.

## AtlasPose validation status

The historical local ConvNeXt-Tiny v6 artifact reached sealed synthetic-test MAE of 58.72 µm AP, 0.934° L–R, and 1.052° D–V. On the 148-section published DeepSlice set, which was later used as development feedback and is therefore not an untouched holdout, v6 reached 245.20 µm, 1.639°, and 3.996° MAE versus 174.26 µm, 1.463°, and 1.268° for the published DeepSlice outputs. It remains experimental and is not packaged as an approved AtlasPose release. The pending v7 pipeline compares three heads and three commercially permissive backbones on synthetic and registered histology, then requires a one-shot sealed release gate; no v7 accuracy is claimed yet. Full protocol and provenance details are in `training/README.md`.

## Build the bundled tracker

This optional section is for creating a distributable Windows executable; it is not required to run the application from Python. Install PyInstaller into the same environment first:

```powershell
.\.venv\Scripts\python.exe -m pip install PyInstaller
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean TrajectoryTracker.spec
```

Deploy the complete generated `dist/TrajectoryTracker` folder. Do not combine a newly built executable with an older `_internal` runtime; the executable and runtime must come from the same build.
