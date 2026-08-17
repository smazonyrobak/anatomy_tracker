# Reproducibility contract

## Frozen starting point

The proposed joint-model work begins from Git commit `c6681039e0b7acf35c9cdbee43040a3dca29cdab` on branch `codex/joint-registration`. Baseline evidence is recorded in [`docs/publication/BASELINE_LEDGER.md`](docs/publication/BASELINE_LEDGER.md). Documentation changes do not alter the baseline software.

## Required run record

Every training/evaluation run must preserve:

- repository commit and clean/dirty status;
- diff or patch when dirty;
- resolved configuration;
- Python, PyTorch, CUDA/cuDNN, NumPy, ONNX and ONNX Runtime versions;
- OS, CPU, GPU and RAM;
- all random seeds and deterministic settings;
- atlas, ontology, input data and split-manifest hashes;
- synthetic generator contract/hash;
- pretrained initialization source, license and hash;
- checkpoint and EMA payload hashes;
- evaluator source and environment hashes;
- start/end timestamps, wall time and interruption/resume history.

## Data reconstruction

Synthetic images are regenerated from versioned manifests rather than stored as an undocumented image pile. A manifest must reproduce pose, geometry, appearance, masks and wrong-plane candidates byte-for-byte or within a documented provider tolerance. Real-data downloads retain upstream identifiers, URLs, access dates, response/checksum receipts and exclusion decisions.

## Determinism

Short CPU/GPU fixtures establish:

- repeated-run equality under deterministic settings;
- uninterrupted versus resumed training equivalence;
- worker-count independence of manifest assignment;
- atomic checkpoint recovery;
- deterministic hidden evaluation from a frozen prediction file.

Full GPU training may have documented nondeterministic operations only if no practical deterministic alternative exists; three or more independent seeds then quantify variability.

## Benchmark discipline

- Animal-level manifests are frozen before training.
- Hidden labels are inaccessible to model developers when feasible.
- Final evaluation reads predictions and writes a receipt; it does not train or select.
- The published DeepSlice set is marked public/development-exposed.
- Failed runs and test consumptions are retained.
- Any amendment states date, reason, affected benchmark and whether labels/results had been accessed.

## Comparator reproduction

Each comparator has a container/environment, exact revision/weight hash, preprocessing contract and invocation in [`publication/comparators.lock.yaml`](publication/comparators.lock.yaml). When a method supports multiple information modes, automatic, series-assisted and user-assisted configurations are run separately. Comparator failure is documented, not imputed.

## Export and desktop parity

The selected checkpoint is exported once as initializer and recurrent-refiner ONNX entry graphs connected by the deterministic host atlas renderer. PyTorch, ONNX CPU and DirectML outputs are compared on identity, known affine/flow, flip/orientation, normal, severe-artifact and boundary cases, at batch sizes one and ten. The exact shipped checkpoint, graphs and runtime asset SHA-256 values appear in setup metadata. A GUI end-to-end receipt records the model, atlas, session, provider and outputs used for every validation case.

## Publication archive

The publication archive contains:

- source tag/commit and repository snapshot;
- model weights and model card;
- training/evaluation manifests and data card;
- machine-readable protocol/gates/comparator lock;
- raw per-case predictions and exclusions;
- bootstrap indices or seeds and analysis scripts;
- final tables/figures generated from raw results;
- logs, environment lock and hardware/runtime report;
- citations and upstream license notices.

Numbers in prose or tables must be reproducible from archived raw outputs. Manual transcription is checked against the machine-readable result ledger before submission.
