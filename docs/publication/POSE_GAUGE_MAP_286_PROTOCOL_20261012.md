# 286: matched pose-gauge map-learning pilot (frozen before training)

## Question

Does fitting the rigid plane to the final intact tissue, while leaving the actual image and every pixel-to-CCF coordinate unchanged, improve image-conditioned local mapping over the v9 pre-artifact pose convention? Audit 285 found that a median 49.51% of intact-tissue squared displacement lies in a support-restricted affine component. This is an optimization/identifiability concern, **not** a mathematical lower bound on the existing full-canvas affine-free field: oracle 281 fitted such a field on individual slices.

## Matched comparison

- Use the frozen scratch-lineage 275 step-8,000 image encoder and lateral features. Both arms start from the same scratch-initialized mapper parameters (the 282/284 initialization seed); freeze the pose encoder/head and train only the same map modules as 284. Use the true synthetic pose and reflection in `map` to isolate conditional deformation learning; this is not a blind or deployable model.
- Arm A receives each v9 image with its unchanged pre-artifact pose. Arm B receives the **same bytes** for its image, pixel-to-CCF target, validity mask, reflection, PSF offsets/weights, and section identity, but receives the `postartifact_gauge_v1` intact-tissue affine pose. The refit excludes invalid and displaced-fragment pixels. No training image is presented with alternative artifacts/background, and each independently drawn physical-section ID occurs once per arm. Reuse of the *identical* section across arms is required for a controlled comparison, not paired appearance augmentation.
- Both arms see the same 3,000 accepted, unique TRAIN sections from deformation bases 0–55, one section per optimizer batch, same ordered schedule and site coordinates. Require at least 512 valid sites; use 512 fixed sites per section. AdamW learning rate `1e-4`, weight decay `1e-4`, gradient clip 2, identical initialization and no hyperparameter retuning between arms.
- Fix 64 new eligible TRAIN-inner sections before training: eight independently drawn sections per held-gradient base 56–63. Both arms read the same image/coordinates and 512 sites. DEV/CAL/FINAL animals, real-data labels, and public benchmarks are untouched. Save row-level metrics and weights at steps 0, 750, 1,500, 3,000.
- The model renders a finite-thickness atlas candidate and predicts a 16×16 smooth local field, upsampled to 64×64. The field is affine-free over the full canvas and has maximum physical derivative 0.35. The refit normal differs slightly from the original through-plane director; `postartifact_gauge_v1` retains the original PSF offsets/weights and records this geometric caveat. Existing v10/v11 appearance generators remain untouched.

## Readout and decision

For every fixed readout section and arm, report exact-state same-grid no-warp and mapped mean Euclidean surviving-site CCF error in mm, incremental map gain, predicted surviving-site shift, pre/post derivative, visible support, artifact flags, base and source ID. Report both arm-level mean and per-base means, plus pose-label centre/normal change. The primary **mechanism** readout is fixed step 3,000, not a selected best step.

The gauge-change signal passes only if arm B's map gain is at least 0.020 mm, exceeds arm A's map gain by at least 0.010 mm, has nonnegative gain on at least six of eight inner bases, and both arms obey derivative ≤0.35001. Report arm B's absolute final error relative to A separately; a lower no-warp baseline alone does not pass the map-learning gate. A pass supports further end-to-end training but cannot validate arbitrary-plane real transfer, uncertainty, electrode regions, or GUI replacement. A failure directs the next mechanism test toward learnable source/atlas correspondence features, not a larger copy of the frozen-feature training schedule.

All draw attempts, unique IDs, shared input/target hashes, source/protocol/parent hashes, checkpoints, readout rows, and gate arithmetic are frozen under one I: run directory. An independent read-only audit must pass before scientific interpretation. Do not alter the source, gates, or frozen outputs after launch.
