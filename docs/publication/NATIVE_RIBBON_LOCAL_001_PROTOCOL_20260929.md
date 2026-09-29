# Native pose-and-ribbon local control 001 — source prepared, not run

Driver: `training/run_joint_v6_ribbon_local.py`. Future output: `I:/AnatomyTracker/runs/joint_v6_ribbon_local_001`. The section-cohort completion hash remains **UNSET** and is asserted before any input read. The image-key parent has now exited and passed its independent promotion audit: checkpoint SHA-256 `d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5`; audit SHA-256 `f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b`, requiring `advance_to_joint_integration_pilot=true`.

This is a **conditional truth-near joint pose+surface control**, not successful global capture, a deployable tracker, calibrated uncertainty or a benchmark. The parent's top-32 finite-frame capture within 1 mm is only 4.28%; a local oracle initializer must not hide that remaining global-placement problem.

## Initialization and training

Load the **whole** audited image-key checkpoint into one joint model; append only fresh zero-initialized coordinate-evidence and six-channel ribbon heads. Exact-load all prior tensors—no encoder merge or imported model weights. Freeze both image/atlas stems, shared encoder, descriptor, global proposal and all other tensors except the refinement pair encoder, shared GRU, 9D pose update, final score, coordinate and ribbon heads. This freeze is the first control, not a permanent restriction on the joint architecture. Legacy 2D SVF and 35D covariance paths are unused.

Read each completed raw section artifact once; keep shared physical-section targets and per-observation channels/support in memory, with **no extra serialized training pack**. There are 512 training and 128 held-out physical sections, each with three paired modes. Preserve every identity/hash, all censored rows and their predictions. Training selects one of the eight subjects uniformly, then an eligible observation uniformly within that subject. A subject with no eligible observation is an explicit failure, not silently dropped. Development retains all 384 observations; no development subject appears in training. These are synthetic subjects from one atlas, not biological animals.

Seed **2026092913**; 2,000 updates, batch 4, FP32 without AMP/TF32, AdamW LR 2e-4/weight decay 1e-4, gradient clipping 1. Three native recurrent updates; pose-only prefix 1. Sample independent uniform signed perturbations in coordinates `(normal tangent u,v rad; normal offset µm; roll rad; in-plane u,v µm; log basis u,v; shear)`:

- First 250 updates: bounds `[.06,.06,250,.08,300,300,.05,.05,.05]`.
- Remaining updates and development: `[.15,.15,600,.20,600,600,.12,.12,.12]`.

Development perturbation is fixed **per physical section** and shared by all three modes. All schedules/indices/seeds are persisted. Inputs contain only frozen image/boundary/availability, the perturbed start and known PSF. Both reflection branches receive equal prior mass; reflection truth enters only losses and oracle diagnostics.

## Losses

Let `q(x)` be frozen **visible finite support**, `Q=sum q`, and `w_s` the once-normalized physical PSF weights. Only the recorded reflection branch receives geometric supervision. Row losses are averaged over eligible rows, not diluted by censored zeros.

1. For iterations 1–3, centre-coordinate squared Euclidean error `sum q||C−C*||²/(Q·100²)`, combined with normalized weights `[.64,.8,1]`. Exclude the iteration-0 initializer.
2. Final slab error `sum_s,x q w_s ||X_s−X*_s||²/(Q·100²)`. This is the PSF-weighted **error**, not error after averaging coordinates; no double PSF integration.
3. Weak 0.1 full-canvas canonical fitted-plane gauge error, squared Euclidean/100². This is canonical anatomy's fit, not a reflected observed-total fit.
4. Weak 0.1 final world-director squared error/0.05², weighted by q. Target is exact-centre-anchored PSF-weighted least squares `D*=sum_s w_s z_s(X*_s−C*) / sum_s w_s z_s²`. Predict `n+R d` and reflect only spatial indices; never compare local coefficients from different frames. Slab/director terms apply only when the deformation stage is active.
5. Final reflection cross-entropy, with the model's one final conditional normalization.

No image-derived mask, annotation-purity suppression, old 2D-SVF label, or covariance likelihood is substituted for exact physical coordinates. Original FP64 coordinate artifacts stay immutable; this control trains on their FP32 copies.

## Endpoints and fixed gate

Evaluate all held-out observations at steps 0 and 2,000. The geometric initializer is the actual perturbed plane with zero fields: selected baseline uses the deterministic **identity** tie of the uniform prior; oracle baseline uses the recorded reflection. The step-zero recurrent model is reported separately, not relabelled as that initializer. Final selected readout uses the model's final component score; oracle readout uses the correct reflection. Never use the final selected flag retrospectively for initial geometry.

Report visibility/PSF-weighted point-distance means and weighted point p95, fitted-plane-normal/tangential error components, director and plane-normal error, reflection accuracy/NLL, and full-canvas diagnostics. A macro p95 is explicitly the mean of per-observation weighted point p95 values, not a pooled calibrated percentile. Zero-visible-mass errors are undefined; censored/full-canvas diagnostics do not imply identifiable localization. Report eligible/all/censored, each mode, each held-out subject and their intersections, with exact counts.

Both **selected and oracle** centre and slab subject-macro means must improve by **at least 20%** over their respective geometric initializers; no eligible mode or held-out subject mean may regress. Empty required groups fail the gate. For all rows, branches and iterations, require finite predictions, positive proper-frame determinants and positive relative ribbon Jacobians measured at every bilinear-cell corner and both z extrema, plus the derivative bound ≤0.35001. Canonical topology is checked **before** raster reflection. Continuous injectivity relies on the primitive's bounded-interpolant contract, not sampled determinants alone. Record limiter rescaling, raw tanh surface/director saturation and pose-update saturation.

Full raw branch predictions and baseline states/metrics are saved in small development batch files with hashes; checkpoints at 0/500/1,000/1,500/2,000 retain full model, optimizer and RNG states. Exact equality of every frozen state tensor/buffer is checked at checkpoints and endpoints. This preserves global retrieval-producing tensors without an unnecessary gallery rebuild; it makes no claim that joint inference or global capture is unchanged. Final source hashes and completion are emitted only after the run ends successfully. All output, temporary files and Torch/CUDA caches use I:.
