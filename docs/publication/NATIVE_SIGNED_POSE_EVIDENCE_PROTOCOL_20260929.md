# Prepared control: signed native pose evidence

Status: **completed and independently audited; predeclared gate failed**. Matched training from `0e6dc0d` (terminal70497) and CPU audit4417 both exited0; integrity passed, but oracle centre/slab and normal gains failed the fixed20% criteria. This is not a qualified model. See the [completed result and exact audit scope](NATIVE_SIGNED_POSE_EVIDENCE_001_RESULT_20260929.md). All development stays on I:; the protocol below remains the frozen prospective contract.

## Completed compatibility preflight

Implementation commit `0e93978`; terminal27377 EXIT0. The actual first scheduled B4 TRAIN batch was used with no optimizer update or development examples. Common fresh parameters and CPU RNG, all zero-projection outputs, and unchanged post-backward parameters matched exactly. The new projection's gradient norm was0.0138903223; pose and ribbon gradients were finite. Cost-map gradients are zero at this deliberately zero projection, not a failed gradient path. The instrumented forward/backward took3.5372s with peak allocation1,372,437,504bytes. Forward encoder calls were4 without probes and22 with probes. This is a compatibility result, not a throughput estimate or quality claim.

Frozen preflight: `I:/AnatomyTracker/runs/joint_v6_signed_pose_evidence_preflight_001/preflight.json`; SHA-256 `4e5a2fd3a9697da6a87c435abf35f337bb383e596c46abaa2cc2560e0a39a76e`. The matched driver pins this receipt and the original comparator's source and exact schedule bytes.

## Question and fixed comparator

Does explicit signed out-of-plane image evidence enable the existing shared updater to correct plane orientation, where its current in-plane displacement correlations did not?

The completed `joint_v6_ribbon_local_001` is the comparator (completion SHA-256 `a2bee6dc1ffcaafcef3b4ad4a49223653f582f41ab071a3ca25928e7db7a83d7`). Its oracle plane-normal mean changed only from about 6.138 to 6.103 degrees. Static review found no obvious detached pose path, incorrect normal-update index, or omitted normal supervision: centre/slab coordinate losses and the canonical full-canvas gauge loss already penalize normal error. This is not a numerical proof that the loss or optimizer is optimal. The existing radius-2, stride-4 correlation volume, spatial ConvGRU and coordinate/reflection evidence remain present.

Use the comparator's archived source/configuration and exact frozen cohort, identities, row/subject sampling, perturbation schedule and evaluation rows. Initialize from the same **whole** original image-key checkpoint, not the trained local comparator or a merged encoder:

- Parent: `I:/AnatomyTracker/runs/joint_v6_imagekey_retrieval_001/joint_model_step_04000.pt`.
- Parent SHA-256: `d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5`.
- Parent independent-audit SHA-256: `f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b`.
- Cohort completion SHA-256: `ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2`.

Before any later launch, bind the comparator's actual archived source and schedule hashes in the new receipt; do not substitute whichever model source is then current. The only model change permitted by this control is the opt-in evidence injection below.

## One intervention

For each existing `(sample, supplied cell, raster representation)` hypothesis, before each of the three pose updates:

1. Retain the central render and current canonical ribbon coefficients. Compose six states from the **current predicted canonical state** using the existing update convention: positive/negative 3 degrees along each of its two normal-tangent axes, and positive/negative 150 micrometres along its normal-offset coordinate. Other update coordinates are zero. These probe steps are fixed inputs, not fitted from development results.
2. Render each probe with the same finite-thickness PSF offsets/normalized weights and the same current canonical ribbon coefficients. Do not run another ribbon prediction for each probe. Holding coefficients fixed across the six probes means taking a pose response at the current deformation; it does **not** mean detaching the current state or coefficients from training gradients.
3. Apply the representation's spatial reflection exactly once, after canonical ribbon construction, using the same observed-raster convention as the central render. Never reverse a director component or normal because the raster is reflected. Do not perturb an already-reflected observed coordinate grid as though it were the canonical physical state.
4. Encode all renders with the existing atlas encoder. At each feature location, use the existing channel-normalized source/atlas features to form `C = 1 - dot(source, atlas)`. Supply the six dimensionless maps `C_u+ - C_0`, `C_u- - C_0`, `C_v+ - C_0`, `C_v- - C_0`, `C_d+ - C_0`, `C_d- - C_0` in this fixed axis/sign order.
5. Add a bias-free, zero-initialized `1x1` projection from six channels to the existing hidden width to the current pair evidence **before the existing GRU**. Keep the central correlation, physical-coordinate evidence, reflection handling, shared update head and ribbon head unchanged. No extra encoder, independent registration model, free-flow state, hard cost argmin or externally learned feature is introduced.

Probe renders/features and their differences remain differentiable through the current state and renderer; freezing encoder parameters must not disable gradients with respect to their input. There is no ground-truth pose, tissue-support mask, reflection label or target coordinate in these evidence channels. Inference uses precisely the same channels. No division mixes radians and micrometres: probe magnitudes are declared separately and all six supplied quantities are cost differences.

No extra probes are needed for the final score-only re-render. Thus the change costs **six additional renders per pose update**, eighteen additional renders for three updates; the existing four central renders remain. It is a learned finite pose-cost neighbourhood, **not** a mathematical Gauss–Newton or Lucas–Kanade update and not a convergence guarantee.

## Unchanged learning and evaluation contract

Retain seed `2026092913`, 2,000 applied updates, batch four, three shared recurrent updates and one pose-only prefix step. Use the comparator's first-250 small-perturbation schedule and subsequent large-perturbation schedule, uniform training-subject then eligible-observation sampling, known synthetic PSFs, identity section processing, and equal initial reflection masses. Retain FP32 with AMP/TF32 disabled, fresh AdamW at `2e-4`, weight decay `1e-4` and gradient-norm cap one.

Retain the existing correct-reflection-branch supervision: normalized `0.8`-decayed per-iteration centre squared error; final PSF-weighted slab squared error; `0.1` full-canvas canonical fitted-plane gauge; `0.1` world-director error; and reflection cross-entropy. Coordinate squared errors use the original 100-micrometre scale, director error the original `0.05` scale, and each row's frozen visible-support weight is normalized once. Do not add a normal loss, change weights or unfreeze feature encoders in this control. The original trainable refinement/ribbon modules plus the new projection are trainable; retrieval-producing tensors stay frozen exactly. Uncertainty remains disabled and uncalibrated.

Preserve zero-step evaluation and the same final full development evaluation, including censored rows in saved outputs and separate selected/oracle readouts. The geometric initializer is the identical perturbed pose with zero fields and uniform reflection prior. Truth-near initialization and known synthetic PSF remain explicit limitations; this is not a global-capture evaluation.

## Prospective success gate

All conditions must hold, computed from saved unrounded per-row measurements with the comparator's eligibility and held-out-subject macro aggregation:

- At least **20% reduction versus the geometric initializer** in both centre and PSF-weighted slab mean error, for **both selected and oracle** readouts.
- At least **20% reduction versus the initializer** in oracle antipodal fitted-plane normal-angle mean. A reflection-selection gain alone cannot meet this condition.
- No mean regression versus the initializer for those dense metrics or oracle normal error in any populated eligible presentation mode or held-out synthetic subject.
- The original all-row/all-branch topology/orientation requirements pass, and frozen retrieval tensors are exactly unchanged.

Also report differences from the completed no-probe comparator; do not equate meeting an initializer-relative gate with demonstrated superiority over that comparator. Save per-iteration pose changes, probe step sizes, raw branch predictions and normal/coordinate metrics so signed orientation learning can be distinguished from reflection or translation improvements. Do not alter the frozen comparator's gate or results.

Only a fixed-small-batch runtime/gradient compatibility preflight is permitted before a separately committed launch after the active comparison exits: same completed cohort rows, no optimizer step or new data, check central/default initialization compatibility, finite gradients including the new projection, and peak memory/runtime for the six added probes. It is not another experiment or benchmark. Preserve existing parameters/RNG draws by initializing the optional zero projection in an isolated RNG scope. The user's existing autonomous-development authorization covers this work; no further user prompt is required.

## Rationale and limits

[CLKN, section 2](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chang_CLKN_Cascaded_Lucas-Kanade_CVPR_2017_paper.pdf) combines learned feature alignment with a differentiable geometric Lucas–Kanade layer. It motivates exposing feature-space geometric response, not a claim that this proposed six-channel injection implements its solver. [SVoRT, section 2.1](https://arxiv.org/html/2206.10802) renders PSF-blurred slices at current transformations and uses acquired/rendered evidence to predict residual transforms; its multi-slice context and task differ from this single-section native ribbon control.

The hypothesis is that the present local in-plane cost volume does not efficiently expose the **sign** of out-of-plane pose corrections. Frozen retrieval features may nevertheless be a poor local metric, partial sections may remain ambiguous, and pose/deformation or appearance effects can flatten or distort probe costs. Probe evidence cannot manufacture missing anatomical information, establish biological generalization, calibrate probabilities, or qualify deployment. No public benchmark or GUI release is part of this control.
