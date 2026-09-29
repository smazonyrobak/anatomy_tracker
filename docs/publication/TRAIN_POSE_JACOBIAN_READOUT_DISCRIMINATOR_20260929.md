# Draft: fixed-feature TRAIN pose-Jacobian/readout discriminator

Status: **unlaunched; pending the normalized control's completed endpoint and a separate launch decision**. One bounded diagnostic, no optimizer, parameter sweep, development examples, model edit, benchmark or qualification claim.

## Frozen inputs

Load the complete unscaled signed checkpoint `I:/AnatomyTracker/runs/joint_v6_signed_pose_evidence_001/joint_model_step_02000.pt`, SHA256 `555c129b2cbe5832b37c7642d5ee15e31c1ba57199952901959751918cae6136`, using its archived source and exact model kwargs. Its independent audit SHA256 is `7acdc3ef87f85691b296f67fc828428aa090b1afed88a5ce05853a6e81853ad5`; the retrieval feature tensors remain equal to the whole original4000 parent. No module merging, new weights or normalized-control weights.

Reuse the exact24 TRAIN observation IDs and saved geometry from `I:/AnatomyTracker/runs/joint_v6_train_pose_cost_direction_001/completed.json`, SHA256 `556fc13ee2f6dfc1db059c3ce8e5a3b00eca429b2de23c394be19e552cdc5ed5`, authenticating its listed artifacts. Subjects00000000..00000007 use section indices `[1,0,2,0,0,1,0,2]` in the `joint-v6-coherent-sections-002-train-subject-XXXXXXXX-section-XXXXXXXX` namespace, each with raw/exact_black/imperfect_brush presentations. Retain recorded eligibility and IDs; do not reselect sections. Reuse the six saved starts per observation: normal-tangent u/v at ±6° and offset at ±300µm. This gives144 paired trajectories, not144 independent animals.

Use saved canonical affine-free residual and local director delta, correct recorded reflection and normalized finite PSF. Hold these local coefficients fixed throughout both arms, while the current proper pose frame maps them into CCF. No field prediction, refit, extra projection, probe-specific limiter or second reflection. Query features use only original image/outline/availability. No support-mask weights enter either solver or evidence.

## Three-step comparison

At every current pose, render central and all six ±3°/±150µm probes with the same fixed ribbon, PSF and one spatial reflection. Encode with the same checkpoint. Let `f` and `a` be channel-normalized source/atlas features (`eps=1e-6`), `r=vec(a-f)` and `h=(pi/60,pi/60,150)`. Perturbations use the existing support-origin-aware pose composition, not an image-space warp. In dimensionless coordinates,

`J[:,i] = vec(a(theta ⊕ h[i]*e[i]) - a(theta ⊕ -h[i]*e[i])) / 2`.

With `M=len(r)`, set `H=J.T@J/M`, `g=J.T@r/M`, `m=trace(H)/3`, then solve

`q = -solve(H + 0.1*diag(diag(H)) + 1e-6*m*I, g)`.

Do not invert the matrix explicitly. A zero-information `m=0` or nonfinite solve is recorded as unavailable/no update, not repaired with a tuned constant. Cap `q` by `q/max(1,||q||2)` and apply only the first three physical increments `h*q`; the other six increments are zero. Recompute all features/Jacobians at each of three steps. This is forward finite-difference damped least squares, not an exact inverse-compositional or exact analytic Gauss–Newton method.

For the learned arm, reproduce the archived central pair evidence, actual observed-centre coordinate addition, six unscaled signed-cost additions, ConvGRU and bounded nine-coordinate head. Keep its own hidden state across three updates, starting from zero. Take only the first three bounded outputs; zero the other six, convert to dimensionless units and apply the **same unit-norm cap**. Save pre-cap outputs too. Both arms use identical operators; their poses/hidden histories may diverge after step1. Final rendering measures step3 without a fourth update. Fixed fields bypass the field head and pose-only prefix in both arms.

## Feasibility, records and interpretation

Archived `refine_ribbon` regenerates fields internally and cannot accept these oracle fields. A small separate flat diagnostic loop must reuse its existing operators; replacing fields after `refine_ribbon` returns would compare the wrong evidence. No operative-model edits or head monkey-patching are needed. Use FP64 geometry/3×3 solves and FP32 rendering/features, no AMP/TF32. At most144×3×7=3024 iterative renders plus final/truth renders, batching paired modes where possible; no new data generation.

Save both state/update trajectories, cosine cost and half squared feature-residual cost, centre and PSF-weighted slab errors against frozen coordinates, physical normal/offset errors, undamped eigenvalues/damped condition numbers, `m`, trust-cap/zero-information flags, and truth-frame cost. Report full-canvas geometry and paired distributions by mode/start-axis/subject with unchanged censor flags. Save feature-normalization epsilon cases because cosine and half squared residual costs coincide only for genuinely unit-normalized features. No oracle-support-weighted solver or result-dependent exclusions.

Geometry and cost improvement beyond the learned arm supports a readout limitation **on these oracle-conditioned starts**. Cost improvement without geometry improvement suggests metric ambiguity/confounding. Failure to lower cost can reflect linearization, damping or conditioning and does not establish metric failure. Nonzero oracle fields during the trained pose-only prefix are off-policy; results do not certify the original recurrent rollout, multiaxis/global capture, deformation learning or biological generalization. Hessian conditioning is not calibrated uncertainty. No existing gate changes.

## Primary motivation

[CLKN, CVPR2017 §2.2](https://openaccess.thecvf.com/content_cvpr_2017/html/Chang_CLKN_Cascaded_Lucas-Kanade_CVPR_2017_paper.html) motivates differentiable feature-residual pose solves; its homography inverse-compositional Jacobian is not reused here. [BA-Net, ICLR2019 §4.3](https://arxiv.org/html/1806.04807) motivates fixed-length damped optimization within a joint learned model, not our chosen fixed damping value. [PointNetLK, CVPR2019 §3.2](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aoki_PointNetLK_Robust__Efficient_Point_Cloud_Registration_Using_PointNet_CVPR_2019_paper.pdf) supports finite-difference feature Jacobians; unlike its template Jacobian, ours must be recomputed after each fresh finite-thickness atlas render. These are design precedents, not histology-performance evidence.
