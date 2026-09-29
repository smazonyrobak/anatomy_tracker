# Independent post-exit image-key001 audit

Do not run `training/audit_joint_v6_imagekey_retrieval.py` until the image-key001
process has exited and root authorizes frozen-output access. It reads that run
only then, writes a separate `I:/AnatomyTracker/runs/joint_v6_imagekey_retrieval_001_independent_audit`
directory, and never changes frozen outputs. Use the I: Python environment with
`-B -u -m training.audit_joint_v6_imagekey_retrieval`. No model or GPU executes;
Torch is used only to deserialize CPU tensors/checkpoints.

The flat audit authenticates source snapshots, the fixed driver, baseline,
prepared data, catalogue, schedules, descriptor gallery and final checkpoint.
It checks the original5120/640 identities and eligibility, five-key split
separation, 4,000 applied updates, all replay indices, candidate unions containing
every batch truth, generated support censoring, whole-model/optimizer/RNG state,
and unchanged untrained parameters. The 32,000 generated and 32,000 frozen query
presentations are replays, not new observations. Source/checkpoint binding is
verified; this is not an independent re-encoding of the gallery or an exhaustive
replay of mined nearest-neighbour pools.

Only after exit, every file in the frozen run is inventoried and hashed,
including both endpoint banks/raw probabilities/metrics and the first generated
batch. Both checkpoint experiments must match the configuration; step-zero
checkpoint/gallery hashes are checked against their saved receipt. No repeated
zero-step numerical evaluation is required.

NumPy re-normalizes the saved query and FP16 gallery descriptors and reconstructs
all640×98,304 cell log probabilities using cosine/0.1, both representation priors
and cell mass exactly once. No training-time proximity exclusions or truth
injection occur in this reconstruction. Predetermined arithmetic tolerances are
5e-5 maximum descriptor-replay log-probability/representation-probability error
and2e-5 raw log-normalization error, allowing CPU FP64 versus GPU FP32 rounding.
Stable lower-index ties, truth ranks, normal marginals and top128 selections are
recomputed; runner evaluation functions and saved aggregate scores are not used.

Physical geometry is independently decoded in NumPy FP64 from saved full-frame
states. Capture requires a single topK candidate to satisfy both antipodal normal
angle<=10deg and sign-aligned offset<=500µm. Corner-based frame diagnostics are
separate. Original positive-weight eligibility and its intersection with every
brush mode are used before equal organizational-group averaging; all/censored
rows remain reported. No new image-content filtering or segmentation is added.

Against the frozen original003 step4000 report, advancement requires at least
5deg eligible normal improvement and .05 absolute top32 plane-capture improvement,
with each eligible mode regressing no more than2deg and .02 respectively. These
predeclared engineering gates remain separate from artifact-integrity status.
Both must pass to motivate joint integration; neither establishes biological
generalization, calibration, global joint registration, DeepSlice superiority
or a shippable model. The auditor is prepared source-only before run completion;
review and explicit post-exit execution remain pending.
