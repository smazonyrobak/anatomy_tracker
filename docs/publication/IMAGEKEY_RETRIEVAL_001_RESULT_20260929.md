# Image-key retrieval001: candidate-stage gate passes, alignment remains inadequate

Run `I:/AnatomyTracker/runs/joint_v6_imagekey_retrieval_001` completed all4,000
updates, terminal8803 EXIT0, from source `edc99b8`. Training took583.409s;
total719.618s. It used32,000 original generated queries and32,000 frozen
presentations, without new query observations or prior model weights.
The [predeclared protocol](IMAGEKEY_RETRIEVAL_PROTOCOL_20260929.md) is unchanged.

Independent CPU audit45886 exited0 after55.055s. Integrity and every fixed
performance gate pass. The whole checkpoint is selected for a conditional native
joint-refinement pilot, not deployment, calibration or public benchmarking.

## Frozen comparison

Eligible equal-organizational-group means:611 observations,40 groups. These
groups come from one atlas and are **not biological animals**. All640 rows and
the29 low-support censored rows remain in raw outputs and separate summaries.
The comparator is original003 **step4,000**, not its20,000-step parent.

| Metric | Original003 at4k | Image-key001 at4k |
|---|---:|---:|
| MAP antipodal normal error |47.49795deg|40.35802deg|
| Top32 physical-plane capture |10.59364%|68.46016%|
| Top128 physical-plane capture |21.62363%|85.54190%|

Physical capture means one and the same score-selected candidate has normal
error<=10deg and sign-aligned plane-offset error<=500um. It does not measure a
correct finite image frame, tissue registration or an electrode position.

Image-key MAP normal-offset error is1945.20um, antipodal full-frame angle66.64deg,
and continuous-canvas four-corner RMS error7513.03um. Full-frame corner capture
within1000um is only4.27930% at top32 and5.28560% at top128. The large gap from
plane capture must be addressed by honest image-selected finite-frame inference;
a truth-near local pilot cannot establish that capability.

Full-cell NLL is7.58977; exact-cell recall is2.78011%/14.70856%/32.38290%/52.73684%
at top1/8/32/128. Normal-marginal MAP error is39.33624deg. None of the probability
values are calibrated confidence claims.

| Eligible input mode | Rows | MAP normal | Top32 plane capture | Top128 plane capture |
|---|---:|---:|---:|---:|
| No brush |207|43.01576deg|66.45833%|85.66667%|
| Exact black exterior |204|38.80001deg|73.66667%|89.16667%|
| Imperfect brush |200|39.52705deg|65.70833%|82.20833%|

The fixed overall gates required at least5deg normal improvement and .05 absolute
top32 capture improvement. Observed improvements are7.13993deg and .5786653.
Every populated eligible input mode also passes the fixed non-regression gates.
This is an architecture/objective mechanism comparison: FP32 image-key training
and original003 AMP differ in precision and compute as disclosed before launch.

## Audit and exact bindings

`training/audit_joint_v6_imagekey_retrieval.py` independently reconstructed every
640x98,304 posterior in NumPy FP64 from stored query/key descriptors and priors
applied once. Maximum log-probability difference8.9332e-6 is below the5e-5 gate.
Stable lower-ID ties/topK were reconstructed from saved FP32 scores; physical
geometry and group/mode gates were independently recomputed. The audit checks
the complete frozen file inventory, prepared inputs, source, schedules, candidate
unions, update counts and checkpoint state changes. It does not re-encode images,
re-render the atlas or replay every mined-neighbour search.

Outputs: `I:/AnatomyTracker/runs/joint_v6_imagekey_retrieval_001_independent_audit`.
SHA256 bindings:

- Audit JSON: `f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b`.
- Whole step4,000 checkpoint: `d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5`.
- Matching gallery: `1568da8f1347641bba4ff93679ff39d5d6250693345db5d915322312bbe64666`.
- Run completion: `141ef87c004ecd30a8268a48ec97850280c01ea3ebf94d825410a991305b90df`.

## Next experiment, not qualification

Continue this whole fresh lineage with the native recurrent curved-ribbon path;
do not merge an independently trained encoder. Prepare8 training and4 development
coherent synthetic subject maps, retain all sampled planes and censor low-support
localization losses explicitly. First test known-PSF truth-near pose/surface
learning, with frozen retrieval features as a control. Separately measure raw
Allen donor-held-out transfer and the finite-frame candidate gap. These are
internal development diagnostics, not untouched biological final tests.

The current40.36deg top-choice orientation error remains unacceptable. Accurate
global joint capture, acquired-background robustness, native constraints,
calibrated joint trajectory/site probabilities, fair frozen external comparison
and desktop integration are still unfinished.
