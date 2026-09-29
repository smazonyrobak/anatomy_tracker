# Fresh image-key retrieval control 001

Launched after rehearsal003's failed local-gate audit, from committed source
`edc99b8`, terminal8803. Driver: `training/run_joint_v6_imagekey_retrieval.py`.
The protocol below was fixed before launch. Outcome remains unmeasured; do not
access its output tree or alter its operative source while the process is active.
Launch only after joint-rehearsal003 exits and its independent result is audited.
Output: `I:/AnatomyTracker/runs/joint_v6_imagekey_retrieval_001`.

This tests whether comparing learned histology descriptors with actual rendered
atlas images gives better global pose candidates than the previous parametric
geometry head. It is not an extension of a failed checkpoint or a new app.

## Fixed intervention

One whole joint model starts randomly, seed2026092805, with the same original
parameters as original003 and the opt-in shared 256D image descriptor. Only
histology stem, atlas stem, shared encoder and descriptor train. Recurrent pose,
deformation and old parametric proposal remain untrained; the resulting checkpoint
is an experimental coarse backend, not an integrated deliverable.

Run4,000 updates with the exact original003 first32,000 generated observations
and32,000 frozen presentations, preserving original IDs, appearance/PSF schedules,
low-support censoring and split membership. No new query observations, learned
features, weights or pseudolabels are imported. Prepared5120 train/640 development
rows remain unchanged. AdamW .001, weight decay .0001, clip5, FP32. Original003
used AMP, so precision and compute differ along with the architecture/objective;
this is a mechanism comparison, not a single-factor attribution.

Each query batch has8 generated and8 frozen inputs. Candidate cells are the
deduplicated union of batch truth cells,32 uniform catalogue draws, and one local
negative per query, chosen uniformly from its8 nearest admissible neighbours.
There are at most64 cells. Duplicate draws are not refilled. Separate negative
seed2026092910 and exact global/local-rank schedules are saved before learning.

For a cell's actual96x96 pixel footprint, define
`mu=O+(95/192)*(U+V)`, `a=(95/96)*U`, `b=(95/96)*V`.
Finite four-corner squared distance between cells is
`||mu_j-mu_y||² + (min(||a_j-a_y||²,||a_j+a_y||²)+||b_j-b_y||²)/4`.
For each query, omit non-truth candidates satisfying **both** antipodal normal
angle<=10deg and this distance<=1000um. Use the same exclusion to construct its
local-negative pool. Own truth is always included. Neighbours are not promoted
to positives, and anatomical ML reflection is never treated as equivalence.
These are fixed engineering choices, not a claim of anatomical interchangeability.

Render both identity and horizontal pixel-permuted keys before encoding. The
bound affine uses centered grid coordinates (zero translation), corresponding to
`x -> 95-x`, not a physical brain reflection. Keys are freshly encoded with the
current shared weights;16-image non-reentrant checkpoint chunks retain both-side
gradients. All key renders use a declared50um normalized nine-sample trapezoidal
PSF. Queries retain their original varying thickness; this mismatch is explicit.

Unit-descriptor cosine scores use fixed temperature .1. Cell score is
`log(cell_mass) + logsumexp_r(cosine_r/.1 + log(representation_prior_r))`.
Priors occur once. The weighted cell NLL normalizes over each query's retained
sampled candidates. Exclusive identity/reflection alternatives are marginalized
inside the logarithm, not averaged as descriptors or forced to both match.
No naive importance correction is applied to mixed forced-positive/mined sampling.
This sampled discrimination loss is not an unbiased complete-catalogue likelihood
and does not train calibrated probabilities.

## Frozen endpoints and decision

At steps0 and4,000 rebuild all98,304 cells with both representations from current
weights. No truth cells or proximity exclusions are injected at evaluation.
Store the FP16 descriptor bank, FP32 full-catalogue log probabilities, query
descriptors, top128 cells and representation conditionals, checkpoint/optimizer/RNG,
source snapshots, exact schedules and receipts binding bank to weights/catalogue/PSF.
Scoring re-normalizes stored descriptors in FP32. The coarse prepared rows do not
carry authenticated reflection labels; save their conditionals without claiming
reflection accuracy or calibration.

Compare with the independently recomputed original003 step4,000 baseline in
`IMAGEKEY_BASELINE_003_STEP4000_20260929.md` and its frozen JSON hash. Both runs
must have4,000 applied updates. Use continuous reference states, stable lower-ID
score ties, positive-weight eligibility, and equal organizational-group weighting.
Primary normal metric is MAP antipodal normal angle. Top32 physical-plane capture
requires the **same** selected candidate to have angle<=10deg and sign-aligned
normal-offset error<=500um; it is not finite-frame or anatomical registration.

Pilot advancement requires all of:

- At least5deg lower eligible group-macro MAP normal error than47.497952deg.
- At least .05 absolute improvement in eligible top32 physical-plane capture
  over .1059363553.
- Every populated eligible brush-mode intersection has normal regression<=2deg
  and top32 capture regression<=.02 absolute versus its frozen baseline.
- Valid finite results, authenticated provenance and the full planned budget.

Report all rows and censored rows separately, full-cell NLL, marginal normal
readouts, exact-cell recall, offset/full-frame error, top128 plane capture and
time/memory cost. A passing pilot justifies joint integration experiments; it
does not establish biological generalization, electrode uncertainty or shipping.
No public benchmark is used, and no failed run is automatically extended.

## Evidence and remaining limitations

The [CoMIR paper](https://proceedings.neurips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html)
supports learning multimodal image representations for registration;
[Synth-by-Reg](https://pmc.ncbi.nlm.nih.gov/articles/PMC8582976/) supports
structure-preserving cross-modal supervision but assumes paired planes and a
previously trained registration network. This pilot borrows the representation
idea, not those assumptions or weights. Its exact objective/thresholds are an
engineering hypothesis, not an established optimum for histology pose retrieval.

These are synthetic organizational groups from one Allen atlas, not biological
animals. Real acquired backgrounds, coherent subject variation, native constraint
learning, curved-surface joint refinement, unseen-animal uncertainty calibration,
and eventual frozen fair DeepSlice comparison remain separate requirements.
