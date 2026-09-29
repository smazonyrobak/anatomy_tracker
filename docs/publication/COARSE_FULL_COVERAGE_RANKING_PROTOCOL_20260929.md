# Full-coverage TRAIN ranking control — prepared, not launched

## Hypothesis and parent

The completed TRAIN census found only48,000 distinct generated observations and
50,125 eligible positive target cells across98,304 cells; B/C reused generated
indices32000:48000. All384 normals and6142/6144 normal-offset planes already have
positives. Incomplete finite-frame/appearance exposure and≈48–64 sampled keys per
query are therefore plausible learning limitations, not proof that unseen plane
directions or insufficient descriptor capacity caused the≈38° MAP error.

C's saved model configuration is64 source channels/128 hidden channels with a
256D descriptor; the retrieval path has344,896 trainable parameters, local
convolutional receptive field11/stride4 and4×4 adaptive spatial pooling. The old
six-residual-block007 control used a different geometric-mixture proposal head,
not this contrastive image-key objective, so it does not rule out present
descriptor-capacity limitations. Instantiate this control from C's exact saved
configuration rather than treating more data as an established explanation.

Preregister both arms, but run the uniform-coverage arm first. Run the hard arm
only if that complete endpoint fails the unchanged overall gate below. A passing
uniform endpoint ends this contrast without a hard run: make no claim about
mining's effect in that case. This sequential rule changes neither arm's fixed
budget/data nor the gate, and does not depend on the pending native result.

Both arms start from the same **whole experimental C10000** checkpoint
`I:/AnatomyTracker/runs/joint_v6_canonical_rehearsal_001/joint_model_step_10000.pt`,
SHA256 `888ad9cc493453ce769628fda1aebe70813c031f7809192076c08b000720d9af`.
Restore its optimizer/RNG independently in each arm. C failed its prior retention
gate and is not promoted by this initialization. No encoder/refiner splicing,
external weights/features/pseudolabels, descriptor redesign or new LR sweep.

## Fixed shared data and update budget

Each arm applies exactly12,576 updates: four generated queries per update consume
50,304 remaining catalogue cells once. Source order is the original committed
complete-cell permutation at indices48000:98304, taken from
`runs/joint_v6_imagekey_retrieval_001/replay003_generated_schedule.npz`
(SHA256 `5f88b717d022026cc386f337e8979cfba11a25f4c491dfe05d49b53113678712`).
The new driver must freeze this cell list before either arm, and record its
disjointness from0:48000. This completes **attempted** positive exposure of every
catalogue cell; censored low-support cases remain in the count and are not
redrawn or represented as successfully supervised positives.

Do not reuse the old appearance/sample-seed arrays. Freeze fresh appearance
draws using seed2026092931: independent sample seeds, thickness25–100µm with the
existing nine-point normalized PSF, gain/gamma/inversion/noise/background and
imperfect-mask semantics. Assign all three existing raw/exact-black/imperfect
modes in a shuffled balanced list (16,768 each), with the existing random
horizontal reflection. Preserve cell geometry and exact provenance. These are
one-atlas synthetic observations, not new biological subjects; fixed-cell poses
still do not constitute continuous subcell or subject-anatomy coverage.

Each update also uses four original frozen TRAIN rows and eight real TRAIN
queries. Freeze both schedules before training, using seed2026092932 for frozen
rows and2026092933 for donor-round-robin then within-donor real sampling. Retain
pose anchors, joint/complex rows, original eligibility weights and all brush
modes; no development-derived balancing or excluded difficult rows.

Use the linked1280 TRAIN images only after the approved1024-section acquisition
has exited and its completed input/geometry hashes are pinned. It must contain
the original256 plus the fixed1024 additions from the same58 donors, with no
development/benchmark image acquisition. All input pixels/preprocessing and
exact IDs/transforms remain unchanged. Render matched/canonical continuous
affine anchors once per distinct TRAIN image with the existing50µm/nine-point
PSF. Recompute common matched/canonical finite-support eligibility≥64 once;
retain ineligible rows with zero real loss rather than filtering/redrawing them.
The chosen PSF and upstream affines remain weak engineering supervision.

Real chart selection uses seed2026092934, with exactly50,304 matched and50,304
canonical presentations before eligibility weights. Preserve each source plane
and roll; canonical centre is `S+n*dot(n,c-S)` with12mm orthogonal edges/no shear.
Neither geometry nor IDs enter the query encoder.

Both arms retain C's FP32/TF32-off implementation, trainable module prefixes,
AdamW LR0.00025/weight decay0.0001, clip5 and loss weights2/3 synthetic,1/3 real.
Budget per arm:100,608 synthetic and100,608 real presentations, of which50,304
synthetic are new generated observations. End optimizer step is22,576. The real
expansion, fresh appearances and longer coverage are **shared interventions**
versus C; if both arms run, only negative-selection policy is attributable to the
paired contrast. This includes any differences in duplicate/effective candidate
counts, not hardness alone or a separately isolated data-scaling effect.

## Matched negative-selection contrast

Use the same canonical catalogue frames, normalized50µm PSF and two raster
representations as the current complete gallery. Preserve the existing eight
geometrically local synthetic negative choices and batch truth keys. Add:

- **Uniform control:**16 shared global uniform cells plus16 query-specific
  uniformly drawn admissible wrong-plane cells, one per query.
- **Hard treatment:**the same16 global uniform cells plus16 query-specific
  snapshot-model highest-scoring admissible wrong-plane cells, one per query.

The query-specific admissible set is identical in both arms: reject cells within
10° antipodal normal **and**500µm sign-aligned normal offset of that query's
physical reference plane, and reject its own positive label when applicable.
Use actual continuous frozen/real references and exact generated-cell geometry,
not a nearest-cell approximation for real images. A query with zero supervision
weight contributes the same predeclared global-uniform fallback in both arms.
Use isolated seed2026092935 for global draws and2026092936 for conditional-uniform
draws, so mining cannot advance data/appearance RNG. Stable ties select lower cell
index. Union/deduplicate keys without result-dependent refill; record candidate
counts and duplicate counts. Maximum synthetic catalogue keys is48 before
deduplication, plus eight selected real anchors as in the existing driver.

This matched uniform arm differs from C's32 unconditional uniform draws: half
its random draws now come from the same per-query wrong-plane set as mining.
It prevents attribution of an eligibility-rule change to hardness alone.

Rank hard candidates by the deployed complete-gallery cell score for both query
types: `logsumexp_r(cosine/0.1 + log_representation_prior) + log_cell_mass`, each
prior once; normalization is unnecessary for argmax. Real paired-NCE still uses
its existing uniform-representation/no-cell-prior objective. These are therefore
hard deployed-gallery candidates, not necessarily the largest-real-loss keys.

Do not intentionally mine roll-only or in-plane-chart alternatives as false
coarse-plane negatives. Existing synthetic **loss masks** still preserve their
finite-frame semantics (10° plus1mm corner-RMS exclusions), and existing real
loss masks still use physical-plane equivalence. Another query's mined key can
be near-plane-but-frame-different for a given synthetic query, just as existing
batch/local keys can. Record this rate; do not silently relabel all such keys as
wrong-plane or change the full-frame objective during this contrast. A later
plane-marginal/multi-positive objective would be a separate declared experiment.

## Mining refresh and bounded compute

For the hard arm, snapshot the current retrieval encoder and refresh its full
98,304×2 atlas descriptor bank immediately before update1 and after each1024
applied updates, through update12,288: **13 training refreshes**. One frozen final
gallery serves endpoint evaluation, totaling14 complete gallery encodings
(2,752,512 cell-representation encodings). The uniform arm needs only its final
gallery; this is not a matched wall-time/compute benchmark. Keep all snapshot and
mining RNG isolated from training/data schedules.

Use existing chunked atlas rendering/encoding. The completed C bank is actually
FP16[98304,2,256],100,663,296 bytes=96MiB; a converted FP32 copy requires192MiB,
or288MiB if both coexist. Include that copy and the frozen snapshot encoder in
memory accounting, rather than calling a FP32 bank96MiB.

During each interval, encode both query and atlas keys with the **same frozen
own-lineage snapshot**, not the changing query encoder against stale key features.
The snapshot receives the actual scheduled TRAIN image/appearance and selects
indices without gradient. IDs are snapshot-hard, up to1023 updates old, not
claimed current-model hardest. Record snapshot step and chosen indices.
**Never use snapshot/cached descriptor values as loss targets or gradient-bearing
keys.** Rerender/re-encode every selected key and query with that arm's current
trainable encoder in the ordinary differentiable loss. This is own-lineage
negative-index selection, not prior-model feature supervision or encoder merging.

A dense16×98,304×2 scoring tensor is≈12MiB FP32 per step. Initial gallery/first
training chunk reports elapsed time and memory; it is a feasibility measurement,
not grounds for changing LR, candidate policy, data, gates or selecting a model.
A resource/numerical failure is reported as an incomplete experiment; do not
switch silently to a cheaper denominator or stop on development fluctuations.

## Readouts and frozen decisions

No interim development evaluation, development-query mining or native-result-
dependent selection. Log losses, support weights, candidate IDs/counts, gallery
refresh step and cumulative distinct attempted/eligible cell coverage. These
stage counters distinguish fresh generated attempts, eligible fresh cells and
the union of all eligible positive cells in this stage; they do not relabel
previously censored lineage cells as supervised. Periodic
checkpoints retain the whole model, optimizer/RNG and exact schedules; no
per-minibatch reauthentication ledger or new generic framework.

At the fixed endpoint evaluate both arms on the unchanged full640 synthetic and
64 real-development rows with the same complete gallery, raw component/cell
scores, mode/censor/group summaries and geometric readouts as the existing
runner. TRAIN gallery readouts use all5120 frozen rows and all1280 real rows;
stream metrics/top128 rather than requiring a new huge TRAIN full-score archive.
Distinguish full-frame/rank metrics from normal/offset/plane capture. Compare
all groups, including failures; do not select evaluation rows from mined errors.

Before either arm, freeze an image-free orientation baseline from all1280 TRAIN
affines, without anchor-eligibility selection: compute the per-donor mean of
`n*n.T`, average equally over58 donors, and take its leading unit eigenvector.
Save its matrix/eigenvalues/vector, input geometry hashes and row/donor IDs.
At the unchanged real endpoint report its donor-macro antipodal normal error
and model-minus-prior error beside model error. This cosine-squared scatter prior
is not necessarily the mean-angular-loss optimum; it supplies no offset,
full-plane or calibrated-posterior estimate and changes no gate.

Carry the previously frozen advancement thresholds unchanged: real improvement
versus original A6000≥5° normal and≥0.10 capture@32; synthetic normal≤A+2° and
capture@32≥A−0.02 for eligible overall and every original mode; real retention
versus B8000 normal≤B+2° and capture@32≥B−0.02. Integrity, complete fixed budget,
frozen nonretrieval tensor preservation and no split/provenance violation are
separate mandatory checks. These are engineering gates, not calibrated
statistical evidence. No arm is promoted solely because it beats the other.

The hard-minus-uniform contrast tests ranking under matched expanded exposure.
If TRAIN full-gallery discrimination remains poor, objective/ambiguity/capacity
remain alternatives; this protocol cannot identify descriptor capacity alone.
If TRAIN improves but development does not, do not infer biological validation
or solve the gap by removing donors. Coarse success still requires sequential
native training of a selected whole checkpoint—not merging the concurrently
trained native control—and later independent animal-level calibration/benchmark
work. No DeepSlice/final benchmark is accessed by this experiment.
