# Frozen baseline evidence ledger

## Snapshot

- Repository commit: `c6681039e0b7acf35c9cdbee43040a3dca29cdab`
- Short commit: `c668103`
- Branch at documentation time: `codex/joint-registration`
- Purpose: immutable pre-development baseline for the proposed joint model

Values below are transcribed from tracked artifacts at this commit. They are not results of the proposed unified model.

## AtlasPose bundled candidate

Tracked artifacts: `models/AtlasPose/atlas_pose.json`, `models/AtlasPose/RELEASE_REPORT.json`, and `models/AtlasPose/provenance.json`.

Architecture/configuration: ConvNeXt-Tiny with binned physical pose head; 100,000 unique synthetic training views; Allen Product 5 registered sections used as trusted real supervision/selection. Product 8 is diagnostic only.

| Evidence set | AP MAE | L--R MAE | D--V MAE | Status |
|---|---:|---:|---:|---|
| Product-5 validation final gate | 20.038 µm | 0.344° | 0.655° | absolute validation gates passed |
| Product-5 locked test final gate | 20.542 µm | 0.305° | 0.632° | absolute test gates passed |
| Synthetic locked test, 8,192 cases | 35.628 µm | 0.803° | 1.105° | synthetic gates passed |
| Consumed DeepSlice comparison cohort | 27.748 µm | 0.429° | 0.595° | historical candidate metrics only; comparator result invalidated by orientation mismatch |

The tracked DeepSlice comparison cannot support a paired superiority conclusion. Audit found that the Allen rasters were supplied in the opposite horizontal view from DeepSlice's intended A-to-P convention. A physical atlas-axis reflection made aggregate numbers appear plausible but reverses laterality and is not a valid repair. The correct rerun preserves raw bytes and raw ground truth, applies one deterministic horizontal image reparameterization before DeepSlice, back-transforms the final O/U/V plane to the raw frame, and then scores it. Until that harmonized rerun is complete, the old DeepSlice AP/L--R/D--V deltas, confidence intervals and joint probability are invalid. The tracked `release_approved=false` and `promotion_ready=false` conclusions remain conservative.

The broad all-product diagnostic test was not a release endpoint and performed poorly because Product 8 carried known registration offsets: AP/L--R/D--V MAE `79.862 µm / 0.536° / 1.284°`, with worst-product AP MAE `486.050 µm`. This negative evidence is retained.

Historical v6 results reported in the repository README are development context, not the current release claim: synthetic AP/L--R/D--V MAE `58.72 µm / 0.934° / 1.052°`; on the subsequently reused 148-section DeepSlice set, `245.20 µm / 1.639° / 3.996°` versus published DeepSlice outputs `174.26 µm / 1.463° / 1.268°`. That public set was later used as feedback and is not untouched.

## Diffeomorphic-registration candidate

Tracked artifact: `models/DiffeomorphicRegistration/dense_registration.metadata.json`.

The candidate predicts absolute forward and inverse pixel maps at a fixed atlas plane. On two locked synthetic qualification seeds:

| Seed | Endpoint p95 | Visible-foreground correspondence | Macro region Dice | Release gate |
|---:|---:|---:|---:|---|
| 83117 | 1.241 px | 0.957840 | 0.916234 | failed |
| 83129 | 1.228 px | 0.958091 | 0.916989 | failed |

The tracked receipt status is `rejected`, `release_approved=false`, and the sealed test status is `not_run`. Its declared scope is synthetic deformation, grayscale appearance, damage and mask robustness on Allen CCFv3, with no claim of real-histology accuracy. DirectML/CPU export parity passed its tracked numerical checks, which establishes implementation parity only.

## What remains unproven at baseline

- one jointly trained pose-and-warp model does not yet exist;
- recurrent warp-to-pose feedback is not validated;
- no new multi-laboratory untouched real benchmark has been consumed;
- dense real-histology accuracy and calibrated confidence are not established;
- market-wide superiority is not established;
- ten-slice end-to-end latency for the proposed model is not measured.

Future results must be appended with run, source, data, checkpoint and evaluator hashes. Existing rows are never overwritten.

## Joint-model development evidence after baseline

At joint-pipeline commit `31d9441`, the corrected Stage-1 premise evaluator
tested whether the frozen dense-registration EMA supplied enough structural
evidence to rank the true plane without learning the new review head. Each
development-only stratum contained 24 sections and eight candidates per section
(true, initializer and six hard negatives). Dense-flow truth was not read.

| Stratum | True plane top-1 | Mean reciprocal rank | Mean true rank | Candidate failures |
|---|---:|---:|---:|---:|
| Clean | 0.5833 | 0.7813 | 1.5000 | 0 / 192 |
| Mild | 0.5000 | 0.7083 | 1.8750 | 0 / 192 |
| Hard | 0.4583 | 0.6771 | 1.9167 | 0 / 192 |

The frozen heuristic is therefore useful but insufficient. These results support
training an explicit wrong-plane compatibility/review head; they are not model
accuracy, confidence or release claims. The dense checkpoint SHA-256 was
`8463296f0a0846f0cd0463b4bc958edd1cf24a0957098b787abd388aca28c635`.
The clean/mild/hard manifest SHA-256 values were respectively
`e17d77dd90d47b8617789b14d02f32df6658144424b0e49313e90b5ec72d9e3f`,
`ab735d3d9d52c434afb20c453ff5d51792d68b5dce28af88e41b09398f21e6ba`,
and `458e7f33e21d64e2f7a31834b38cfdd6e8d3d2857edaf4c7c34df50545b67284`.
The corresponding summary-file hashes are
`5b6e821b79a1e3b5602f33a116a1fdb2ca5b34411ca888be66578b73d5fff77b`,
`1d6d7924bcaf8c452c3c9f22f292977b6120c0b57628c6e81833626689f3746a`,
and `eea89952f4cfeb53a86e23e61e0d344f5b4227cb70cae04471bb42afea2cdd52`.

## Joint-training systems preflight

Two 200-view development runs tested the complete review, geometry and joint
training state machine on the RTX 2080 Ti. The optimized run trained with the
three mandatory adjacent negatives (25 um AP and 0.25 degree L--R/D--V), while
both runs were evaluated against the same six-negative panels. Forward-only
review inference and omission of a discarded terminal training registration
were verified independently to preserve the relevant outputs and gradients.

| Run | Overall views/s | AMP overflows | Synthetic AP/L--R/D--V MAE | Product-5 AP/L--R/D--V MAE | Six-negative rank |
|---|---:|---:|---:|---:|---:|
| Original behavior | 0.057994 | 7 | 71.92 um / 0.886 deg / 1.788 deg | 18.70 um / 0.321 deg / 0.695 deg | 0.0833 |
| K=3 optimized | 0.292458 | 0 | 69.68 um / 0.907 deg / 1.806 deg | 18.55 um / 0.303 deg / 0.734 deg | 0.0833 |

The optimized preflight was 5.04 times faster end to end. A separate matched
40-view full-joint timing window measured 0.030909 views/s for the original
K=6/chunk-2/65,536-scale behavior and 0.160966 views/s for the optimized
K=3/chunk-4/512-scale behavior, a 5.21-fold improvement. The control incurred
seven skipped overflow updates; the optimized window incurred none.

These cohorts are deliberately too small for a quality claim: the review head
remained near chance after only 100 optimizer updates. Their role is to prove
pipeline stability, unchanged six-negative validation difficulty, and a
meaningful throughput improvement before a longer development run.

Original preflight hashes: config
`fe58d780e0f638d39619aac54fc1d7a45d9e9107a4a13b838ea0b9b2fb92e1ed`,
validation
`33993443ce0976fe8d121ae81e8d4434beeb7bc0f592636e7636d20a04f4d4b6`,
best checkpoint
`10cee3d282e44eb80712d5fd5decf3af563bbe0fadc2cf2e9c8cfb764c43bc88`.
Optimized preflight hashes: config
`f4c0496be8c26cd3a3fd1caebc50d6208b4f5034f68b1c1344da0f2bc6d3b183`,
validation
`f162f2c3154c94bc80f92e587317387ed7bb3bb3ea66e59ed093453970ac2d63`,
best checkpoint
`8bc7f2a285ac7935d6437c79b17906c40be5f2f3aaa54201d64e903a5671d575`.
Matched timing config/log hashes were
`70265b97df094a04ccb2cdc9bad96a11b9920eb60426245e34a32bee6f6c1902` /
`fbdd646947378a632425cdacbe4dd0e1047778a5491636d9c3c9d93be03a8fd3`
for the control and
`4ea7b430c7e0d741cdb60cccbdf222997c85b93a3cc41294d0076a3afafe2532` /
`700ec25b0c3c57da63a7a27fc674fb4def47da408cb853888fa5aa3e121feced`
for the optimized window.

## Staged 2,000-view development run

Run `joint-development-5000-r4322` stopped at its prespecified 2,000-view
decision point after 600 review-only, 1,000 geometry and 400 full-joint views.
The same fixed validation panels were evaluated every 500 views: 48 synthetic,
24 high-tilt synthetic and 96 specimen-disjoint Product-5 sections. This was a
development experiment, not a release qualification.

At 2,000 views, normal synthetic AP/L--R/D--V MAE was
`30.64 um / 0.576 deg / 1.021 deg`, AP p95 was `73.04 um`, exact-region
correspondence was `0.8817`, and macro Dice was `0.7522`. High-tilt AP/L--R/D--V
MAE was `65.47 um / 1.282 deg / 1.418 deg`, with AP p95 `237.15 um`.
Product-5 AP/L--R/D--V MAE was `22.80 um / 0.358 deg / 0.736 deg`.
No invalid endpoints, coverage failures or skipped dense batches occurred.

The wrong-plane reviewer learned on synthetic data: top-1 rank improved from
`0.417` at 500 views to `0.583` at 2,000 views and hard-stratum rank reached
`0.250`. Product-5 ranking cross-entropy improved monotonically, but its final
top-1 rate was `0.062`, below the six-negative chance rate of `0.125`.
High-tilt tails also remained outside the development guardrail. The run was
therefore not resumed to 5,000 views. The next matched experiment keeps both
pretrained components frozen and trains only the reviewer on mixed synthetic,
high-tilt and Product-5 batches, isolating reviewer learning from warm-start
forgetting before any lower-rate unfreezing experiment.

The configuration SHA-256 was
`14ea5221c10d8b189a0b1dcf7593cd77c021c877c48cf763eb0ac62178e524dd`.
The 500/1,000/1,500/2,000 validation-report hashes were respectively
`fd4c5d0684a8527ff6b317f6155349dcd2615b44e31bd749e4b256d90d7ce8f6`,
`5aabf3d4f98de5c4277b967e2b7684183571f71215ce9306babcad1ca970ef63`,
`d9411e74462b0ac7d51378c50ef165425d5e20420c361349be9d839a504837f4`,
and `ea41b5e75f37e32fc97b5e32a664d414aaaab2a19a6a333262dd4b02350b4883`.
The selected development checkpoint SHA-256 was
`afcd2592419cd65b288965fdab38ed92d93095341e31f7e8a87361f60b91fe84`.

### Prespecified reviewer-only control decision

Before inspecting any control validation result, the 2,000-view endpoint was
declared as the sole decision point; intermediate panels are trajectory
diagnostics. The pose initializer and registrar must remain exactly equal to
their warm starts, while fixed-panel initializer and teacher-pair dense metrics
must remain invariant from the control's first anchor. Any nonfinite value,
invalid endpoint, skipped dense batch or coverage failure stops the experiment.

With eight validation candidates, the Product-5 primary ranking gate is at
least `19 / 96 = 0.198` top-1 and cross-entropy at most `1.704`. Retention floors
are synthetic top-1 at least `26 / 48 = 0.542` with cross-entropy at most
`1.629`, hard-synthetic top-1 at least `4 / 16 = 0.250`, and high-tilt top-1 at
least `14 / 24 = 0.583` with cross-entropy at most `1.641`.

Pose correction uses
`E = AP_MAE / 60 um + LR_MAE / 0.9 deg + DV_MAE / 1.75 deg`. Product-5 final
`E` must be at most `1.138`, below its frozen initializer, with no component
more than 10 percent worse than the staged-run endpoint. Synthetic and
high-tilt `E` may be no more than 5 percent worse; correspondence and Dice may
fall by no more than 0.01 absolute. A conservative 0.1-times pretrained-module
unfreeze is allowed only if every freeze, ranking, pose and retention gate
passes. If Product-5 ranking fails while synthetic/high-tilt retention passes,
the next experiment diagnoses the Product-5 negative lattice instead of
unfreezing or enlarging the model.

### Reviewer-only control result

Run `joint-review-mixed-2000-r4322` used the same 5,000-view cosine schedule
and a prespecified stop at 2,000 views, while keeping the AtlasPose initializer
and dense registrar frozen throughout. Product-5 and high-tilt batches made up
25 and 20 percent of training draws. At the endpoint, all 205 initializer
tensors and all 180 registrar tensors were bit-for-bit equal to their warm
starts. Fixed-panel initializer and teacher-pair dense metrics were exactly
unchanged from the 500-view anchor. There were no invalid endpoints, skipped
dense batches or coverage failures.

| Panel | AP / L--R / D--V MAE | E | Top-1 | Ranking CE | Correspondence / Dice |
|---|---|---:|---:|---:|---:|
| Synthetic | 31.05 um / 0.711 deg / 1.046 deg | 1.905 | 0.583 | 1.152 | 0.8818 / 0.7623 |
| High tilt | 70.21 um / 1.052 deg / 1.447 deg | 3.166 | 0.625 | 1.184 | 0.8291 / 0.6776 |
| Product 5 | 22.25 um / 0.359 deg / 0.644 deg | 1.137 | 0.083 | 1.939 | not densely supervised |

The Product-5 frozen-initializer `E` was `1.090`, so reviewer correction made
it worse. Hard-synthetic top-1 was `0.188`. The control therefore failed the
prespecified Product-5 top-1 and cross-entropy gates, hard-synthetic retention,
synthetic `E` retention, and the requirement to improve Product-5 over its own
initializer. Conservative backbone unfreezing is not authorized by this
result. The next development step is a deterministic rank-by-offset audit of
the Product-5 candidate lattice and label precision; model capacity and loss
weights remain unchanged.

The control configuration SHA-256 was
`dfb4714d2cfc4e369e5d9ca3aab0ad81841dfe36ca20ac0953938c9500f9057b`.
The 500/1,000/1,500/2,000 validation-report hashes were respectively
`5683569d829ea0fd89a12c3e729a058ed1cb4b1f56cc401874bdad292e3d6120`,
`0866c138502bed55f3ce84d783c0787807315764cfdda9b4d666288a4a47d4d1`,
`c191b53427b93684b424588cff2bc81f9d2212c1244c598ff52063075169cebe`,
and `aa3bd4ff5ba017a633931eecb35d292a1d8527709f49609b22b7d5c97da7cb09`.
The final latest and validation-selected checkpoint hashes were
`0b7a940ee586cf4f91a9dc72895b50136804c9b8164490f4ad37eda28aa52790`
and `09ef228b2c97766cb284ed7b9b94a2fa833d4c5c503d037b8ab7d380c3ea406f`.

### Prespecified Product-5 offset audit

Before consuming its result, the diagnostic cohort was fixed to the same 96
validation sections and the control's final EMA state. Each metadata plane is
paired independently with signed AP offsets of 25, 50, 100, 250, 500 and
1,000 um and signed L--R/D--V offsets of 0.25, 0.5, 1, 2, 5 and 10 degrees.
No weights are updated. Offsets of at least 100 um AP or 1 degree tilt are
declared resolvable; 25 um and 0.25 degree offsets are the nearest-neighbor
stratum. Product-5 sections are 100 um thick and their within-specimen adjacent
AP spacing has 5th/median/95th percentiles of 91.88/96.24/99.84 um.

Frame integrity requires metadata O/U/V rederivation to agree within float32
tolerance, a known asymmetric atlas self-render to prefer zero offset without
reflection, and no H, V or H+V reflected real-source variant to improve both
frozen registration evidence and reviewer margin with a paired 95 percent
interval excluding zero. A frame failure is repaired before any retraining.

Sub-resolution ambiguity is diagnosed only if frame integrity passes,
resolvable offsets have a pooled truth-win Wilson lower bound above 0.5, the
nearest-neighbor truth-win interval contains 0.5 with a margin interval
containing zero, and removing nearest neighbors improves reconstructed top-1
by at least 0.10 absolute. In that case, real-section training uses an explicit
tolerance/tie policy instead of unique hard labels at sub-resolution offsets.
If frozen registration evidence passes on resolvable offsets but reviewer
truth-win does not, the failure is reviewer domain transfer. If neither passes,
the limitation is the real-image registration evidence rather than the review
head alone. Architecture size, recurrence count and loss weights remain fixed
until this causal classification is complete.

### Product-5 offset audit result

The corrected v2 diagnostic completed on the fixed 96-section panel without
changing any model weights. Version 1 is invalidated only because one
unit-inappropriate `1e-5` scalar tolerance rejected an AP metadata round-trip
residual of `0.0001945 um`. Version 2 uses prespecified component tolerances of
`0.01 um` AP and `0.0001 deg` L--R/D--V. Its pair CSV and orientation CSV are
byte-identical to v1, and all pose, evidence, ranking and top-1 results are
unchanged.

Frame integrity passed. The asymmetric positive-determinant self-render chose
the unreflected zero-offset plane under frozen registration evidence. H, V and
H+V reflections all worsened both registration evidence and reviewer margin,
with paired intervals below zero. O/U/V rederivation passed all three physical
components. A frame or reflection bug therefore does not explain the
Product-5 ranking failure.

The prespecified taxonomy gives a mixed result and no gate is moved to force a
cleaner label. Resolvable offsets were ranked correctly by frozen registration
with specimen-cluster truth-win `0.894` and 95 percent interval
`[0.878, 0.909]`; the reviewer reached `0.765 [0.732, 0.795]`. Removing nearest
candidates raised current-list top-1 from `14/96` to `58/96` for registration
and from `8/96` to `32/96` for the reviewer. However nearest-neighbour
registration remained weakly above the null (`0.566 [0.524, 0.608]`) and its
margin interval did not contain zero, so the formal symmetric
sub-resolution-ambiguity criterion did not pass. Because reviewer performance
on resolvable pairs was also above chance, the pure reviewer-domain-transfer
criterion did not pass either. Exhaustive resolvable-only top-1 remained low
(`16/96` registration; `2/96` reviewer), demonstrating severe multi-candidate
crowding and direction-specific transfer rather than a single frame failure.

For future cold-start experiments, this evidence changes Product-5 candidate
construction but does not relax labels or evaluation. Singleton listwise
cross-entropy is retained because the conjunctive ambiguity rule failed. Each
real Product-5 list instead contains six sign-balanced one-axis negatives
(`+/- AP`, `+/- L--R`, `+/- D--V`), with levels scheduled across batches to
cover nearest, the prespecified `100 um / 1 deg` resolvable boundary and wider
offsets. The former K=3 construction supplied only one randomly signed nearest
negative per axis and therefore never taught consistent signed resolvable
ordering. Exact synthetic training remains unchanged. If this balanced
cold-start experiment still fails exhaustive ranking on a fresh
specimen-disjoint validation panel, a monotonic lattice-ranking term must be
prespecified before it is tested; label tolerance cannot be inferred from this
consumed audit.

The v2 report, pair CSV and orientation CSV SHA-256 values are respectively
`d61e2ec6077639111ed59d1d0909f7ca6f1043d3138dc074d1580df6f464ee00`,
`8b08f647a580ae80344865adfd7d24727a663a36602d4ca8e4561e338dd2bbc6`,
and `b06a756d05c6aa8dbcb577d8e54519392e2a875cf9a741fcb44411f63574d0ab`.
There were 3,456 requested pairs, 3,393 scored pairs and 63 declared
out-of-domain pairs.

### Independent architecture decision

The warm-start joint implementation is now classified as a legacy-seeded
systems prototype and development diagnostic. It contains 30.91 million
parameters, of which 96.7 percent belong to the inherited AtlasPose branch,
2.5 percent to the inherited dense registrar and only 0.8 percent to the new
reviewer. It has no recurrent hidden state and its high-resolution candidate
path is too slow to be presumed suitable for the final task. None of its
weights or inherited architectural constants is promotion eligible.

Release-eligible models start from random initialization with an empty learned
checkpoint dependency list. The matched screen compares a compact factorized
CNN control, a recurrent local-correlation/ConvGRU pyramid, and a windowed
cross-attention pyramid after export preflight. AtlasPose, AtlasWarp and
DeepSlice remain frozen comparators only. The deterministic CCF renderer,
coordinate contracts, data generators, constraint solver and benchmark
infrastructure are retained because they are neutral task machinery rather
than learned legacy components.

### Cold-start implementation checkpoint

Commits `9470464`, `addb64b` and `0517bce` implement the standalone recurrent
model, matched factorized/attention controls, hash-bound synthetic and Product
5 data adapters, and the audited cold-start trainer. Every learned parameter is
randomly initialized; the checkpoint lineage rejects learned dependencies and
binds the exact initial state, architecture graph, source/data/atlas hashes,
training-animal IDs, optimizer schedule, EMA state and development-panel raw
predictions.

The input contract implements the optional-outline decision directly: 35%
accurate outline with black exterior, 35% independently perturbed outline with
black exterior and 30% no outline with acquired background. An explicit
availability scalar distinguishes absence from an empty mask. Tissue damage,
missing tissue and correspondence validity remain independent targets.

At this checkpoint, 56 focused model/data/variant/trainer tests pass, including
dynamic-batch ONNX Runtime execution, candidate-order randomization, three-step
back-propagation through time, synthetic-only truth-plane dense supervision,
map-direction/validity contracts, animal-disjoint EMA selection and atomic
resume. These are implementation tests, not accuracy evidence. No long
architecture-screen run or final benchmark has been launched.

One non-comparative RTX 2080 Ti integration step used a real CCF-backed hard
synthetic sample at batch size 1 with FP16 autocast. The complete three-update
path plus the one exact dense teacher pass produced a finite loss, a finite
nonzero gradient through the first recurrent pose, and used 1.613 GiB peak
allocated CUDA memory. Its cold wall time (17.45 s, including unoptimized
training graph execution) is a plumbing check only and is not a throughput or
model-quality benchmark.

The subsequently frozen equal-workload preflight used three source images,
three candidates per source, three updates and one dense decode at 320x464,
with one warm-up and three measured iterations. It performed zero optimizer
steps. On the RTX 2080 Ti, the recurrent-correlation, factorized-stateless and
recurrent-attention families respectively measured 1.369/1.387/1.393 million
parameters, 36.24/42.92/37.27 billion scoped MAC proxies, 1.86/2.14/2.05 GiB
peak allocated memory, 91/88/100 ms median forward time and 154/117/221 ms
median backward time. All three candidate-scorer ONNX graphs passed checking
and executed through DirectML; maximum absolute PyTorch/ORT differences were
`1.26e-6`, `1.60e-5` and `1.28e-6`. The MAC proxy counts convolution, linear
and explicit local-attention work but excludes elementwise operations and grid
sampling. These measurements establish comparable feasibility only and do not
rank accuracy.
