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

### Completed 2,000-step cold-start architecture pilot

The development-only pilot defined by
`training/configs/independent_architecture_development_screen.json` completed
on 2026-08-18. It ran from source commit
`750bc723241e8e1626ca197a4af517a897f72b88`; its protocol and shared panel
contract SHA-256 values were respectively
`6a04a27888a8293b3ffecaa5a08044ec8e12f252313b26c3d971d2a73057c882` and
`b78af91f5022f49dd5d890faf33e42d9f6259e6f76b9d28148e5ed110a18f4c7`.
All three runs reached 2,000 optimizer steps without a stderr record. The run
root was
`F:\AtlasJointTraining\runs\independent-architecture-screen-r4322\independent-joint-development-screen-v1`.
The exact artifact paths and hashes are recorded in
`publication/architecture_screen_pilot.yaml`.

The audit reproduced each screen-setup, source, architecture, initial-state,
panel-manifest, trainer-lineage and best-checkpoint hash. Every model and EMA
checkpoint tensor inspected was finite. The three runs used empty learned
checkpoint dependency lists; 2,098 Product-5 training animals and 112
validation animals were disjoint; and calibration and final-test access were
both false. Each best checkpoint was selected from the EMA state and contained
the raw predictions from the freshly evaluated panel that selected it.

| Family | Parameters | Best step | Best internal metric | Step-2,000 metric | Product-5 no-outline AP MAE / bias / p95 (um) | L--R / D--V MAE (deg) | Ranking NLL / top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recurrent correlation/ConvGRU | 1,369,070 | 500 | 13.30342795588076 | 14.788587550073862 | 813.3569 / 54.2878 / 1653.8094 | 0.7754 / 4.2166 | 1.9460069 / 1 of 9 |
| factorized stateless CNN | 1,387,342 | 250 | 13.306698899902404 | 15.537300672382116 | 801.8859 / 90.3684 / 1687.3277 | 0.7734 / 4.1840 | 1.9462054 / 0 of 9 |
| windowed attention/ConvGRU | 1,393,454 | 500 | 13.302073648571968 | 16.495703548192978 | 813.6517 / 53.8670 / 1653.9488 | 0.7754 / 4.2287 | 1.9460236 / 0 of 9 |

These are development diagnostics, not benchmark results. The p95 values in
the table use NumPy's linear percentile definition on nine raw per-animal
errors. All seven candidates were valid for those cases, so uniform ranking
NLL is `log(7) = 1.9459101490553132`. The internal composite used 0.50
Product-5 absent-outline, 0.25 paired hard-synthetic absent-outline and 0.25
hard high-tilt absent-outline panels. Only 9 of the 30 Product-5 panel animals
were assigned to the primary absent-outline mode; 10 had accurate and 11 had
imperfect outlines. That small repeatedly consumed primary subset is
insufficient for an architecture claim. A future decision panel should run
the full real panel without an outline and treat paired outline conditions as
a separate sensitivity analysis.

No winner was selected. The numerically lowest composite, from the attention
model, was only 0.0102 percent below the smaller recurrent-correlation model;
the largest best-score separation was 0.0348 percent. More importantly, all
three missed the frozen AP-MAE, AP-bias, AP-tail and D--V eligibility limits,
their wrong-plane ranking was indistinguishable from uniform at this panel
size, and none of the declared correspondence, macro-Dice, Jacobian or
end-to-end runtime gates was evaluated by the pilot runner. Each model's
metric worsened after its early best checkpoint. These observations diagnose
an inadequate early training regimen; they do not rank the fusion families.

The run followed its own frozen 2,000-step development contract, but that
contract did not implement the publication-level architecture-decision
protocol. The latter specifies a 20,000--50,000-view comparison and a staged
foundation-to-closed-loop curriculum. This pilot instead presented 4,000
batch items, cycled clean, mild and hard synthetic cases from the start,
introduced high-tilt and Product-5 batches immediately, and always used three
updates. A post-run audit also found that applying the 35/35/30 outline
allocation independently to every two-sample training batch rounded each
batch to one accurate, one imperfect and zero absent-outline examples. The
realized training mix was therefore 50/50/0, despite the correct declared
global 35/35/30 contract; this directly invalidates the pilot as a test of
no-outline generalization. Future runs precompute one global hash-bound
outline plan before slicing it into batches. The runner also had no
cross-family eligibility decision and the
practical-equivalence margin had no frozen numeric value. The pilot must
therefore remain a systems and regimen diagnostic. Its development panel is
marked consumed and cannot become calibration or final-test evidence.

Finally, each final training-batch receipt contained two non-finite AP
covariance entries serialized as null, although the corresponding Cholesky
factors, best-panel outputs and checkpoint tensors were finite. The shared
cause was float16 overflow when forming a covariance in physical units under
AMP; this was not an architecture-specific failure. Commit
`f8547db78016a2b260572944459acc86247605d6` moves the probabilistic Cholesky and
covariance branch to float32 and adds an autocast/export regression test. The
completed pilot artifacts retain their original source lineage and are not
rewritten or promoted after that fix.

### Cold-start initializer foundation attempts

Two development-only attempts tested whether the randomly initialized shared
encoder and probabilistic pose head could establish a usable pose foundation
before closed-loop registration. Both used only synthetic CCF data, the
initializer-only execution path, an empty learned-checkpoint dependency list,
the same initial state (`370934a18d8d873c3ced7a0ed17a963b929c9ceb452190e64c92047f7f1136a6`)
and the same fixed 24-case development-panel contract
(`787b617b5f6b7f1a4fc23002d421bc8511f41cad18766edff25d9c7be576f2d7`).
Product-5, calibration and final-test access were false. The planned run was
2,500 unique views, with a prespecified stop at 1,000 if overall physical plane
error had not fallen by at least 15 percent.

The first attempt ran from commit
`51357bf8e5a451c93fcc33d1331e891ea260bca7` under config contract
`597c237997819d38ef508caa93e0300aea022e3032c921aa65a8f503909aa0fd`.
It stopped after 93 optimizer updates and 186 unique views when the receipt
recorded one non-finite training event. Run diagnosis localized the event to an
AP-logit gradient overflow under the default AMP `GradScaler` initial scale of
65,536; the receipt records the non-finite gradient-norm stop but does not
store the offending tensor name. Because the run stopped before the 500-view
warm-up boundary, it supplied no post-warm-up gradient summary and no endpoint
comparison. Commit `7679e03cf3a4afd1939dfb845748b4f6935377d6`
made the initial scale an explicit frozen config value of 512 without changing
the model architecture.

The corrected attempt ran from that commit in a separate artifact root under
config contract
`718ce78c706e79d5ac571cc5b92ed2c5f313d0fab0563ac93f67e210d781214d`.
It reached the prespecified 1,000-view interim decision with zero non-finite
training or panel outputs. On the fixed 24-case panel, overall physical plane
error changed from `1168.9459228515625 um` at initialization to
`1166.142333984375 um` at 1,000 views, a reduction of
`0.0023983905605730158` (0.239839 percent). This failed the frozen 15 percent
gate, so the runner stopped cleanly rather than continuing to 2,500 views.
Across 250 post-warm-up updates, the pose head was clipped on 100 percent of
updates (median clip factor `0.2336307245404901`), while the encoder was
clipped on zero percent (median factor `1.0`).

| Attempt | Views | Overall AP / L--R / D--V MAE | Physical plane error | Non-finite training count | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| default AMP scale | 186 | initialization only: 1162.526 um / 5.553 deg / 2.985 deg | initialization only: 1168.946 um | 1 | numerical stop |
| explicit scale 512 | 1,000 | 1152.145 um / 5.594 deg / 2.986 deg | 1166.142 um | 0 | failed 15 percent interim gate |

These attempts are optimization diagnostics, not benchmarks or accuracy
evidence. The first is numerically incomplete and the second failed its
prespecified qualification gate; neither selects an architecture or authorizes
promotion. The development panel is consumed for development, while calibration
and final-test data remain untouched. Exact paths and independently recomputed
artifact hashes are recorded in
`publication/initializer_foundation_attempts.yaml`; the original run artifacts
were not rewritten.

### Pose-readout identifiability diagnostic

The failed foundation attempts were followed by a narrower development-only
diagnostic at commit `966743a45efc4585dde03261dbbf09e0efcc3a6f` and config
contract `c7a1ce8e2808cb14cc806075a397053e54eea7886b45ed105855a6da09073d26`.
It trained only the randomly initialized source-image pyramid and categorical
plus sub-bin pose readout; atlas, recurrent and dense-registration parameters
were frozen. The fixed absent-outline synthetic construct contained 24 latent
poses spanning six AP levels and all sign combinations of 13.25-degree L--R
and 18.25-degree D--V tilt. Two 24-case panels used the seen nuisance-transform
set, while two 24-case panels used a disjoint generator realization and held
rotation/scale values. Product-5, calibration and final-test access were false,
and the learned-checkpoint dependency list was empty.

The run reached its terminal 300 optimizer updates and 7,200 sample
presentations without a non-finite training or evaluation value. Its
post-warm-up gradient-clipped fraction was `0.30`, below the prespecified
strict maximum of `0.50`. Nevertheless, accuracy on the 48 seen-transform
cases was only `0.3125 / 0.6458333 / 0.7916667` for AP / L--R / D--V bins,
against gates of `0.95 / 0.90 / 0.90`. Residual learning was worse than the
zero-residual bin-centre baseline on every axis, with relative improvements of
`-36.4744 / -11.5419 / -9.2451`.

On the 48 held-transform cases, AP / L--R / D--V MAE was
`1425.3208 um / 12.7109 deg / 15.2627 deg`, missing the respective
`250 um / 3 deg / 3 deg` gates. The corresponding-plane error was
`1673.2141 um`, compared with `1403.5191 um` for the constant-pose prior, so
the defined physical improvement was `-0.1921563` rather than the required
`0.50`. Held residual improvements were also negative on all axes
(`-37.0086 / -15.9478 / -19.3503`). Prediction-to-truth standard-deviation
ratios did pass, showing that the failure was not merely a constant-output
collapse.

The prespecified terminal classification is therefore
`pose-representation-not-identifiable-on-seen-transforms`. This is not a
benchmark, an accuracy claim or a model-selection result. It authorizes only a
controlled spatially aware pose-readout diagnostic on consumed development
data, with the comparison contract and protected-data boundary held fixed; it
does not authorize architecture promotion or calibration/final-test access.
The sole state artifact is an atomic resume state, explicitly not a selected
model checkpoint. Exact receipt, state, raw-prediction and log hashes are in
`publication/pose_identifiability_diagnostic.yaml`.

### Spatial-moment pose-readout diagnostic

The authorized spatial-readout ablation ran from commit
`c72439b62463573d8657d04dae0cc4a908b7b257` under config contract
`85b50df71447ce8c7340672b388a3b0896dc0cf66f3ace4246fa2c863de7e746`.
It added four learned spatial-softmax maps, each summarized by two means, two
variances and one covariance, to the existing multilevel global-average pose
context. This added 4,268 parameters (0.3117 percent) without changing the
fixed absent-outline panel contract, latent poses, nuisance assignments,
training schedule, losses, optimizer, seed or gates. It remained independently
random-initialized with an empty learned-checkpoint dependency list; Product-5,
calibration and final-test access were false.

| Diagnostic | Parameters | Seen AP / L--R / D--V bin accuracy | Seen AP / L--R / D--V MAE | Seen physical error | Held AP / L--R / D--V MAE | Held physical error / improvement over prior |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global-average base | 1,369,070 | 0.3125 / 0.6458 / 0.7917 | 1405.29 um / 9.406 deg / 7.684 deg | 1549.72 um | 1425.32 um / 12.711 deg / 15.263 deg | 1673.21 um / -0.1922 |
| Spatial moments | 1,373,338 | 0.3750 / 0.8542 / 0.9167 | 1489.23 um / 3.989 deg / 3.247 deg | 1480.73 um | 1400.53 um / 13.817 deg / 10.060 deg | 1539.11 um / -0.0966 |

The 300-update run completed with zero non-finite training or evaluation
values. Its post-warm-up clipped fraction was `0.3814815`, below the frozen
`0.50` limit. Spatial moments materially improved seen-transform tilt
decoding: D--V bin accuracy passed its `0.90` gate, L--R reached `0.8542`, and
their seen MAEs fell by `5.418` and `4.437` degrees. Seen physical error fell
by `68.99 um`, and held physical error fell by `134.11 um` relative to the
base diagnostic.

The ablation did not solve the task. Seen AP accuracy was only `0.375` against
the `0.95` gate and its AP MAE worsened by `83.94 um`. On held transforms,
AP bin accuracy fell, L--R MAE worsened, every held MAE gate failed, and the
physical error remained 9.66 percent worse than the constant-pose prior.
Residual learning was worse than the zero-residual bin-centre baseline on all
axes in both partitions. The terminal classification therefore remains
`pose-representation-not-identifiable-on-seen-transforms`.

This is immutable negative development evidence, not a benchmark, performance
claim, architecture selection or promotion decision. It does not authorize a
longer run or gate changes. The only authorized next experiment is a matched
supervised source-view canonicalization ablation on this consumed construct:
global-average base plus canonicalizer versus spatial moments plus the
identical canonicalizer. Exact lineage, result, raw-record, state and artifact
hashes are recorded in
`publication/spatial_moment_pose_identifiability_diagnostic.yaml`; the original
artifacts were not rewritten.

### Supervised source-view similarity canonicalization diagnostic

The authorized matched ablation ran from commit
`f35b234bb8abceb5dc4f938414b959b8b9436ae5`. Its two frozen config contracts
were `ac8167ec0324fe50290487e987b21c34d9070062959b3c0c1a39048d25a1cc82`
for the global-average model and
`af6563df1a5f8c7f498d4584650b18c33398aeca4884ee6138f0c38c61a55108`
for the spatial-moment model. Both used the same consumed 24-pose synthetic
CCF train construct, two seen and two held panels, absent outlines, random seed
4322, 300-update schedule, losses, optimizer and gates. Product-5, calibration
and final-test access were false, and learned-checkpoint dependencies were
empty.

The identical randomly initialized canonicalizer was supervised only with the
exact synthetic source-view rotation and scale; anatomical AP/L--R/D--V targets
were unavailable to it. Pose-loss gradients were blocked at the sampling
parameters, and pose and canonicalizer gradients were clipped separately. Both
runs completed 300 updates and 7,200 sample presentations with no non-finite
training or evaluation values.

| Diagnostic | Seen AP / L--R / D--V bin accuracy | Seen rotation / scale MAE | Held AP / L--R / D--V MAE | Held rotation / scale MAE | Held physical error / improvement over prior | Post-warm clipping |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global-average + supervised canonicalizer | 0.4166667 / 0.7083333 / 0.7083333 | 19.9740200 deg / 0.0376704 | 1637.5049 um / 12.2104 deg / 15.2750 deg | 13.4144669 deg / 0.0315908 | 1880.4042 um / -0.3397782 | 0.3888889 |
| Spatial moments + supervised canonicalizer | 0.4166667 / 0.9375000 / 0.7291667 | 19.9740200 deg / 0.0376704 | 1532.3495 um / 14.4083 deg / 13.7384 deg | 13.4144669 deg / 0.0315908 | 1658.7432 um / -0.1818459 | 0.4481481 |

The source-view gates were at most 2 degrees and 0.03 on seen transforms and
3 degrees and 0.05 on held transforms. Both canonicalizers therefore failed
seen rotation and scale and held rotation; only held scale passed. The final
canonicalizer tensors were bit-exact between runs with SHA-256
`37c91f6a84c8b3412b2261e0319cd3e78e89510d0654d0d0317345e79bc2a261`.
All 11 canonicalizer optimizer states, all 300 canonicalizer gradient/loss
records, and all 96 source-view prediction/error fields were also exact-equal.
This isolates the shared learned canonicalizer as the upstream failure rather
than attributing it to either pose readout.

Both terminal classifications are
`source-view-canonicalizer-not-identified-on-seen-transforms`, with decision
`stop`. The spatial-moment run's lower physical error is a descriptive result
inside this consumed single-seed diagnostic only: both models remained worse
than the `1403.5191 um` constant-pose prior and failed the core pose gates. No
architecture is selected, no performance or accuracy claim is made, and
neither longer training of this learned canonicalizer nor gate relaxation is
authorized.

The only authorized next experiment is a matched exact-source-view oracle
canonicalization upper-bound diagnostic on the same consumed synthetic
construct. Source generation sampled with `inverse(view_h)`; the oracle must
sample the observed source with the exact forward `view_h` as its
output-to-input map, matching the proven asymmetric-marker direction test. It
bypasses the learned canonicalizer and may use exact synthetic rotation and
scale only, never anatomical pose targets. Its purpose is causal diagnosis of
whether perfect nuisance removal makes the downstream pose readout
identifiable; it remains development-only with all protected-data access
false and cannot select an architecture.

Exact source, config, setup, fixed-panel, input, result, raw-record, model,
canonicalizer, receipt, prediction, resume-state and stdout/stderr paths and
SHA-256 values are recorded in
`publication/supervised_similarity_pose_identifiability_diagnostic.yaml`; the
original run artifacts were not rewritten.

### Exact-source-view oracle canonicalization upper bound

The authorized oracle diagnostic ran from commit
`db6ed2163e1181dc8ddb1efdbee8461caa6d9286`. Its frozen config contracts
were `3202f03d7214ee4767ccbbecedd6b5e1f1e18529de8453e284785cded8976d87`
for the global-average pose readout and
`4b4b0ffc8f545dbe7ce631b9c8c3f6a9e6160f8aa2a2858f90c55f0a08e7552d`
for spatial moments. The two runs used the same consumed synthetic CCF train
examples, manifests, nuisance assignments, seed, 300-update schedule, losses,
optimizer and gates. Product-5, calibration and final-test access were false,
and learned-checkpoint dependencies were empty.

This upper bound did not estimate nuisance parameters. For each fixed panel it
used the exact synthetic rotation and scale to sample the observed source once
with forward `view_h` as the output-to-input map. Source generation had sampled
with `inverse(view_h)`. The model still received only the resulting image,
zero absent-outline mask and zero availability flag; it received neither the
nuisance parameters nor anatomical AP/L--R/D--V targets. The protocol is a
double-bilinear-resampling ceiling that retains source-view crop and
interpolation loss, not a deployable invariance result.

Oracle integrity passed identically in both runs: zero parameter mismatches,
zero non-finite canonicalized values, all four fixed panels present, and one
canonicalization warp per panel. Observed-input hashes and the four
canonicalized-source hashes were exact-equal between runs. Both runs completed
300 updates and 7,200 sample presentations with zero non-finite training or
evaluation outputs.

| Diagnostic | Seen AP / L--R / D--V bin accuracy | Seen AP / L--R / D--V MAE | Seen physical error | Held AP / L--R / D--V MAE | Held physical error / improvement over prior | Post-warm clipping |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global-average + exact oracle | 0.4166667 / 0.6875000 / 0.7291667 | 1429.7738 um / 8.3367 deg / 10.0016 deg | 1582.8687 um | 1548.4435 um / 11.7089 deg / 12.3831 deg | 1793.5379 um / -0.2778864 | 0.4185185 |
| Spatial moments + exact oracle | 0.3125000 / 0.8125000 / 0.8750000 | 1603.5231 um / 5.0932 deg / 4.6592 deg | 1763.1318 um | 1958.4974 um / 12.7465 deg / 7.0351 deg | 2143.8478 um / -0.5274803 | 0.3962963 |

Both runs failed every seen bin-accuracy gate, every held MAE gate, seen and
held residual-improvement gates, and the held physical-improvement gate. All
residual improvements were negative, and both held physical errors were worse
than the `1403.5191 um` constant-pose prior. Their terminal decision is `stop`
with classification
`pose-representation-not-identifiable-on-seen-panels-conditional-on-oracle-source-view-canonicalization`.

Because exact nuisance parameters, the proven forward transform direction and
all oracle-integrity gates did not rescue either readout, source-view nuisance
estimation is not the sole blocker under this fixed double-resampled protocol.
The remaining failure may reflect crop/interpolation information loss, pose
representation, optimization or the consumed diagnostic construct. The
descriptive differences between these two single-seed runs do not select an
architecture, support a performance claim or authorize promotion, longer
training, gate relaxation, or protected-data access.

Exact source and config lineage, matched observed-input and canonical-image
hashes, oracle-integrity evidence, full metrics, result and raw-record hashes,
model states, receipts, predictions, resume states and stdout/stderr paths and
SHA-256 values are recorded in
`publication/oracle_similarity_pose_identifiability_diagnostic.yaml`; the
original run artifacts were not rewritten.

### Frozen-context AP probe and trust-NCG solver rescue

The frozen-context AP transfer probe was frozen at commit
`2ae0f965beae50592847f9686a94ce1e77e18f9f`. Its global-average and
spatial-moment config contracts were respectively
`bcfaeac84717ae702f4b239f40004b3525bccb5301e0ddbdade3bae8cd47a2a5`
and `50650ba2244d1cd25cef1528cbd73afe2416f73f5f9736173d8f6540421b35d3`.
The probe froze the terminal 192-dimensional AP contexts from the two oracle
diagnostics, then fit only a 41-class linear AP head in float64 on CPU. Each
arm used the same two 24-case seen panels and the same three transfer fits.
Product-5, calibration and final-test access were false; the role was
development-only diagnosis, never model selection or promotion.

The original L-BFGS-B pair stopped numerically. All three global-average fits
reached the 1,000-iteration limit with independent final gradient infinity
norms between `3.0975291161320804e-4` and `8.453242841426724e-4`. The
spatial-moment opposite-panel fits returned SciPy success, but their independent
gradient norms remained `2.7879679012592147e-6` and
`2.180999848982589e-6`; its pooled-seen-to-fresh-held fit also reached the
1,000-iteration limit at `4.7691638577228085e-6`. Both receipts therefore
stopped as `ap-head-solver-result-not-successful`. Those runs could not support
a causal panel-instability interpretation.

Commit `e5b051a692778b3d6809e18f6a9b7377e88c9a64` added a solver-only
trust-NCG rescue, bound byte-for-byte to the consumed context tensors. Its
global-average and spatial-moment contracts were
`b73d5216666fe75171fdd73a37a3486e1134a4079e4ea1867959bec342f1df7a`
and `00780370c7855243f9fc36e9fa8798f39cb295f83471506bc60191085da44f78`.
All six fits converged successfully in 18--35 iterations, with independent
gradient infinity norms from `9.168150722112245e-13` to
`8.590920992473906e-10`; none exhausted the 250-iteration or 1,250
objective-gradient-plus-Hessian-vector call limits.

| Frozen context | Single-panel train correct | Pooled-seen train correct | Opposite-panel correct | Prespecified minimum | Authoritative classification |
| --- | ---: | ---: | ---: | ---: | --- |
| Global average | 24/24 and 24/24 | 48/48 | 21/24 and 18/24 | 23/24 each | `terminal-ap-context-panel-resampling-instability` |
| Spatial moment | 24/24 and 24/24 | 48/48 | 23/24 and 22/24 | 23/24 each | `terminal-ap-context-panel-resampling-instability` |

The rescue receipts pass every solver-validity, independent-gradient, budget,
operator-accounting, finite and training-fit check, and explicitly permit the
causal classifications above. They explicitly forbid fresh-held transfer
interpretation because opposite-panel transfer was a prerequisite and failed
for both arms. For provenance only, the global-average fresh-held bin-centre
AP MAE, residual-added AP MAE and prediction-to-truth SD ratio were
`1156.25 um`, `1148.7797760575388 um` and `0.6943662052101008`; the
spatial-moment values were `937.5 um`, `927.9678975505134 um` and
`0.8235325051784915`. These are non-authoritative descriptive values, not
held-panel performance evidence and not a basis for comparing the two arms.

Both decisions remain `stop`. No architecture is selected, no learned
candidate is promoted, and no protected-data access is authorized. Exact
per-fit solver results, all four run roots, config and contract bindings, and
receipt, tensor, stdout and stderr SHA-256 values are recorded in
`publication/frozen_context_ap_probe_solver_diagnostic.yaml`; the original
artifacts were not rewritten.

### Oracle pre-source-view atlas-pair energy diagnostic

The independently random-initialized atlas-pair energy diagnostic ran from
commit `2afea47fc7c113b84e577fdeee4a3a4dd9b6ef08` under frozen config contract
`506ebbb88857bd4242939220d0c98e3c5ac99c389690d051c2cab36adac89ccb`.
It tested a narrow causal premise: whether a 271,450-parameter energy model
could rank the true atlas plane when given the synthetic generator's
`moving_raw_uint8` tensor before any source-view transform. The source was
normalized and downsampled exactly once to 160 by 232; rotation was zero,
scale was one, the outline mask and availability flag were both all-zero, and
candidate pose coordinates were not scorer inputs. Training, development and
qualification used only clean synthetic CCF train-split data. Truth poses were
restricted to the prespecified interior support (500 um inside the full AP
domain and at most 25 degrees absolute tilt), so this diagnostic is not
evidence about the outer runtime domain. Product-5, calibration and final-test
access were false, and learned-checkpoint dependencies were empty.

All 1,500 optimizer updates were applied consecutively with AMP remaining at
512. The final and resume model states were exact-equal, all recorded values
were finite, stderr was empty, and the post-freeze qualification lineage and
receipt bindings passed independent audit. Execution integrity is therefore
`GO`. Scientific evidence did not pass: development truth-in-set top-1 was
only `10 / 48`, `12 / 48` and `16 / 48` at updates 500, 1,000 and 1,500,
against the frozen minimum of `46 / 48`.

| Qualification seed | Truth-in-set top-1 | AP / L--R / D--V MAE | Physical error / constant prior | Improvement | Ten-slice projected p95 | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1204322 | 14 / 48 | 276.627625 um / 1.717448 deg / 5.260417 deg | 319.831631 um / 1155.620127 um | 0.723238 | 11.154710 s | fail |
| 1304322 | 16 / 48 | 301.822937 um / 4.567708 deg / 5.625000 deg | 405.291410 um / 1208.670849 um | 0.664680 | 11.005519 s | fail |

Both untouched qualification panels passed the non-finite, invalid-render,
broken-pair control, order-equivariance, physical-improvement and runtime
checks. Seed 1204322 failed truth-in-set, AP and D--V accuracy; seed 1304322
failed truth-in-set and all three pose-accuracy gates. The terminal receipt is
`passed=false`, so the scientific premise decision is `NO-GO`. This is
immutable negative development evidence, not a benchmark, performance or
accuracy claim, architecture-selection result, learned-candidate selection,
or promotion authorization. It does not authorize protected-data access,
longer training or gate relaxation.

Exact commit, source, config, data, atlas, checkpoint, freeze, manifest,
development, qualification, receipt and stdout/stderr hashes are recorded in
`publication/atlas_pair_energy_diagnostic.yaml`; the original run artifacts
were not rewritten.

### Frozen source-scale 1.0 paired causal ablation

The development-only source-scale intervention ran from clean commit
`af3052c3e49c9b7af1d4603ce7c257b2377b5602` under frozen config contract
`dea56ad8ccfe8e42ff75a98cef8fbe5b2b11294b7bd1b92ca4543d53ad9d8a2a`.
It reused the final 271,450-parameter atlas-pair checkpoint and the same 48
ordered development realizations, fixed candidates and scorer. For each
realization, the treatment changed only the singleton synthetic child
manifest's source `scale` to exactly `1.0` and recomputed that manifest's
commitment. Rotation, translation, deformation, appearance, noise,
realization ID and seed were preserved. Every baseline scale differed from
one, every treatment raw and 160-by-232 source hash differed from its paired
baseline, and the source mask and availability flag remained absent/all-zero.

The baseline replay was exact: all 48 stored source hashes, candidate poses,
kinds and targets matched, all nine stored energy arrays per realization were
bit-exact, and normal/broken-atlas/broken-source top-1 counts remained
`16 / 0 / 1`. The treatment used the same frozen model, rendered candidate
tensors and within-condition broken-source derangement. Product-5,
qualification, calibration, final-test and training access were all false;
the training scale manifest was reference metadata only and was not read at
runtime.

| Prespecified check | Gate | Observed | Result |
| --- | ---: | ---: | --- |
| Baseline normal top-1 replay | exactly 16 / 48 | 16 / 48 | pass |
| Treatment normal top-1 | at least 24 / 48 | 15 / 48 | fail |
| Paired net corrections | at least 8 | -1 | fail |
| Exact two-sided McNemar | at most 0.01 | 1.0 | fail |
| Median paired truth-gap improvement | at least 0.25 | 0.02158522605895996 | fail |
| Strict truth-versus-joint-global pairs | at least 0.97 over exactly 144 | 144 / 144 = 1.0 | pass |
| Broken-atlas / broken-source top-1 | at most 12 each | baseline 0 / 1; treatment 0 / 1 | pass |
| Non-finite / invalid source or render | 0 / 0 | 0 / 0 | pass |

Two baseline errors were corrected (sample indices `4` and `42`), while three
correct baseline decisions regressed (`3`, `19` and `44`). In the lower
absolute-log-scale stratum, top-1 stayed `10 / 24` and the median paired
truth-gap change was `-0.015467524528503418`; in the higher stratum, top-1
fell from `6 / 24` to `5 / 24` despite a descriptive median gap change of
`0.19292253255844116`.

Execution and receipt integrity are `GO`, but the preregistered branch is
`causal_gates_fail` and the scientific scale-causality premise is `NO-GO`.
Setting source scale to one is therefore not a sufficient explanation for the
local discrimination failure. This consumed development result is not a
benchmark, performance or accuracy claim, architecture-selection result,
learned-candidate selection, or promotion authorization. It does not authorize
gate relaxation, protected-data access or further execution; a separate
spatial-aggregation diagnostic requires its own frozen contract.

Exact source, config, checkpoint, development, atlas, receipt and stdout/stderr
paths and SHA-256 values, plus the independently recomputed paired statistics
and access audit, are recorded in
`publication/atlas_pair_source_scale_ablation_diagnostic.yaml`; the original
run artifacts were not rewritten.

### Paired fixed-Haar spatial-aggregation causal diagnostic

The paired diagnostic ran from pushed commit
`a95420b3474bc106e1f6a783bdf9eb4af4879349` under frozen contract
`93fd5d04575e10e023d46735abc7e601b1234f8b5b54ac8cb080328dd9f8d6b2`.
It compared two independently trained, random-initialized 271,780-parameter
atlas-pair scorers. Both used the same 162 global mean/max correlation
statistics. The treatment additionally received 243 fixed top--bottom,
left--right and diagonal 2-by-2 Haar contrasts, while those inputs were exact
zero in the parameter-matched null arm. Complete initial states were
bit-identical. This tests only causal access to these fixed contrasts, not
spatial architectures generally. The null arm's dormant contrast columns also
create the prespecified functional-input-rank mismatch.

Training used only clean synthetic CCF train-split data, the oracle
pre-source-view image downsampled once to 160 by 232, and absent/all-zero
outline inputs. Product-5, calibration, final-test and protected-data access
were false, and learned-checkpoint dependencies were empty. All 1,500 paired
updates and 3,000 source presentations per arm are present; AMP remained at
512, paired-input and step-barrier checks passed throughout, and the final and
resume model states are exact-equal.

The original execution reached update 1,500 and wrote the immutable terminal
development result, but a concurrent read-only audit caused a transient
Windows file lock and `os.replace` failed while publishing the already-written
resume-state temporary file. Qualification had not begun. The valid update
1,475 checkpoint was retained. A replay of updates 1,476--1,500 was rejected
by the immutable-development guard because its CUDA receipt differed and none
of that replay was adopted. The original update-1,500 temporary state then
passed the committed resume, model, optimizer, scaler, RNG, development and
training-integrity validators before transparent promotion; its SHA-256
remained
`0b0dcc59d82f9ca472326703d25c02e9e1719330668f4a3851ed820f9886c359`.
A final runner loaded it without further training, froze both models, generated
only fresh seeds `1604322` and `1704322`, wrote the complete receipt, and
exited zero. Execution integrity is therefore `GO`, with recovery disclosed;
this is not an uninterrupted-execution claim.

Development treatment truth-in-set top-1 was `9 / 48`, `11 / 48` and
`11 / 48` at updates 500, 1,000 and 1,500, against the frozen `46 / 48`
minimum. The paired treatment never exceeded the null arm at these boundaries.

| Qualification seed | Fixed top-1 null / treatment | Net corrections / exact McNemar p | Treatment AP / L--R / D--V MAE | Treatment physical error / improvement | Ten-slice p95 null / treatment |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1604322 | 13 / 48 / 13 / 48 | 0 / 1.0 | 237.1745 um / 2.9635 deg / 3.2552 deg | 266.7706 um / 0.773972 | 10.5020 s / 12.2130 s |
| 1704322 | 14 / 48 / 13 / 48 | -1 / 1.0 | 158.7240 um / 1.9661 deg / 2.6940 deg | 189.5650 um / 0.808898 | 10.5899 s / 12.2199 s |

The treatment failed the central fixed-ranking and both paired causal gates by
a wide margin. Seed `1604322` additionally failed the treatment D--V gate.
Physical improvement and runtime passed on both seeds. The null arm had lower
descriptive free-search errors, but its strict absolute order-difference check
failed on both panels (`1.907e-6` and `2.861e-6` versus `1e-6`), despite
allclose energies, unchanged top-1 and zero decoded-pose difference. That
control warning further precludes a positive causal conclusion; it does not
explain away the treatment's `13 / 48` fixed-ranking results.

Both seed branches and the family branch are
`both-fail-insufficient-stop`. Fixed 2-by-2 Haar contrasts did not rescue local
ranking, no independent confirmation is authorized, and no architecture is
selected. This is a hash-bound negative development diagnostic, not a
benchmark, real-histology result, product-performance or accuracy claim, or
evidence against learned topology-preserving, recurrent, equivariant,
attention-based or deformation architectures generally.

Exact source, config, model, training, development, recovery, freeze,
qualification, raw prediction, receipt and artifact hashes are recorded in
`publication/atlas_pair_spatial_aggregation_diagnostic.yaml`. The preserved
update-1,475 checkpoint and the recovery-path mutation are part of that record.

### Paired native-topology causal diagnostic

The next diagnostic ran without interruption from pushed commit
`7ec5ce15f7d51a03dfc59ce269a62c947541ee8e` under frozen contract
`1f9c551bb3ac5837ac41e2692a5cc58a159c8a95326820c0fab23264bb51b4dd`.
It compared two independently stored, random-initialized 284,058-parameter
atlas-pair scorers with identical learned layers and exact-equal initial states
and outputs. Both processed the full correlation-plus-mask lattices with the
same small spatial CNN. The treatment saw native lattice layout; the control
saw one precommitted bijective random permutation at each scale, inverted only
after spatial processing. This isolates native rectangular neighbourhoods
within this architecture. Sparse random adjacencies remain in the control, so
it is not a topology-free network.

Training used 1,500 paired updates and only clean synthetic, interior
oblique-coronal CCF train-split data. Source tensors were generated once and
presented identically to both arms. All 1,536 off-centre convolution
coefficients per arm started at zero, received finite nonzero gradients from
the first update, and were finite, nonzero and arm-distinct by update 2 and at
the end. Complete training, optimizer, AMP, RNG, pairing, barrier, topology,
final-equals-resume and qualification-freeze validators passed. There were no
learned-checkpoint dependencies and no Product-5, calibration, final-test or
other protected-data access.

Development fixed-candidate top-1 was null/treatment `10 / 12`, `10 / 20`
and `9 / 15` out of 48 at updates 500, 1,000 and 1,500. The transient update
1,000 paired advantage did not meet the `46 / 48` absolute requirement and
did not persist.

| Qualification seed | Fixed top-1 null / treatment | Net corrections / exact McNemar p | Free-search physical error null / treatment | Treatment AP / L--R / D--V MAE | Ten-slice p95 null / treatment |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2104322 | 9 / 48 / 16 / 48 | +7 / 0.092285 | 363.924 um / 712.170 um | 644.857 um / 4.264 deg / 12.746 deg | 21.850 s / 20.945 s |
| 2204322 | 17 / 48 / 17 / 48 | 0 / 1.0 | 321.216 um / 618.447 um | 562.565 um / 4.270 deg / 11.137 deg | 17.563 s / 17.467 s |

Every one of the 96 unique fixed rows and both arms' four-stage free-search
outputs was independently recomputed from raw arrays and exact regenerated
source tensors. There were no invalid renders or non-finite values. Energy
allclose, top-1 and decoded-pose order checks passed for both arms on both
seeds; the largest descriptive reorder difference was `9.536743e-7`.

Both arms failed both fresh panels. The family branch is therefore
`both-fail-change-feature-or-candidate-construction-before-recurrence`:
native topology did not provide the prespecified reproducible causal rescue,
and recurrence is not licensed by this test. This is narrow evidence about the
fixed feature, candidate and scorer construction, not evidence that topology,
recurrence, attention or equivariance are useless generally. The treatment's
descriptively worse free-search errors are also not product-accuracy estimates.

The result is strictly coronal-family evidence. Sagittal, horizontal, grazing
and general oblique frames were not representable, so it cannot support the
required arbitrary-plane claim. Development now moves to a versioned
arbitrary-plane generator, structured 3-D slice-frame/QuickNII O/U/V geometry,
and revised features/candidates before any recurrent model is built. Exact
source, config, training, qualification, receipt, artifact and audit hashes are
recorded in `publication/atlas_pair_topology_diagnostic.yaml`; the frozen v2
sources and run artifacts remain unchanged.
