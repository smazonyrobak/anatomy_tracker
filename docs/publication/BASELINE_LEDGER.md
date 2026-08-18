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
