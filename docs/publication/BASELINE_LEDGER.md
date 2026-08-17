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
| Sealed DeepSlice comparison | 27.748 µm | 0.429° | 0.595° | absolute gates passed; comparator superiority gate failed |

On the sealed comparison, AtlasPose had lower paired absolute error than the DeepSlice MEns-AI-CI reference for AP and L--R. D--V superiority was inconclusive: the one-sided 95% upper confidence bound for the candidate-minus-reference D--V error was `+0.1891°`, and the joint probability of lower error on all three components was `0.6417`, below the frozen `0.95` requirement. Consequently `release_approved=false` and `promotion_ready=false`. It must not be described as globally superior to DeepSlice.

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
