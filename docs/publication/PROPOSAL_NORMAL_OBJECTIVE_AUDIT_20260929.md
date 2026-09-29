# Extra normal supervision did not solve coarse localization

Frozen 005 completed 20,000 attempted / 19,996 applied updates. Independent CPU
recomputation shows a modest normal improvement over 003, but poor capture
persists and some endpoints worsen. **Scientific gate remains FAIL** for
ordinary joint refinement, deployment and public benchmarking. No new inference,
GPU work or external benchmark was performed in this audit.

The comparison uses the same 640 development sections and 40 organizational
groups from one atlas, not 40 biological animals. Metrics are group-macro means;
raw probabilities are uncalibrated. Angles use the absolute antipodal plane
normal, without post-hoc anatomical ML-reflection equivalence.

| Final endpoint |003: joint-cell NLL |005: joint-cell + normal-marginal NLL |
| --- | ---: | ---: |
| Joint-cell NLL, nats |8.09677 |8.10757 |
| Normal-marginal NLL, nats |5.18413 |5.06250 |
| Joint-MAP normal error |43.7265° |42.1356° |
| Normal-marginal MAP error |43.4609° |41.1715° |
| Joint-MAP within 10° |11.2500% |12.1875% |
| Normal-marginal MAP within 10° |11.4063% |12.9688% |
| Normal probability mass within 10° of truth |7.0718% |7.8975% |
| Normal-offset error |2082.02µm |2261.68µm |
| Exact nearest-cell top-32 recall |10.3125% |10.7813% |
| Exact nearest-cell top-128 recall |27.3438% |28.9063% |

Joint-MAP is the highest-probability normal/offset/roll cell; normal-marginal
MAP first sums over all offsets/rolls. Neither is anatomical landmark accuracy.
The additional objective has weight 1.0. These single-seed results do not
establish a general improvement or identify the cause of poor learning.

## Support and input-mode checks

| Subset |Rows |Joint-MAP normal error,003→005 |Joint NLL,003→005 |Top128,003→005 |
| --- | ---: | ---: | ---: | ---: |
| Support-eligible |611 |43.51°→41.62° |8.0972→8.0593 |27.02%→29.13% |
| Censored low support |29 |52.13°→48.32° |8.5812→8.9202 |28.43%→25.49% |
| Brush absent |213 |46.55°→43.18° |8.4510→8.1430 |21.42%→28.33% |
| Accurate brush |214 |42.24°→41.78° |7.5974→7.5319 |34.42%→35.92% |
| Imperfect brush |213 |42.57°→41.48° |8.2139→8.6470 |27.00%→22.33% |

The imperfect-brush trade-off is adverse despite lower mean normal angle.
Low-support metrics are descriptive, not evidence that those rows have an
identifiable point pose. The legacy JSON name `identifiable` denotes support
eligibility only. Censored rows remain present; they were not removed from
coverage or aggregate evaluation. Full subset normal NLL, probability mass,
within 10°, and support-by-mode results are retained in the audit JSON.

## Integrity and comparison limits

- Step-zero model parameters, optimizer, scaler, and Python/NumPy/CPU/CUDA RNG
  states match exactly. Numeric generated schedules and frozen-row order match;
  the five generated ID fields differ only by run namespace. Frozen identities
  and development records are unchanged with zero train/development ID overlap.
- Both scheduled 160,000 new renders, including all 98,304 catalogue cells.
  Both recorded 159,071 support-eligible and 929 censored generated observations.
  Four AMP skips occurred in each run, but at different batches: 003 skipped
  14209/16266/18718/19155; 005 skipped 10374/14991/17375/19835. Thus planned
  exposure is matched, but the exact applied sample sequences are not identical.
- All final raw predictions and model tensors are finite. Maximum absolute
  log-normalization error is 4.69e-7 for 003 and 5.46e-7 for 005. Saved cell ranks,
  predictions and identities match; float64 normal-NLL recomputation differs
  from saved float32 reductions by at most 6.08e-6 per row.
- Exactly 34 proposal-path tensors changed in each run; final optimizer counts
  agree with trace and completion records. Last-100 weighted joint training
  NLL is 6.78992 versus 6.75583; held-out joint NLL did not improve.
- The joint-model source also acquired an opt-in uncertainty branch between
  runs. It defaults to `None`, is absent from these model configurations, and
  is never reached by proposal-only inference. Exact initialization/RNG equality
  confirms no extra parameter draws. The optional 384-normal residual readout
  was disabled and untrained in 005; this comparison is not its evaluation.

## Reproduction and receipts

The flat CPU script is
[`training/audit_joint_v6_normal_objective.py`](../../training/audit_joint_v6_normal_objective.py).
The full result is
[`results/proposal_normal_objective_audit_003_005.json`](results/proposal_normal_objective_audit_003_005.json).
Original audit and per-row arrays remain in
`I:/AnatomyTracker/runs/joint_v6_proposal_normal_objective_audit_003_005/`.
The original JSON SHA-256 is
`12b4ec26bc48222ea331ed3ee5cbdbd301e5e56ca1ed4703263d7ff571d0718a`;
the repository copy changes only text line endings/final newline and has SHA-256
`b5f27c15c9f1d563e850b65e7ce9018149582d9b139c492889a97b78626f674a`.

| Frozen005 artifact |SHA-256 |
| --- | --- |
| Final raw log probabilities |`b2e9f03d293c1e1c871bd153db50f506b7260a209ceb5aa57a9095285e921133` |
| Final checkpoint |`edbd96a8eefbe5b2ae004a557a249af9196a5f1deea3ae2bed720dedec84eb19` |
| Configuration |`3b85dc22322214d9478a1fe8abb4a1f441a8b357633d70d68fb63189ff80171a` |
| Generated schedule |`9685c7074284cd9a31aede25a9b03cbbe47c28623bf2f24e08f69704663c0e69` |
| Trace |`291c7a6d1045d3896691bf413404cfec903a85cf0ef89488b2824043831b7922` |

This result does not establish biological generalization, calibrated electrode
uncertainty, successful joint deformation, or superiority to DeepSlice.
