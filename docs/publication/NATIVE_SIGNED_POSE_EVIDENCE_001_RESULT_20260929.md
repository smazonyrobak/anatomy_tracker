# Native signed pose evidence 001 — failed matched control

**Completed 2,000 updates; the predeclared gate failed, independently confirmed.**
Training terminal 70497 and independent CPU audit 4417 both exited 0 before their
outputs were inspected. Audit integrity passed and the local gate failed. Do not
promote or extend this control on the basis of its small coordinate improvement.

The sole intervention supplies six signed out-of-plane render-cost maps before
each shared pose update: ±3° on two normal-tangent axes and ±150 um normal
offset, through one initially zero projection. It retains the original whole
image-key 4000 parent, frozen retrieval features, the comparator's schedule,
losses, known PSFs, truth-near initialization, three updates and 2,000-update
budget. See the [frozen protocol](NATIVE_SIGNED_POSE_EVIDENCE_PROTOCOL_20260929.md)
and [matched no-probe result](NATIVE_RIBBON_LOCAL_001_RESULT_20260929.md).

All 384 development observations were retained; **270 eligible observations**
enter equal-held-subject macros, with 114 censored. The four held-out coherent
synthetic subjects share one atlas; this is neither biological validation nor
honest global retrieval/capture evaluation. The selected geometric initializer
uses the fixed identity-reflection tie; the oracle uses recorded reflection.

## Primary outcomes

| Eligible subject-macro metric | Geometric initializer | No probes | Signed probes |
|---|---:|---:|---:|
| Selected centre mean (um) | 2685.479 | 838.822 | 842.274 |
| Selected slab mean (um) | 2685.480 | 838.825 | 842.276 |
| Oracle centre mean (um) | 814.235657 | 716.556673 | 709.931486 |
| Oracle slab mean (um) | 814.237311 | 716.559497 | 709.934255 |
| Oracle fitted-plane normal angle (°) | 6.138344 | 6.102826 | 6.101891 |
| Reflection accuracy (%) | 40.8565 | 94.8141 | 94.3563 |

Oracle centre/slab improve only **12.81%**, versus the required 20%; normal
error improves **0.5939%**, also versus 20%. Relative to no probes, signed
evidence reduces oracle centre/slab by just **6.6252 um** and normal angle by
**0.00093485°**, while selected centre/slab worsen **3.4518 um**. No seed-replicate
or statistical superiority claim is made from this single matched comparison.

Five saved conditions fail: the three oracle overall 20% reductions, plus
oracle-normal nonregression for held-out subjects 2 and 3. Selected centre/slab
overall reductions pass, as do all dense-coordinate per-mode/per-subject
nonregression conditions. Finite-output and exact frozen-retrieval-tensor checks
pass. Saved canonical topology/orientation checks pass; the audit authenticates
them but does not independently replay topology. Thresholds were not relaxed.

## Presentation modes and held-out synthetic subjects

Oracle values below are initializer → signed endpoint; distances are um.

| Scope | Eligible rows | Centre mean | Slab mean | Normal angle (°) |
|---|---:|---:|---:|---:|
| Raw | 91 | 816.780 → 713.774 | 816.782 → 713.777 | 6.144613 → 6.137924 |
| Exact black | 91 | 816.780 → 704.569 | 816.782 → 704.572 | 6.144613 → 6.083557 |
| Imperfect brush | 88 | 809.252 → 711.697 | 809.254 → 711.700 | 6.125254 → 6.082796 |
| Subject 0 | 64 | 852.411 → 698.899 | 852.412 → 698.902 | 5.826826 → 5.646039 |
| Subject 1 | 66 | 783.867 → 690.606 | 783.869 → 690.609 | 6.054569 → 5.983149 |
| Subject 2 | 75 | 842.889 → 767.573 | 842.892 → 767.577 | 7.008387 → 7.114619 |
| Subject 3 | 65 | 777.775 → 682.647 | 777.776 → 682.649 | 5.663596 → 5.663759 |

Normal initializer gains are only .006689°/.061057°/.042458° for raw/black/brush.
Raw is actually .002642° worse than the matched no-probe endpoint. Subjects 2
and 3 regress .106232° and .000163° from the initializer; the latter is small
but the frozen gate has no tolerance waiver. Subject 0's 18.01% centre reduction
is the largest, still below 20%.

Selected centre endpoints are 863.374/824.515/836.709 um for raw/black/brush.
Their differences from no probes are −13.257/−15.106/+39.533 um. Thus even
selected performance does not improve consistently across presentations.

The oracle centre's fitted-plane-normal error component changes only
**355.756 → 351.141 um**, versus **684.815 → 566.602 um** tangentially. Together
with the near-unchanged fitted normal, this does not demonstrate meaningful
out-of-plane capture. This is a descriptive endpoint decomposition, not a
zero-field counterfactual or causal attribution of pose versus deformation.

## Decision and frozen provenance

The explicit six-channel injection did not solve plane learning under this
matched budget. The result alone cannot distinguish weak frozen feature-cost
directions from an ineffective readout/optimization path. A separate small
TRAIN-only oracle cost-direction diagnostic addresses that distinction; its
results are reported separately, not folded into this fixed gate. It is not
another quality trial or permission to tune against these development subjects.
No uncertainty calibration, public benchmark, model merging or deployment claim
follows.

Independent NumPy coordinate recomposition agrees within **.00581836 um**;
saved mean metrics agree within **.00187565 um**, and fitted-plane normal angles
within **5.7619e-6°**. The largest diagnostic weighted-p95 difference is
**2.76136 um**, from discrete-CDF tie selection; p95 is not a gate. The audit
verifies source/parent/data bindings, initial RNG and matched schedules, the
2,000-update budget, raw identities, and exact frozen retrieval tensors.
Topology is authenticated from saved results, **not independently replayed**.

Frozen run: `I:/AnatomyTracker/runs/joint_v6_signed_pose_evidence_001`.

- Completion SHA256:
  `7e814be7b2207e54544b67f6180a61808de17ea6bc530b41930092ced90828c1`.
- No-probe completion SHA256:
  `a2bee6dc1ffcaafcef3b4ad4a49223653f582f41ab071a3ca25928e7db7a83d7`.
- Whole parent checkpoint SHA256:
  `d4d706e8d80e53a3638a70e79ce8661ff4af41f7b846143aa1ec68372bfb2ae5`.
- Cohort completion SHA256:
  `ba51982a5b03b61d4bcf7f37f2c139dd6c1caf7ff66a6124cb678ab5ee9dd1f2`.
- Independent `joint_v6_signed_pose_evidence_001_independent_audit/audit.json`:
  `7acdc3ef87f85691b296f67fc828428aa090b1afed88a5ce05853a6e81853ad5`.

The run reports unchanged source, 2,000 applied optimizer updates, and elapsed
3653.4365 seconds. Concurrent timings are not an inference/hardware benchmark.
