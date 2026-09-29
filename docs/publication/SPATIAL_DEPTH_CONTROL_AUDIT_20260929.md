# Spatial depth control: extension gate failed

Frozen 007 finished all 4,000 attempted updates without AMP skips. The CPU
audit independently confirms failure of both predeclared coarse-only extension
criteria: marginal-normal error needed to decrease by at least 5° from 005,
and normal NLL by at least 0.25. Instead the angle worsened by 0.1984° and NLL
improved by only 0.09116. **Do not extend or select 007.** This engineering
decision does not qualify any model for deployment or external benchmarking.
See the prior [control protocol](SPATIAL_DEPTH_CONTROL_007.md).

| Matched 4,000-step endpoint |005 base encoder |007 six residual blocks |
| --- | ---: | ---: |
| Joint-cell NLL |9.75639 |9.80774 |
| Normal-marginal NLL |5.62423 |5.53306 |
| Normal-marginal MAP error |48.4800° |48.6784° |
| Joint-MAP normal error |49.6611° |48.6504° |
| Normal-marginal MAP within 10° |6.5625% |4.6875% |
| Joint-MAP normal within 10° |5.9375% |5.1563% |
| Normal probability mass within 10° |2.8866% |3.1180% |
| Exact nearest-cell top-32 recall |1.7188% |0.6250% |
| Exact nearest-cell top-128 recall |5.1563% |4.5313% |

These are group-macro results over the same 640 sections / 40 single-atlas
organizational groups, not biological animals. Normal capture is absolute
antipodal-normal geometry, not a truth-selected ML-reflection minimum, landmark
registration, or electrode localization. All probabilities remain uncalibrated.

## Source components and brush modes

Entries show **005→007**. Source components come from the authenticated pose/joint
ID namespaces; complete component×mode statistics are in the JSON.

| Subset |Rows |Marginal-normal error |Normal NLL |Marginal MAP within 10° |
| --- | ---: | ---: | ---: | ---: |
| Pose component |384 |47.13°→47.40° |5.5903→5.5495 |6.77%→5.47% |
| Joint component |256 |50.50°→50.60° |5.6752→5.5085 |6.25%→3.52% |
| Pose, support-eligible |380 |46.95°→47.13° |5.5908→5.5471 |6.83%→5.49% |
| Joint, support-eligible |231 |51.13°→49.49° |5.6690→5.4741 |6.62%→3.97% |
| Censored low support |29 |50.72°→60.32° |5.7243→5.8041 |2.94%→0.00% |
| Brush absent |213 |48.94°→50.62° |5.8305→5.7950 |5.08%→4.33% |
| Accurate brush |214 |45.90°→47.36° |5.4422→5.3426 |9.92%→7.08% |
| Imperfect brush |213 |50.72°→48.09° |5.6018→5.4487 |4.42%→2.75% |

Poor capture persists in both source components after censoring is excluded;
it cannot be attributed solely to the 29 low-support rows. Support eligibility
is not proof that every image is informative or depicts a whole section. The
prepared payload stores channels, weight, truth, label and IDs but no finite
support/visible masses. Therefore occupancy bins are unavailable here; we did
not invent them from image intensity or brush-boundary channels. These data
do not justify dismissing errors as intrinsic ambiguity.

## Integrity

All original step-zero model tensors, outer CPU/CUDA/Python/NumPy RNG states,
and scaler match. Exactly 48 residual-branch tensors are added; each block's
final convolution starts at zero. All numeric rendering schedules and frozen
row schedules match, and generated IDs differ only in run namespace. Acquisition
settings match; planned totals intentionally differ because 005 continued to
20,000 steps while this control ended at 4,000. The compared prefix is identical.

Both checkpoints report 4,000 applied updates. Raw probabilities and final
model tensors are finite; maximum log-normalization error is 8.72e-7 / 1.01e-6.
Saved identities and metrics agree with recomputation, with largest macro
difference below 1.8e-7. The 82 changed 007 tensors include the new spatial
branches. Last-100 training joint NLL is 9.6280 / 9.6468; the added depth has
not demonstrated useful improvement within this controlled budget.

The first audit attempt had two audit-only predicate mistakes: matching `.5.`
also selected non-final layers of block five, and full generator metadata
comparison included the intentionally different planned totals. These were
corrected, the CPU audit exited successfully, and no frozen results or metrics
were changed. No GPU inference or activation investigation was performed.

## Reproduction

Script:
[`training/audit_joint_v6_spatial_depth_control.py`](../../training/audit_joint_v6_spatial_depth_control.py).
Full result:
[`results/spatial_depth_audit_005_007.json`](results/spatial_depth_audit_005_007.json).
Original JSON and row metrics remain in
`I:/AnatomyTracker/runs/joint_v6_spatial_depth_audit_005_007/`.
Original audit SHA-256:
`74e5bc03109d01ac5d6da5f46513fcb91dc8c2a97337511e7be79dc048597d74`.
The repository JSON changes only text line endings/final newline and has SHA-256
`a1bc55cbcd74ccb8ba3daaa6e2cff1ee0ccf70ad3c226f7085401e00c50b7c6d`.

007 checkpoint SHA-256:
`279f244b20cd6f45fc8651662009ad26d761e27f3410d11e5a9543f0f15ab2ee`.
007 raw development predictions SHA-256:
`ea13286c8b29d7622391ef132f09e6e75547fbff9792bb3106ed7ff961f32aae`.
