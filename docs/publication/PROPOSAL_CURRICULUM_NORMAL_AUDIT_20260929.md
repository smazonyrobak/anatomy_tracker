# Completed curriculum003: normal capture improves, remains inadequate

CPU-only audit after confirmed exit 0. Run003 completed 20,000 attempted /
19,996 applied updates, 160,000 newly rendered observations, and coverage of all
98,304 catalogue cells (159,071 generated point-supervised; 929 censored).
The prepared audit at commit `a1aef9a` recomputed initial/final normal marginals
from the frozen full-cell log probabilities; no model inference or GPU ran.

All main-table values are macro averages across the same 40 **organizational
synthetic groups**, 640 sections from one Allen atlas. These are not independent
biological subjects. Normal error uses antipodal plane normals. Exact
log-sum-exp gives `joint NLL = normal NLL + offset|normal NLL + roll|normal,offset NLL`.

| Metric | Uniform | 001, 10k | Corrected 004, 10k | Curriculum003, 20k |
| --- | ---: | ---: | ---: | ---: |
| Joint NLL | 11.49582 | 9.580307 | 9.756497 | 8.096767 |
| Normal NLL | 5.95064 | 5.685893 | 5.827418 | 5.184132 |
| Offset given normal NLL | 2.77259 | 1.980596 | 2.001566 | 1.493639 |
| Roll given normal/offset NLL | 2.77259 | 1.913818 | 1.927513 | 1.418996 |
| Marginal-normal MAP error, degrees | — | 50.475148 | 46.262714 | 43.460871 |
| Joint-cell MAP normal error, degrees | — | 50.829943 | 47.641659 | 43.726498 |
| Marginal MAP within 10 degrees | — | — | 7.34375% | 11.40625% |
| Posterior normal mass within 10 degrees | — | — | 3.84125% | 7.07182% |
| Best normal among top eight, oracle error, degrees | — | — | 34.215151 | 25.738121 |

003 learns more normal information, not just offset/roll. Nevertheless, 77.45%
of its joint-NLL improvement over uniform still comes from conditional
offset/roll. Only 11.4% of marginal MAP normals are within 10 degrees, and even
oracle choice among the eight highest-probability normals averages 25.7 degrees.
The latter is candidate coverage with truth-based selection, **not achieved
prediction accuracy**. Switching joint MAP to marginal-normal MAP barely changes
003's error. Target-normal versus nearest-geometric-normal NLL is 5.1841 versus
5.1952; catalogue quantization is not the main deficit.

Removing the 29 censored rows leaves 611 identifiable rows, normal NLL 5.1718,
marginal MAP error 43.3503 degrees and 11.73% within 10 degrees. Low-support rows
therefore do not explain the failure. Identifiable mode-specific synthetic-group
macros are:

| Input mode | Rows | 004 normal NLL / error | 003 normal NLL / error |
| --- | ---: | ---: | ---: |
| Raw, brush absent | 207 | 6.0203 / 49.5527 degrees | 5.3685 / 46.7139 degrees |
| Accurate brush | 204 | 5.4485 / 41.8957 degrees | 4.9468 / 41.3803 degrees |
| Imperfect brush | 200 | 5.7846 / 46.5129 degrees | 5.2139 / 42.3744 degrees |

Improvements are descriptive, **not an isolated causal curriculum estimate**:
003 is fresh, uses new rendered/frozen mixtures and twice the attempted updates;
004 continues corrected 002 weights from step 3,000. 001 has a different model
capacity and predates the precision correction. No biological generalization,
calibration, electrode-location probability, final benchmark or shipping claim
follows from these diagnostics. Joint recurrent/deformation training is still
not qualified by this coarse-capture result.

## Narrow label and renderer audit, 2026-09-29

No actual image/normal-label mismatch was found in the traced paths. This is a
bounded source/coordinate audit, **not certification of the entire pipeline**.
It does not exclude anatomical ambiguity or inadequate learned image features.

The frozen 003 `experiment_source.py`, lines 105–106 and 201–204, renders the
catalogue state indexed by the same scheduled cell ID used for supervision.
Its finite nine-point symmetric PSF is explicit; subsequent intensity inversion,
gamma/gain, backgrounds/noise and brush dilation/erosion do not rotate the frame.
The sole generated spatial augmentation, line 248, flips all input channels
horizontally together: the allowed raster-reflection nuisance, not a different
physical plane. No unlabelled rotation/crop or segmentation step was found.
All eight saved `first_generated_batch.pt` states equal their scheduled
catalogue states exactly (IDs 71574, 23303, 65674, 38498, 45395, 19086, 47061,
72699). Catalogue stored normals match decoded frame normals to 3.33e-16.

Frozen-row preparation uses `canonical_effective_quicknii_ouv_float64`
(`training/run_joint_v6_proposal_experiment.py:109`); image channels and truth
come from the same authenticated row. The physical OUV conversion and coupled
normal/offset/roll assignment are in
`training/arbitrary_plane_training_data_v6.py:159,187`: normal distance is
`acos(abs(n_truth dot n_cell))`, with the same sign used for the plane offset.
The renderer (`training/arbitrary_plane_full_frame_primitives.py:275–287`)
uses x/W, y/H raster coordinates and physical voxel-centre conversion before
ML/DV/AP grid sampling. No inconsistent axis permutation was found.

CPU-only arithmetic on the completed 001 catalogue/prepared development tensors
compared those continuous physical truth normals with assigned cell normals,
restricted to the 611 rows with positive pose weight. Assigned errors were
median **3.004925°**, p90 **4.200981°**, maximum **5.653778°**; nearest-normal
errors were median 2.997780°, p90 4.096249°, maximum 5.618861°.
The catalogue cover radius is 5.896075°; coupled assignment adds median 0°,
maximum 1.873536° over nearest-normal assignment. Thus the **43.726498° model
error is not a roughly 3° label-quantization error**. These are row-level
quantization statistics, not the group-macro model metrics above.

Proposal preprocessing (`training/arbitrary_plane_recurrent_model_v6.py:112`
and `training/arbitrary_plane_recurrent_model.py:317`) resizes to the configured
96-pixel raster, then applies convolution/GroupNorm; it does not canonicalize
orientation. The 8×8 spatial pool and 64-dimensional context compression
(`training/arbitrary_plane_coarse_proposal_v6.py:199`) remain plausible capacity
limitations, not a demonstrated label bug. This audit used no model inference,
GPU, new experiment, active-005 artifacts or modification of frozen outputs.

## Next focused training decision

Run the smallest controlled objective intervention first:
`weighted_mean(joint_NLL + normal_marginal_NLL)`, coefficient one on the extra
normal term, using exactly 003's fresh initialization seed, model, generated
and frozen row schedule, support censoring, optimizer and training horizon.
Use the normal of the existing joint catalogue label; changing to a different
label simultaneously would confound the objective control. This effectively
doubles the normal term while preserving the full joint conditional objective.
It does not certify calibrated probabilities.

Track marginal-normal NLL, MAP angle, 10-degree mass/recall and top-eight normal
capture alongside joint NLL and all three input modes. Accepting only a lower
joint NLL would repeat the old diagnostic mistake. If extra normal weighting
fails to improve physical normal capture over matched 003 checkpoints, the
next capacity control should replace the low-dimensional smooth normal scoring
basis with a direct expressive 384-normal readout, preserving offset/roll
conditionals; deeper spatial image encoding or render-based retrieval follows
if even training normals cannot be captured. Do not bundle these changes into
the first objective experiment.

## Frozen receipts

Run directory: `I:/AnatomyTracker/runs/joint_v6_proposal_curriculum_003`.
Audit directory: `I:/AnatomyTracker/runs/joint_v6_proposal_curriculum_003_normal_capture_audit`.
The audit retains per-section normal probabilities and metrics, including
censored/mode/group subsets; frozen run artifacts were not modified.

| Artifact | SHA-256 |
| --- | --- |
| 003 `experiment.json` | `43f78e82b3d550cabd3f3a974d47d5d5256d9a9126a309cedb15b223eec70b07` |
| 003 `completed.json` | `e0998299a2cd7ca02ebf0e2c344b2684ad396ef8901e1f31724aaa533fd53733` |
| 003 raw log probabilities, step 0 | `e0bf0c6d95bb15eaf3092810099d6a84e56e7165d137a50f507cafa71252e02c` |
| 003 raw log probabilities, step 20,000 | `1f7aa72e7e0d564c6024d2b9f8ce0e37ffffe106329eaf130e6ee147868f2dbe` |
| Audit `summary.json` | `cf593e3a21f3ceab8a38bddb1fd08f38e90b62cfc11fe6dee20bc9d94c0b7bd0` |
| Audit script | `1710265205b76540e4146ddad1511c2e9e849384db8e8db00b5e3b6c5dad03f7` |
| 004 raw log probabilities, step 10,000 | `eebc70438f28d189b9345ba027e9fb865b4dcab71f45bc57b451fabf3b84759a` |

004 was decomposed separately on CPU with the same float64 log-sum-exp,
catalogue ordering and grouping; its raw-array receipt matches the prior
precision audit. Shared development/cache receipts and 001 source results are
in `PROPOSAL_NORMAL_CAPTURE_DIAGNOSIS_20260928.md`; the correction/lineage details
for 004 are in `PROPOSAL_PRECISION_RECOVERY_AUDIT_20260928.md`.
