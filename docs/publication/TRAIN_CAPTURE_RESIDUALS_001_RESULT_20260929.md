# TRAIN catalogue starts versus the native initializer

CPU analysis86237 exited0 from `a35038e`, using the independently audited whole
coarse A22576 endpoint. No model inference, GPU, DEV cases or benchmark was used.
All5,120 synthetic and1,280 real TRAIN rows and their original IDs are retained.
This measures flat acquisition-frame residuals, not curved-surface accuracy.

## Main result

The old local teacher initializer does not cover actual retrieved frames.
Among eligible rows,95.15% of synthetic top1 starts and100% of real top1 starts
exceed at least one component of the old LARGE generation box. Even the
truth-assisted best finite frame among128 cells/both representations exceeds
that box on83.63%/100% respectively. This is an exact **target-to-start**
composition comparison; inverse correction coordinates are saved separately.

| Eligible TRAIN readout | Synthetic | Real weak-affine reference |
|---|---:|---:|
| Rows | 4,843 | 1,233 |
| No physical-plane candidate in top128 | 61 | 0 |
| Top1 predicted-R four-corner RMS, median | 6.650mm | 2.944mm |
| Top1 predicted-R four-corner RMS, group/donor mean | 6.857mm | 7.249mm |
| First near-plane cell, predicted-R RMS, median | 1.691mm | 2.810mm |
| Best-frame cell/R oracle RMS, median | 1.520mm | 2.031mm |
| Best-frame cell/R oracle tangent-centre distance, median | .611mm | 1.436mm |
| Top1 predicted-R differs from best-frame R for that cell | 19.20% | 46.72% |

The first-near readout is conditioned on existence (4,782 synthetic and1,233 real
rows). The near and best-frame selections use reference truth **for TRAIN
diagnosis only**; neither is an available inference selection rule. A correct
antipodal plane does not establish roll, reflection, tangent centre, image span
or anatomical warp. Retain both R alternatives, not the coarse argmax R alone.
The median/mean discrepancy in real top1 RMS exposes a substantial bad-reflection
tail, not a numerical inconsistency. These RMS values include predicted R;
earlier reflection-minimized plane/frame summaries are different quantities.

For real best-frame starts,95.70%/98.22% exceed the old600um teacher translation
box along its two transported axes, and98.22% exceed the old.12 log-V-span box.
Median required log-V-span correction is.16164 (p95 .19538). Scale and tangent
coverage remain necessary even with oracle candidate/reflection choice.

All rows outside supervision eligibility remain saved:277 synthetic and47 real.
Across all rows,167 synthetic and13 real have no near-plane candidate at128.
Organizational synthetic groups share one atlas and are not biological animals;
the58 real donors have weak upstream affine references, not dense warp truth.

## Coordinate checks and scope

Saved GPU top128 order and R probabilities are used without CPU reranking.
Synthetic prepared states are canonical; their independently authenticated row
reflection is applied once. Each candidate R is then undone on the observed
target for fixed-R composition analysis. Pixel centres use0..95 and physical
x/96,y/96, including the finite-centre correction. Plane offset at the fixed
support origin is reported separately from finite-centre normal displacement.

Independent FP64 inverse/recomposition of the archived nine-coordinate update
agreed within9.186e-11um across both datasets. No exact-antipodal ambiguity
occurred in the selected branches. Necessary three-update normal/offset/log-span
bounds are reported, but do not establish reachable poses; no cumulative
translation/roll/shear reach claim is made. Three-step failures for a predicted
wrong R do not imply that its separately retained opposite R is also unreachable.

Next: once useful local plane learning is established, continue **this whole
coarse model**, using coherent acquisition views and actual TRAIN candidate
starts, with explicit broad pose capture before deformation. Keep hard
distractors/failures and paired raw/black/imperfect-brush presentations. Do not
splice the separate native-control refiner, silently enlarge topology limits,
or evaluate with reference-chosen candidates. This analysis alone does not
choose a new update budget or demonstrate that broader training solves the gap.

Output: `I:/AnatomyTracker/runs/joint_v6_train_capture_residuals_001`.
The summary retains complete input/output bindings, raw residual arrays and IDs.

- Summary SHA256: `17c1db469633ca7b070e2921fb5f5dc6ff1de27aa37a8759f6b54de16cceb1e6`.
- Archived analysis: `8c4ebd1921f19a6965723ed461804ab04545f25445ab28531a61b2327dcc29d6`.
- Synthetic residuals: `9713335acffc669dfdb04630b3f58476bba921ad38fde39b60428d4cc0564444`.
- Real residuals: `e67c937dfaf81257da1991438ff21da51450f7a9e7e17846bba4f5f5837b9581`.
