# Matched-budget continuous-plane chart bridge

Prepared, not yet executed. Driver: `training/run_joint_v6_canonical_bridge.py`.
Output: `I:/AnatomyTracker/runs/joint_v6_canonical_bridge_001`.

The training-only closure diagnostic found good exact-anchor retrieval but poor
full-gallery capture. All256 real training frames have no catalogue candidate
within the declared1mm finite-frame/10deg tolerance. Nearest-frame squared error
is dominated by tangent centre displacement and span mismatch, not shear. This
motivates testing chart transfer, not declaring the entire failure solved.

## Intervention and matched control

Both arms start from the identical whole own-lineage A6000 checkpoint
`280836b65fb6db22930c8ee268eb4a880997c7858fb91a1e31ad6c7c94937eb2`,
with its exact AdamW state and RNG. No encoder merging, old weights or teachers.
Use the previous real+synthetic control's exact2,000-update schedule, seeds,
8real+8synthetic batches, candidate-cell draws, FP32, LR.001, weight decay1e-4,
gradient clip5 and trainable stems/shared encoder/image-key descriptor. All other
parameters stay frozen. Each arm independently ends at cumulative step8000.

For continuous affine O,U,V, let u=normalize(U), v=normalize(V−u(u·V)), n=u×v,
c=O+(U+V)/2, and S be the catalogue's declared support origin. The canonical
chart has c0=S+n[n·(c−S)], U0=12000u, V0=12000v, O0=c0−(U0+V0)/2.
This preserves the exact physical plane and U-axis roll while removing tangent
centre displacement, noncanonical span and shear. No nearest-cell label or
additional half-voxel correction is introduced. O/U/V use the existing96-pixel
extent convention; finite pixel-centre edges remain0..95.

- A: each real positive is the matched-affine chart.
- B: a frozen seed2026092920 schedule chooses the matched or canonical chart for
  each presentation, exactly8,000 of16,000 presentations each. This samples two
  positive log-losses; it does not log-sum positives and let the easier chart
  hide failure on the other. Preserve roll cues and both horizontal reflection
  representations. Query pixels and query inputs remain unchanged.

Pre-render and preserve both atlas charts for every training row with the same
50um nine-point normalized PSF as the gallery. This is an engineering assumption,
not a measured optical PSF. Cache atlas images only; descriptors use current
trainable weights on every update.

Two common corrections apply to BOTH arms, so their effect is not confused with
the chart intervention: a real pair has positive training weight only when BOTH
charts have at least64 support-pixel mass; invalid real anchors are not negatives
for other real queries. All256 rows remain in schedules, saved outputs and
unfiltered evaluation, including empty-anchor cases. This is weak-reference
eligibility, never automatic segmentation of a user's input.

Exclude other real-loss keys with antipodal normal<=10deg AND sign-aligned normal
offset<=500um regardless of finite chart. Always retain the own positive. This
prevents physically equivalent chart variants from becoming false negatives.
The synthetic objective/negatives/weights remain unchanged. Total loss is
.5 times eligible-weighted real paired NCE plus.5 times synthetic sampled NLL.
Zero eligible mass contributes zero without redrawing. Preserve losses by chart,
weights, exact IDs, schedules and both rendered supports. No geometry or IDs
enter the query encoder.

## Frozen endpoints and decision

At each final checkpoint rebuild the same98,304-cell/two-reflection gallery.
Evaluate all256 training rows/58donors, unchanged640 synthetic development rows,
and unchanged64 real development rows/six donors once. Save full component/cell
scores, descriptors, raw IDs, top128 candidates, geometry and source/checkpoint/
gallery hashes. Report training all-row and common-eligible donor macros,
matched-anchor and canonical-anchor retrieval separately, and predicted-cell
finite-frame error as a downstream diagnostic. It cannot be a1mm coarse-stage
success gate when the catalogue contains no such candidate for any training row.

Keep the previous fixed improvement gate versus A6000: real development
equal-donor normal error improves by at least5deg AND top32 physical-plane
capture improves by at least.10 absolute. Synthetic eligible overall and each
original input mode may regress by at most2deg and.02 capture. Report paired
B−A differences separately; A is a new control, not the earlier failed recipe,
because both arms now share corrected reference eligibility and near-plane
exclusion. No extension, preprocessing change or outcome-driven threshold change
within this comparison. A gate pass is only a coarse-stage continuation decision.

Preserve arbitrary-plane synthetic support and raw/exact-black/imperfect-brush
inputs. These near-coronal real upstream affines are weak pairing references,
not expert dense correspondence or uncertainty truth. Previously inspected
development donors are not final-test animals. No public benchmark, calibrated
posterior claim, trajectory confidence claim or native joint qualification.

## Evidence and limitations

The distinction between averaging positive log-losses and taking a log of summed
positive scores follows the multiple-positive analysis in
[SupCon](https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf).
[SimCLR](https://proceedings.mlr.press/v119/chen20j.html) supports treating view
construction as a consequential part of contrastive learning. Neither establishes
that this particular weak-affine histology bridge works. A partial acquired image
may share little visible anatomy with a canonical chart; eligibility based on
atlas support does not measure actual tissue overlap. Retain this limitation and
the full per-row geometry rather than silently selecting only favourable cases.
