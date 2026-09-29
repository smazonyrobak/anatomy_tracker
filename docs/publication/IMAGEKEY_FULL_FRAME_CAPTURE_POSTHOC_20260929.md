# Completed image-key001: post-hoc full-frame capture diagnosis

The independent frozen retrieval gate passed; this separate CPU analysis does
not alter that gate or any frozen output. It uses only the saved top128 IDs,
catalogue and original continuous development truth—no model execution,
gallery reconstruction, additional candidate search or benchmark. All 640 rows
are retained; the headline statistics use the original 611 positive-weight
rows and equal organizational-animal group means. These are not biological
animal validation results.

## Result

| Saved candidates | Minimum corner RMS, group-mean / row median / row p90 | Capture <=1 mm | <=2 mm | <=3 mm | <=5 mm |
|---|---|---:|---:|---:|---:|
| Top32 | 2.326 / 1.745 / 3.932 mm | 4.28% | 63.99% | 86.25% | 92.29% |
| Top128 | 1.854 / 1.611 / 2.363 mm | 5.29% | 77.69% | 94.13% | 97.07% |

The original plane capture (angle <=10 degrees **and** sign-aligned offset
<=500 um for one candidate) is reproduced exactly: 68.46%/85.54% at top32/128.
The original 1 mm corner rates also reproduce within 1e-12. Corner geometry
deliberately matches the independent audit's OUV edge-chart corners at s,t in
{0,1}, not the slightly inset 95/96 pixel-centre endpoints. Identity versus
horizontal raster correspondence is minimized; anatomical ML reflection is
never treated as an equivalent anatomy.

For rows with a plane-captured candidate, the oracle minimum-corner choice has:

| Conditional diagnostic | Top32: 418 rows | Top128: 523 rows |
|---|---:|---:|
| Median corner RMS | 1.574 mm | 1.549 mm |
| Median / p90 residual roll | 5.23 / 11.28 degrees | 5.15 / 11.23 degrees |
| Median / p90 in-plane centre shift | 0.583 / 1.048 mm | 0.599 / 1.065 mm |
| Median / p90 absolute span error, either axis | 10.36 / 13.80% | 10.15 / 13.75% |
| Median normal centre shift | 0.181 mm | 0.173 mm |

The catalogue has exactly 12,000 um spans in both directions and zero shear.
Thus adding catalogue normals/candidates alone cannot remove its continuous
scale mismatch. Shear is effectively absent in this original development
subset too (median mismatch below 1e-6 degrees); these data do not establish
learned shear capture.

For the same top128 near-plane oracle choices, correcting the centre alone
would still leave median corner RMS 1.376 mm. The exact decomposition is
`corner_RMS^2 = ||delta_centre||^2 + ||delta_U||^2/4 + ||delta_V||^2/4`.
The edge terms mix normal tilt, roll and scale; they are not separate causal
attributions. Their group-mean squared contributions total 5.987 mm^2 versus
0.549 mm^2 from the centre, with large-tail rows inflating these means.

The separate oracle selector minimizing `(angle/10deg)^2+(offset/500um)^2`
illustrates the danger of selecting on plane geometry alone: at top128 its
median corner RMS is 1.924 mm, with roll p90 79.15 degrees. This selector is
truth-dependent and is **not** an inference rule. Near-plane conditional
statistics have different row sets at top32/128, so their means need not improve
monotonically. Across raw/accurate/imperfect-brush modes, top128 minimum-corner
medians are 1.629/1.569/1.627 mm and 2 mm capture is 78.08/79.83/75.46%.

## Consequence for the next experiment

Inference: retrieval supplies useful plane neighborhoods, but finite-frame
capture still requires explicit roll, in-plane translation and span correction.
The present evidence supports learning those native pose updates before asking
affine-free nonrigid deformation to improve alignment. Enlarging top32 to
top128 materially improves a 2–3 mm starting neighborhood, but barely improves
1 mm capture because it does not add continuous scales/centres.

After the conditional native learner succeeds, an honest capture experiment
should take the actual retrieved candidates, retain their reflection branches,
refine full pose in bounded chunks and select using the model's own image
score. It must not substitute any oracle selector from this diagnosis. Report
selected endpoint point/corner errors and failures on all eligible rows, with
the existing censored/mode/group accounting. The current local truth-near
curriculum is a separate capability test, not proof it covers this retrieval
distribution; especially check its in-plane/scale perturbation range against
these tails. None of this licenses biological calibration or full benchmarking.

## Reproduction and provenance

- Source: `training/diagnose_joint_v6_imagekey_frame_capture.py`; SHA-256
  `76418424055424a1512fbb5f244603918d5a45a213917800f49e23d5f2264ee9`.
- Frozen parent audit SHA-256:
  `f9eb9c6845e048e5fc3ca840a4c5effcf48c1d6ed98afef9daa65a3983e4ca9b`.
- Output: `I:/AnatomyTracker/runs/joint_v6_imagekey_frame_capture_posthoc_001/`.
  `summary.json` binds all three authenticated inputs and defines every
  statistic/selector; `rows.npz` retains all 640x128 candidate measurements,
  selector indices, per-row metrics, original eligibility, groups and modes.
- `rows.npz` SHA-256:
  `bf354f5b7fcb75d515f361c7688e943a252d0c744d39b29ea20ed0506de58a36`.
- CPU command exited 0 in 6.4 s. Roll is measured after minimal-rotation
  transport of the properly antipodal-normal-aligned frame; centre shifts are
  projected onto the truth frame. Physical normal-centre shift and plane-offset
  error are intentionally separate. Quantiles above are pooled row quantiles,
  never mislabeled group-macro quantiles.

No active coherent-subject plan output was inspected; no model source or frozen
retrieval artifact was modified.
