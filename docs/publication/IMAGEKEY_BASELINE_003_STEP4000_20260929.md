# Frozen original003 step4,000 physical retrieval baseline

CPU report exited0 using completed `joint_v6_proposal_curriculum_003`, not the
live similarly named joint-rehearsal run. All4,000 attempted updates were
applied. No model inference, GPU execution or external benchmark was used.
The independently reconstructed existing normal/offset/topK metrics agree with
the original saved readout; maximum NLL macro difference is1.35e-8 nat.
All640x98,304 log probabilities are finite; maximum log-normalization error
is6.965e-7. The prepared catalogue/development hashes match the original receipts.

## Frozen metric definitions

Order all catalogue cells by descending model score, breaking ties by lower
catalogue index. Derive physical OUV and normals independently in NumPy float64
from stored12D states. Continuous prepared truth is not rounded to its cell.

- Normal angle: `atan2(||n_pred cross n_true||, |n_pred dot n_true|)` in degrees.
- Offset: `|d_pred - sign(n_pred dot n_true) d_true|`, where
  `d = (centre - support_origin) dot n` in um; zero dot uses sign+1.
- `physical_plane_capture_at_32` / `_128`: at least one score-selected topK
  candidate **simultaneously** has normal angle<=10 degrees and offset<=500um.
  Separate best-angle and best-offset candidates cannot satisfy the criterion.
- Optional `frame_corner_capture_at_32` / `_128`: at least one topK candidate
  has four-corner RMS Euclidean error<=1000um after choosing the lesser of
  identity and horizontal corner reversal `[1,0,3,2]`. Corners are continuous
  `O,O+U,O+V,O+U+V`, not last pixel-centres. This separately measures finite-frame
  agreement and does not additionally require the plane criterion. It quotients
  horizontal frame representation, not anatomical ML reflection.

Use original `weight>0` eligibility. Average selected rows within each
organizational synthetic group, then give contributing groups equal weight.
Row capture counts are also saved; they are not the group-macro denominator.
These thresholds are engineering development definitions, not calibration,
biological-animal inference or a claim of usable registration.

| Stratum | Rows / groups | MAP normal, deg | MAP offset, um | Plane capture top32 | Plane capture top128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eligible overall | 611 / 40 | 47.497952 | 2864.663375 | 10.593636% | 21.623626% |
| Eligible raw / brush absent | 207 / 40 | 52.132647 | 3055.494373 | 6.708333% | 17.416667% |
| Eligible black / accurate brush | 204 / 40 | 44.431261 | 2859.949102 | 15.166667% | 28.541667% |
| Eligible imperfect brush | 200 / 40 | 46.030372 | 2608.590029 | 9.750000% | 18.916667% |
| All, descriptive | 640 / 40 | 47.833944 | 2926.579722 | 10.937500% | 22.656250% |
| Censored, descriptive only | 29 / 17 | 54.935057 | 4208.239418 | 17.647059% | 41.176471% |

Eligible plane-capture row counts are65/611 and133/611 for top32/128.
Optional eligible four-corner capture is0% and0.46875% (0 and3 rows); the
top128 cases comprise one raw and two accurate-brush rows. Plane-only capture
therefore must not be presented as finite-frame or deformation accuracy.
Eligible full-cell NLL is9.653926585; exact-cell top32/128 recall is1.296474%
and5.902015%. None of these metrics is a globally adequate model result.

## Artifacts and exact receipts

Flat reproducible script: `I:/AnatomyTracker/tmp/imagekey_baseline_003_step4000.py`.
Results directory: `I:/AnatomyTracker/runs/joint_v6_imagekey_baseline_003_step4000`.
`baseline.json` contains definitions, hashes, all/censored/mode intersections,
macro results and raw counts. `rows.npz` contains `label`, `top_indices`,
`weight`, `group`, `mode`, and each metric above, plus `nll`, `hit_at_32`,
`hit_at_128`, `plane_angle_deg`, `normal_offset_error_um`, `frame_corner_rms_um`.
Mode keys retain original `smart-brush-absent`, `smart-brush-accurate`, and
`smart-brush-imperfect`; append `/support_eligible` for the comparison strata.

- Baseline JSON: `5672303902e0b3d108507e40a883b848418c656c20e1bdc48754469009961eeb`.
- Recomputed row NPZ: `adbf20d8b46b02a7ad420a48f054008f5955de2e5699d5f63086f3f3940ddf47`.
- Flat report source: `75b41c04d2dcee1765686a85784c2b5271a759339993da517c44b2ac8e52877e`.
- Original003 step4,000 raw probabilities: `351dc6955f3d8142ad27cdd71d0c3ecf17b1738d5d182eb7bb88efced1bb153a`.
- Prepared catalogue: `9b49d203cc73ce3a66e648bbe5228231eb5cc9c17d5db4669eefe0f08ae22c71`.
- Prepared development: `89e078582aaa365299d9823deb2a005308f360ab8909db7125a4534556d622c4`.
- Original003 driver snapshot: `550bb4660fbfbc10752f23442c02c78e3bb6f0282667a24a597ff572e4461705`.
- Original003 config: `43f78e82b3d550cabd3f3a974d47d5d5256d9a9126a309cedb15b223eec70b07`.

The JSON additionally hashes the original trace and saved metric summary.
Comparison with a new retrieval pilot must retain these exact definitions,
eligibility, group weighting and original precision provenance; it must not
replace the baseline with a more convenient endpoint or censored-row subset.
