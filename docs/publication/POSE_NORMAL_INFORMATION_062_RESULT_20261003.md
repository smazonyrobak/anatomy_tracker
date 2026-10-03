# 062 frozen-encoder normal probe: no useful gain

**Decision: reject the simple normal readout; do not continue it into the joint model.** It learned from fresh synthetic sections, but remained worse than the frozen 059 parent on the identity-disjoint synthetic development panel at every checkpoint. This diagnostic did not change the deployed research reference, local mapper, or atlas matcher.

The 64-anchor antipodal readout trained for 5,000 batches on 20,000 accepted, independently drawn arbitrary-plane synthetic TRAIN sections. Its 22,097 draw records have 20,000 unique accepted physical-section IDs, all in TRAIN, and 6,620 raw / 6,786 exact-black / 6,594 imperfect-brush accepted images. These are not paired backgrounds for a repeated physical section. The 059 encoder and all its pose/mapping parameters were frozen; the new head alone was trained. Train block-mean cross-entropy fell from 4.039 in batches 1–1,000 to 3.294 in batches 4,001–5,000, and train normal angle fell from 49.2° to 37.0°. A falling training loss is not an accuracy result.

The frozen evaluation used 246 eligible sections from eight independent synthetic deformation identities on the existing 061 panel. Values are equal means across identities; normal angles identify `n` with `−n` and are not full-plane tissue-coordinate error.

| Checkpoint | Parent selected normal | Probe selected normal | Parent top-eight normal | Probe top-eight normal | Parent / probe top-eight within 15° |
|---:|---:|---:|---:|---:|---:|
| 0 | 31.56° | 56.70° | 9.13° | 21.43° | 85.9% / 30.9% |
| 1,000 | 31.56° | 43.04° | 9.13° | 16.46° | 85.9% / 54.9% |
| 3,000 | 31.56° | 39.99° | 9.13° | 16.24° | 85.9% / 58.7% |
| 5,000 | 31.56° | **38.15°** | 9.13° | **12.79°** | 85.9% / **71.2%** |

The predeclared advancement gate required at least a 10° improvement in selected angle and a 15-point increase in top-eight-within-15° coverage without appearance collapse. The final readout instead worsened selected angle by 6.59° and top-eight coverage by 14.7 points. At 5,000, its selected normal angle was 39.97° raw, 36.07° exact-black, and 39.28° imperfect-brush, worse than the corresponding parent 30.65°/31.58°/32.21°. The parent already places a normal within 15° in its top eight for 85.9% of sections, yet 061's truth-best full pose is around 1 mm and chosen pose around 2.6 mm. This focuses the next diagnostic on *coupling orientation to position, roll, reflection, and candidate quality*, not just learning another normal score.

The probe's continued learning trend does not prove an optimized head could never catch up. It shows that this inexpensive frozen-feature/64-anchor intervention is not a useful advance over the current parent at its fixed 20,000-section decision point. Synthetic identities share one atlas and are not biological animals; no real expert oblique truth, final-test animal, public benchmark, or probability calibration was used.

The completed run is `I:/AnatomyTracker/runs/pose_normal_information_062_pilot`: draw SHA-256 `b1faa83f34385cd4ebcc8d7f155bceee3dc4fe9f4ef04fe3c6bf765aec1b81b6`, train-log SHA-256 `72d6f13c7c162129510e9d7ade0e721bb3e6d7d4267ff74645a61d0fdbab0240`, config SHA-256 `4ce53a8c9b303c476a2af5a8483b2ef026495a344c0f50146a85fb8dce5ef89b`. The evaluation is `I:/AnatomyTracker/runs/pose_normal_information_062_development_eval`: 984 raw rows, rows SHA-256 `606d0e0aa2788278745aa7f3434a4d5ef61f36faede1c7c851f129023fc2836b`, summary SHA-256 `f4fc9c2ad6d0bc105fbcf4be4d0a2f70f49afd6abb627488c6f9f95ed9479c52`. The 12 post-exit source, parent, checkpoint and file-hash checks passed, and the primary identity-equal numbers were independently recomputed from raw rows.
