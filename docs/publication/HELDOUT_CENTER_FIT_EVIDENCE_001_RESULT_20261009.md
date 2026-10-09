# Centered atlas-fit evidence 001: negative development result

The predeclared no-training diagnostic **failed**. Matching the image descriptor to the already-fitted atlas plane at the exact same spatial location did not select better planes than the frozen 094 score. Do not use this centered cosine as pose feedback or tune its weight on this panel.

The readout used the 237 eligible sections/eight synthetic plans in the **reused 103 v3 DEV panel**. It loaded exact 094 fitted candidate states and mapped physical errors plus the internally initialized 099 step-4,000 descriptor; no weights changed. At each section it evaluated the centered, zero-depth fine match on predicted-tissue cells held out from the 094 coarse fit, requiring atlas support common to all 14 candidates. Forty-eight sections had fewer than 16 such cells and fell back to the old selector. The mean common-site count was 56.5. Saved candidate scores used no synthetic truth; physical error entered only the readout.

| Plan-equal mapped error | Old 094 score | Centered descriptor | Support-only | Physical best of 14 |
| --- | ---: | ---: | ---: | ---: |
| All 237 | **4.156 mm** | 4.185 mm | 4.034 mm | 2.271 mm |
| Raw/no brush, 176 | **4.612 mm** | 4.724 mm | 4.533 mm | 2.742 mm |
| 45–54.7° nearest-axis, 19 | 3.766 mm | **3.490 mm** | 3.635 mm | 2.103 mm |

Across the scored sections, mean within-section Spearman association with **lower** physical error was 0.063 for old score, 0.038 for centered descriptor and 0.107 for support alone. The heavy-oblique subset improved, but is only 19 sections from seven synthetic plans and cannot outweigh the failed all/raw criterion or establish all-angle physical validity. The support-only selector's small numerical gain is a warning about a support shortcut, not a trustworthy anatomical score. The combined predeclared criterion failed: selected gain was −0.028 mm, correlation gain −0.024, and centered correlation did not exceed support-only. Raw and heavy-oblique nonregression conditions passed.

This result closes the obvious scalar-score variants: the older all-bin search could find unrelated anatomy, but simply pinning it to zero displacement loses useful flexibility and still does not reveal coherent anatomical fit. The next consequential intervention should be a **jointly trained, occlusion-aware spatial registration field** with an explicit no-match/no-displacement option, bounded piecewise-smooth deformation, and support-matched wrong-plane examples. It must improve blind candidate selection *and* decoded tissue mapping relative to leaving a good plane unshifted before its loss is permitted to teach the pose head. Keep the direct probabilistic pose head and the same randomly initialized model lineage. This is a design decision, not an asserted successful architecture. No calibration, acquired all-angle truth, GUI promotion or public benchmark is implied.

Runner exited 0. A separate check matched its 237 unique section rows/eight plans, 48 fallbacks, and the row/summary SHA-256 values recorded in `I:/AnatomyTracker/runs/heldout_center_fit_evidence_001/completed.json`; independent plan-equal aggregation reproduced the saved means. Output: `I:/AnatomyTracker/runs/heldout_center_fit_evidence_001`.
