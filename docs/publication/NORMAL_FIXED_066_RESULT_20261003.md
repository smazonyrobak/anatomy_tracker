# 066: normal-fixed in-plane feasibility result

The frozen 066 oracle passed its predeclared **geometric headroom** threshold, but it is not a blind model result. On 246 eligible synthetic sections from eight synthetic identities, the prior 059 beam's truth-best full-frame error was **1,021.5 µm**. Holding each candidate normal fixed while using unavailable tissue-to-CCF truth to fit an unconstrained in-plane affine frame reduced the truth-best-of-14 error to **255.8 µm** (threshold: ≤750 µm). The rigorous normal-only mean-displacement lower bound was **221.2 µm**. An affine fit permits shear and is more flexible than a rigid or similarity registration.

| Identity-equal mean | Before fit | Normal-fixed truth fit |
| --- | ---: | ---: |
| Prior-selected candidate | 2,655.9 µm | 798.6 µm |
| Truth-best of 14 candidates | 1,021.5 µm | 255.8 µm |

The truth-best fitted error by input appearance was 259.6 µm on exact-black exteriors (101 sections), 268.2 µm on imperfect-brush inputs (67), and 244.3 µm on raw backgrounds (78). These are synthetic appearances, not independent biological animals. The prior-selected and best-of-14 unfitted errors reproduce the frozen 063 rows section by section.

This isolates the main missing capability: the existing normal proposals can support substantially better localization **if** the image can identify an appropriate in-plane frame. The oracle does not show that an atlas-to-histology image score can find that frame, nor that the chosen candidate can be ranked without truth. It therefore authorizes a small blind atlas-driven search experiment, not model promotion, uncertainty calibration, GUI deployment, or a public benchmark. The next experiment should test the actual blind selected-plane error with input-appearance and weak-real-donor regressions as guardrails.

Frozen run: `I:\AnatomyTracker\runs\normal_fixed_066_development_audit`. Source: `training/diagnose_normal_fixed_066.py`. The independent post-run audit matched SHA-256 of `config.json` (`c825f9bc7089478d5ab7188d39a6236bdbe67985ea60916d4fc07d2a4b5401f0`), `rows.jsonl` (`addb0e50278da3174550682c20abbf1c4342f397bf57039f7804ed149fe99424`), and `summary.json` (`3d50024ac4d801c0a359be8ff48f00a3f9bde2826c403dd1fd8a561ec2a61819`) against `completed.json`, and recomputed identity-equal means from the raw 246 rows. The frozen parent, panel, reference rows, and source hashes are recorded in `config.json`. No images were opened, no training occurred, and no public or final-test data were used.
