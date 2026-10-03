# 063–064: existing pose factors do not recombine into an adequate plane

Both diagnostics used the frozen 059 model and the 246 eligible sections/eight synthetic deformation identities from the 061 panel. They read only synthetic development truth and actual model-produced old-top-eight/new-top-six reflected proposals; no image was opened, no model trained, and no real/public/final-test case was used. All figures weight identities equally. Every truth-based choice below is an **unavailable-at-inference upper bound**.

063 reproduced the 061 parent baseline *exactly*: image-prior-selected visible-tissue CCF error **2.656 mm**, truth-best among 14 **1.022 mm**. Its factor audit was:

| Actual proposal chosen by | Visible-tissue error | Normal angle | Centre error | Effective in-plane frame angle |
|---|---:|---:|---:|---:|
| Image prior | 2.656 mm | 31.91° | 1.861 mm | 45.57° |
| True physical error | 1.022 mm | 13.25° | 0.893 mm | 15.53° |
| Nearest normal | 1.706 mm | 7.99° | — | — |
| Nearest centre | 1.593 mm | — | 0.736 mm | — |
| Nearest in-plane frame | 1.314 mm | — | — | 10.95° |

The best-normal proposal coincided with the best full pose in only 34.2% of sections; best centre did so in 47.4%, best frame in 52.7%. The selected reflection was correct in 59.8% of sections versus 70.9% for the truth-best full pose. Incorrectly reflected selections averaged 2.988 mm error; even correctly reflected selections averaged 2.438 mm, so reflection mistakes contribute but are not the sole bottleneck. Selecting any single factor in isolation remains much worse than selecting a good complete plane.

064 then retained each proposal's physical image-centre point and its *complete* reflection-sensitive in-plane vectors, and truth-scored all 14×14 centre/frame combinations. Its best recombined mean was **0.864 mm**, only **0.157 mm** better than the best original proposal, below the predeclared 0.25-mm factorization gate. The best pair used the same original branch in 27.4% of sections. Recombination improved raw, exact-black and imperfect-brush section-mean oracle errors from 0.977/1.011/1.107 to 0.857/0.844/0.920 mm, respectively, but none approaches the 0.129-mm known-pose mapping control from prior experiments. Simple recombination of available centres and frames therefore cannot make the current beam sufficiently precise.

The scientific decision is to stop spending training on normal-only classifiers, scalar rankers and recombination of the same proposals. The next architecture must generate *better coupled full-plane hypotheses* from image anatomy before fitting/deformation can refine them; its success must be measured by real proposal coverage and selected visible-tissue error, not only training loss or a truth-injected exact-plane score. These synthetic identities share one atlas and are not biological animal-level validation.

Frozen 063: `I:/AnatomyTracker/runs/pose_factors_063_development_audit` (246 rows; raw-row SHA-256 is in its completion receipt). Frozen 064: `I:/AnatomyTracker/runs/pose_recombination_064_development_audit`, rows SHA-256 `781713768907f570e1a5d5496bb72f3f75b8c70147e3d621410b5f85625a3be1`, summary SHA-256 `3567fa0b3595e801908e45e6dfd6639baf21897711daca9ce17ecc3aebba340b`. Each source, parent, panel and output hash checked after exit (six checks each); all 063 selected/best figures match the separate 061 evaluator to numerical precision.
