# 057 result: frame mismatch dominates; PSF does not

The frozen 025 descriptor was measured on all eight combinations of exact-versus-bank frame, exact-versus-67-µm pixel scale, and section-versus-fixed bank PSF. The same 32 GT-valid observed patches and true CCF centres were used on 185 synthetic DEV sections/eight held-out deformation identities. The atlas grid position was **not** snapped in this factorial. The exact/exact/section and bank/67/fixed endpoints reproduced 056 within the prespecified 0.003 cosine tolerance.

| Rigid atlas patch geometry at the true anatomical point | Identity-equal cosine |
| --- | ---: |
| Exact frame, exact pixel lengths, section PSF | 0.7971 |
| Nearest bank frame only; exact lengths and section PSF | 0.5844 |
| Fixed 67 µm/pixel only; exact frame and section PSF | 0.7017 |
| Fixed bank PSF only; exact frame and lengths | 0.7972 |
| Nearest bank frame **and** fixed scale/PSF | 0.5483 |

Relative to the exact rigid baseline, changing only the frame costs **0.2127 cosine**, changing only scale costs **0.0954**, and changing only PSF changes cosine by less than 0.0001. The combined penalty (0.2488) is smaller than the sum of isolated penalties, so the effects interact and must not be added as independent errors. The frame factor also removes the observed in-plane shear; however, the absolute synthetic shear parameter is only 0.013 at its section median and 0.030 at its 90th percentile (about 0.7° and 1.7°), whereas 056 measured mean nearest-bank frame mismatch of 13.7°. This supports, but does not separately prove, orientation discretization as the major component of the frame effect. Raw, black and imperfect-brush inputs all show the exact-to-bank decline.

The result directs engineering away from spending resources on more PSF samples. Finer orientation coverage and scale tolerance are relevant, but **geometry alone is not sufficient**: even an exact-frame, exact-scale rigid true-location key has mean cosine 0.797, below the 056 mean highest-scoring whole-brain lookalike of 0.876. A vastly denser bank could also create more false lookalikes. The next architectural decision must use spatially distributed whole-slice evidence and pose/deformation consistency rather than assuming a better single local descriptor will identify any atlas location unambiguously.

The evaluator exited normally. SHA-256 read-back matched its config/rows/summary; 185 rows match the frozen DEV panel with eight conditions and 32 queries each. No image was visually inspected, no model trained, and no final-test animal, public benchmark, probability calibration or GUI replacement was used. Exact pose and anatomical points are diagnostic oracles, not inference inputs.

Frozen output: `I:/AnatomyTracker/runs/atlas_key_geometry_factors_057_development_eval`.
