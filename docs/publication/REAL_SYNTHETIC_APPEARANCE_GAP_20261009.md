# TRAIN image-appearance audit, 2026-10-09

This is a TRAIN-only data audit, not a model benchmark or a measurement of laboratory artifact prevalence. The frozen donor-equal real census sampled one image from each of 512 distinct donors in the full reserved TRAIN pool (263,754 sections from 1,885 donors). The frozen synthetic census used 256 independently sampled eligible physical planes from the v2 artifact generator (288 attempts, 256 distinct physical-section IDs). Its geometry covers the sphere of plane normals; its 64 deformation bases are synthetic, not 64 independent biological brains. Both reports measured the entire image canvas, not tissue-only pixels. Real images were stored at 192² and the synthetic census rendered at 256², so exact-zero pixel fractions are approximate cross-domain comparisons; the outer-border result is more direct.

| Whole-canvas metric | Real TRAIN, 512 donors | v2 synthetic, 256 planes | v2 raw only, 100 planes |
| --- | ---: | ---: | ---: |
| Mean exact-zero fraction | 0.695 | 0.571 | 0.302 |
| Mean fraction below 0.03 | 0.790 | 0.650 | 0.497 |
| Mean outer-border fraction above 0.03 | 0.000 | 0.141 | 0.343 |
| Brush-outline available | 0 | 156/256 | 0/100 |

The discrepancy is structural: the v7 source sampler chooses raw, exact-black brush, and imperfect-brush modes equally. The v2 raw branch chooses an exact-black exterior only 40% of the time; thus an exact-black **without-brush** image is only about 13% of independent synthetic draws before eligibility filtering. All 512 inspected real TRAIN images had dark borders and no brush channel. The v2 stress stream also intentionally makes fragment, fold, bubble, and shifted seam events frequent (129, 138, 104, and 154 cases out of 256 respectively); these are chosen stress probabilities, not measured real prevalence. The inherited checkpoint was trained on the older v1 stream, **not** the v2 stress stream; this audit does not prove that v2 caused any current checkpoint error. The older stream likewise chooses modes equally and does not explicitly provide a dominant raw/exact-black combination.

Decision: retain arbitrary-plane geometry and all artifact types, but make a new versioned TRAIN input sampler with a majority of independent raw/no-brush, exact-black-exterior cases, plus non-black raw and optional brush cases. Reduce the deliberately stress-heavy artifact-event probabilities for routine training while retaining the frozen v2 stress panel as an out-of-distribution check. Do not pair or reuse the same physical plane to manufacture background variants. The donor-equal Allen census does not define the deployment laboratory's background/artifact distribution; a representative, consented physical-slide sample with specimen-level provenance is still needed before setting a final acquisition model. No large training or claim of calibrated probability follows from this audit.

Frozen receipts:

- Real: `I:/AnatomyTracker/runs/reserved_real_train_appearance_census_001/summary.json`; 512 distinct TRAIN donors; rows SHA-256 `356b143640ac19110f805906f51b75b5d336aee4ec1b4f5150b945aa42587ec8`.
- Synthetic: `I:/AnatomyTracker/runs/slide_v2_train_appearance_census_001/summary.json`; 256 distinct physical IDs; all three frozen output hashes independently matched `completed.json`: rows `3df0e4fb393baaa0b834680fc30e15bafd8a8e101f770579a76b2682c5051dfa`, attempts `71eedea3198304d354e3d38aab1059f985d8996941681c9457cfaef43fbf83ba`, summary `a88acc763f5682b2ae037296b9550ce4a7486e96958ba8a4c0a28005c261b75e`.

Scientific limitation: these global pixel statistics cannot assess whether realistic anatomy, tear geometry, folded tissue, contrast, PSF, local tissue displacement, or ambiguous sections are represented adequately. The 64-case v2 stress assay separately showed no *relative* heavy-oblique collapse, but its selected fitted mapped error remained 2.423 mm and best-of-eight oracle error 1.507 mm. A useful arbitrary-plane *synthetic* response is not physical-angle generalization.
