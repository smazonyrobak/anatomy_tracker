# 057: isolate the atlas-bank frame, scale and PSF penalties

The 056 matched geometry ladder found a large 0.249 mean cosine loss from exact rigid geometry at a true anatomical point to the bank's nearest frame/fixed-scale/fixed-PSF geometry at the *same* point. This combined comparison cannot justify spending time on a denser orientation bank versus multiple pixel scales or thickness settings. 057 separates those three render changes while holding the observed query, anatomical centre, frozen 025 weights and selected nearest bank frame fixed.

For the same 185 synthetic DEV sections/eight deformation identities and 32 GT-valid query pixels, render all eight combinations of:

- exact observed in-plane frame (including shear) versus nearest stored bank frame;
- exact observed physical pixel lengths versus 67 µm/pixel;
- section-specific finite-thickness PSF versus the bank's fixed ±50 µm PSF.

All atlas patches are centred at the same true synthetic CCF coordinate; no bank-grid snapping occurs in this experiment. Compute identity-equal query/atlas descriptor cosine for each combination and raw/black/imperfect-brush strata. Verify that the exact/exact/section and bank/67/fixed endpoints reproduce 056 within 0.003 cosine. Compare one-factor changes from the exact baseline and leave interactions visible rather than attributing the full combined loss to one factor. The factor labels refer to patch-render geometry; an exact pose and CCF point are oracle inputs, not available in inference. No images are to be viewed, and no training, public benchmark, final-test animal or calibration is involved.
