# 183 TRAIN DAPI specimens: gross alignment only

The frozen CPU run `I:/AnatomyTracker/reports/dapi_train_multispecimen_affine_qc_183_001` completed normally. It reused the prior BM03 registration and fitted separate same-stain DAPI-average affines for TRAIN BM04, BM06, and BM10. The three selected refinements improved correlation and whole-brain overlap versus the BM03-seeded transform. No model was trained and **none of these transforms is approved as a local CCF or electrode-site label**.

| TRAIN specimen | Seed tissue Dice | Selected tissue Dice | Seed eroded-tissue correlation | Selected correlation |
| --- | ---: | ---: | ---: | ---: |
| BM04 | 0.91 | 0.947 | 0.21 | 0.548 |
| BM06 | 0.95 | 0.972 | 0.55 | 0.740 |
| BM10 | 0.96 | 0.974 | 0.53 | 0.721 |

The three-plane overlays show improved gross agreement, but cortex, ventricles, and internal structures still have visible green/magenta offsets; BM04 is notably weaker in internal intensity correspondence. These displays and same-grid fit metrics cannot establish small-nucleus correspondence, left/right or AP sign, or a true native-to-CCF mapping. Individual NIfTI world origins and units are not established by their identity sforms; the script explicitly assumes the header spacing in millimetres, native origin zero, and positive voxel axes. The volumes reconstruct physical 100-µm coronal sections. Virtual oblique reslices from them are training views, **not acquired oblique-section validation**. The DAPI population average includes all 12 source animals, so its use also prevents calling BM17/BM18 a fully independent reference holdout.

The script compared each accessed TRAIN source hash against the frozen source manifest. After exit, the 12 referenced reports/transforms/overlays matched the completion receipt, as did the committed script, protocol, and source manifest. `complete.json` SHA-256 is `5a7cccfaaf5a877bcd3d117748859aedf8ead0c09da329cc77701a2aa8124aa2`. Only BM03/BM04/BM06/BM10 individual images were accessed; DEV BM15/BM16 and sealed BM17/BM18 individual images were not opened. No CCF annotations or label volumes were written.

Next, this cohort can inform **appearance-only or explicitly weak-coordinate** development, but not site-level mapping supervision or calibration until independent specimen/reference landmarks and native-axis conventions are validated. A blinded competent anatomical review of cortex, ventricles, hippocampus, and deep nuclei across the volume is required to admit specific specimens/regions as spatial labels. Separately acquired, animal-disjoint steep-oblique sections remain necessary to assess the arbitrary-plane claim. The [CC0 DAPI source](https://doi.org/10.12751/g-node.16wrxa) does not override the separate [Allen Content terms](https://alleninstitute.org/legal/terms-of-use) for atlas-derived labels or any later redistribution.
