# 113: virtual-oblique pose adaptation result

The 4,000-batch TRAIN continuation and the independently written frozen-checkpoint evaluator completed on `I:`. The run saw 8,000 independently drawn full-angle synthetic sections, 4,000 acquired weak-affine coronal presentations, 2,000 acquired weak-affine sagittal presentations and 2,000 virtual oblique cuts from 55 audited regular TRAIN donor stacks. Eight other regular stacks supplied 32 fixed, newly resliced virtual cuts; the parent 111 model had previously seen some cardinal images from those same animals. This panel is therefore **not animal-independent**. All 113 run/config/draw/training/checkpoint hashes and all evaluator source/row/summary hashes passed; the evaluator compared identical sections at steps 0, 1,000, 2,000, 3,000 and 4,000. No final animals, expert pose truth, calibration or public benchmark were used.

| Donor-equal error, mm | Step 0 | Step 4,000 | Change |
| --- | ---: | ---: | ---: |
| Virtual-cut held-out-donor subset, selected plane | 8.235 | 8.475 | **+0.241** |
| Virtual-cut held-out-donor subset, best of 14 proposals | 5.467 | 5.348 | −0.119 |
| Independent synthetic DEV, selected plane | 3.350 | 3.395 | +0.044 |
| Independent synthetic DEV, best of 14 | 1.710 | 1.604 | −0.106 |
| Acquired coronal DEV, inherited weak affine | 0.999 | 0.760 | −0.239 |
| Acquired sagittal DEV, inherited weak affine | 2.156 | 1.805 | −0.351 |

The desired arbitrary-angle transfer **failed**: 31/32 virtual holdout cuts still exceeded 3 mm selected-plane discrepancy at the final step, versus 30/32 initially. Median virtual TRAIN selected discrepancy over successive 250-cut blocks remained approximately 8.5–9.5 mm, while the best of all 160 proposed branches remained approximately 4.3–4.6 mm. The fixed real cardinal weak-reference gains are useful but do not offset the virtual failure, and the synthetic selected score did not improve. Do not promote any 113 checkpoint as an arbitrary-plane model or infer that additional fitting alone can repair a 5.35 mm best-of-14 proposal gap.

The virtual cuts inherit weak Allen affines and are trilinear reslices of serial acquisitions, **not acquired oblique histology or expert labels**. Their stripe/interpolation texture and the low effective virtual loss weight are plausible explanations for the transfer failure, not established causes. Before scaling this proxy, check whether actual virtual anatomical content carries enough plane information and whether the optimization can fit it without sacrificing independent synthetic/cardinal development sets. The 112 image-dependent atlas-fitting pilot is a separate diagnostic: it can test whether fit quality points back toward a correct nearby plane, but cannot make 113's absent coarse proposals correct by itself.

For scale and scope, [DeepSlice](https://www.nature.com/articles/s41467-023-41645-4) used an Xception-based **coronal** predictor with 131k Allen histology, 443k serial two-photon images and approximately 0.9M synthetic coronal sections. This is not a direct metric comparison: 113 concerns full-angle synthetic and weak-label virtual cuts, with far fewer new presentations. DeepSlice also notes that visually ambiguous single sections benefit from same-block image groups. That reinforces retaining multimodal uncertainty and optional constraints rather than forcing a confident answer for every arbitrary cut.
