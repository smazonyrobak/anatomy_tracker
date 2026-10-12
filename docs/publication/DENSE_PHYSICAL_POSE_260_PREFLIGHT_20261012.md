# 260 preflight: tissue-weighted fit is close to the pre-artifact plane

The 260 trainer fits a plane to its predicted, validity-weighted dense CCF coordinates, but supervises that fit with the generator's pre-artifact full-canvas affine plane. These two gauges could conflict after tissue deformation and partial visibility. Before launching training, `training/assay_dense_pose_gauge_260.py` measured the discrepancy on the exact 227 eligible sections of the frozen 248 synthetic DEV panel. This is an **oracle-target consistency assay**, not a model prediction or a real-slide validation.

The assay pooled the saved true observed pixel-to-CCF coordinates and validity to the trainer's 96×96 grid, fitted the visible valid coordinates by least squares, and compared the fitted plane to the saved pre-artifact target state after the exact horizontal-reflection/raster-centre conversion. It verified each panel file hash and reported means with eight synthetic deformation plans weighted equally.

| Comparison | Equal-plan mean | Case median |
| --- | ---: | ---: |
| Centre discrepancy | 0.145 mm | 0.087 mm |
| Unoriented normal-axis discrepancy | 0.51° | 0.34° |
| U/V edge-vector discrepancy | 0.639/0.662 mm | 0.440/0.466 mm |
| Fitted/true minor physical FOV | 13.70/13.88 mm | 13.11/13.24 mm |
| Pre-artifact/best-fit error on valid CCF pixels | 0.214/0.133 mm | 0.179/0.122 mm |

Only 2/227 centre discrepancies exceeded 1 mm, and no axis discrepancy exceeded 20°. This is small compared with 258's 3.135 mm fitted and 3.193 mm dense coordinate errors and 54.22° axis error, so the gauge mismatch does not invalidate the planned supervision on this panel. It is not zero, especially in some edge vectors, and could matter once the model is substantially better; remeasure on acquired tissue before calibration.

The first scratch assay output `dense_physical_pose_260_gauge_preflight_001` had a reporting-only offset error in its pre-artifact pixel residual calculation: it used image `x,y` where its stored centre required `x−0.5,y−0.5`. The corrected, authoritative result is `I:/AnatomyTracker/runs/dense_physical_pose_260_gauge_preflight_002`. The other gauge metrics were unchanged. The scratch output was retained and must not be cited as the corrected residual result. No weights, public benchmark, CAL/FINAL animal, or GUI path was touched.
