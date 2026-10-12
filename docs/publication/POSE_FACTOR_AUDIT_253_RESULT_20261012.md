# 253: frozen 247 image-only pose-factor audit

The read-only 253 audit reran only the frozen 247 image-only predictor on all 227 eligible sections from eight independent synthetic 248 DEV plans. Every section ID, source-file hash, selected mode/reflection and normal-axis error matched the frozen 248 readout. No atlas fitting, training, CAL/FINAL data, or public benchmark was used.

| Metric | Case mean | Case median | Plan-equal mean |
|---|---:|---:|---:|
| True major in-plane FOV | 16.454 mm | 15.916 mm | 16.447 mm |
| Predicted major FOV | 10.950 mm | 10.861 mm | 10.950 mm |
| True minor in-plane FOV | 13.883 mm | 13.243 mm | 13.882 mm |
| Predicted minor FOV | 1.676 mm | 0.985 mm | 1.675 mm |
| Absolute major/minor FOV error | 5.513 / 12.207 mm | 5.410 / 11.747 mm | 5.505 / 12.207 mm |
| Centre Euclidean error | 2.792 mm | 2.655 mm | 2.783 mm |
| Frozen image-only axis / rigid error | 54.66° / 4.247 mm | 56.90° / 4.121 mm | 54.66° / 4.247 mm |

True minor FOV was never ≤1.2 mm (0/227; minimum 8.571 mm). Predicted minor FOV was ≤1.2 mm in **161/227 (70.9%)**. Of the 11 cases with image-only axis error ≤15°, **all 11** still had ≥2-mm rigid tissue-coordinate error (mean 3.593 mm, minimum 2.700 mm); their mean predicted versus true minor FOV was 1.625 versus 13.295 mm. Thus normal-axis capture alone cannot explain or repair the current selected-coordinate failure. The predicted minor median of 0.985 mm is consistent with the 12-mm span head's `exp(-2.5)` lower log-span clamp; this is evidence of a severe learned span collapse, not proof of the training mechanism that caused it.

Output: `I:/AnatomyTracker/runs/pose_factor_audit_253_001`. Its completion record binds script SHA-256 `dbe0bcf95399385cd8036f87c6212706e3b3b0eaddf15e29e98d808d8571a6c1`, checkpoint `2e32064df84b961a455089af28db5de1bfbf674ddd8a218e0012362029faf088`, panel completion `55c4d8c5080cfc1bf8e191729ef02f9a9dfe634211f986de75f4f90aa80a4e4a`, and prior 248 readout completion `abd1bf3c9cafa5de0c60387119c8d7a5dbf608da37d6ad0eb4992be1ec3765d1`. Row and summary hashes are in `completed.json`.
