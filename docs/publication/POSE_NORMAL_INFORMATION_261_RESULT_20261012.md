# 261: direct coordinate model's normal is almost constant

The predeclared read-only 261 assay extracted the fitted normal from the frozen 260 batch-3,000 checkpoint on all 227 eligible frozen v7 synthetic DEV sections (eight independent deformation plans). It used no regenerated images, training, acquired labels or public benchmark. Axes are antipodal, so the diagnostic uses outer products and absolute cosine.

| Measure | Predicted normals | True normals |
| --- | ---: | ---: |
| Eigenvalues of mean axis outer product, ascending | 0.0034, 0.0088, **0.9878** | 0.3093, 0.3186, 0.3720 |

Nearly all predicted axes point along one direction, whereas the true axes span the sphere. Image-matched equal-plan mean absolute cosine was **0.5434**; a deterministic 1,024-draw within-plan pairing permutation averaged **0.5357**, an advantage of only **0.0077**. This fails the prespecified ≥0.05 narrow threshold for useful image-specific normal information. The equal-plan normal-axis error reproduces the 260 readout at 54.16°. The physical span improved in 260, but its normal remained effectively a population-average direction, not a reliable orientation estimate.

This does **not** mean arbitrary cuts are unidentifiable from the synthetic images. The 257 read-only oracle-atlas comparison separated nearby wrong tilts in 97.84% of eligible support-matched comparisons when given the correct pose/reflection. It means this direct dense coordinate regressor, objective and exposure did not convert image anatomy into a global full-sphere direction. More unchanged training is unjustified. The next experiment should explicitly preserve competing global plane hypotheses and test whether atlas-image evidence can rank/correct *imperfect, blindly generated* hypotheses, before building a larger joint refinement/deformation loop. Low-information slivers must remain eligible for uncertainty/abstention rather than forced single-plane certainty.

Independent read-back checked 227 unique case IDs, all eight plans, output/protocol/source hashes, outer-product eigenvalues, matched score and exact casewise agreement with frozen 260 axis errors. Frozen output: `I:/AnatomyTracker/runs/pose_normal_information_261_dev_001`. No model or GUI was promoted, and no calibrated probability or all-angle biological accuracy is claimed.
