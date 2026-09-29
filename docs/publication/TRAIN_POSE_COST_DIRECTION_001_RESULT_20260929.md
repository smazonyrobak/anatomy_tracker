# TRAIN pose-cost direction: strong oracle signal, not a learned-update result

Independent NumPy remeasurement of the completed diagnostic reproduced **128/144
(88.89%)** correct full-map cost directions and **137/144 (95.14%)** with oracle
visible-support weighting. In every agreeing comparison the preferred probe also
reduced the saved physical slab error. There were **no exact ties and no cost
differences with magnitude ≤1e-6**. This is positive directional evidence, not
evidence that the feature metric is uninformative.

The diagnostic used the whole original image-key4000 checkpoint, eight TRAIN
synthetic subjects, the first eligible section per subject and all three paired
modes:24 eligible observations. Each had six one-axis starts (nominal±6° normal
or±300µm offset), probed at±3° or±150µm. Reflection, finite PSF and anatomical
ribbon fields were supplied correctly and held fixed. No learned updater was
used. The144 comparisons are paired/repeated observations, not144 independent
animals, and no censored section is represented in this result.

## Cost magnitudes

Define δ=C(minus probe)−C(plus probe); its sign is the preferred update direction.
The aligned margin is δ times the independently recomputed geometry-preferred
sign, so a positive margin means lower cost toward lower physical error.

| Weighting | Median abs δ | p10–p90 | Minimum abs δ | Mean aligned margin | Median relative abs δ |
|---|---:|---:|---:|---:|---:|
| Full map |0.025027|0.005478–0.059932|0.0002001|+0.025416|7.057%|
| Oracle visible support |0.062810|0.019068–0.179226|0.0001001|+0.086689|12.150%|

Relative magnitude divides by the mean absolute cost of the two probes. Raw
signed δ averages almost cancel because both start signs are present; that
cancellation must not be interpreted as a weak signal. Current mean costs were
0.31980/0.56143 for full/visible weighting.

| Subset (48 comparisons each) | Full correct | Full median abs δ | Visible correct | Visible median abs δ |
|---|---:|---:|---:|---:|
| Normal-u |43/48|0.023027|45/48|0.053935|
| Normal-v |42/48|0.021211|46/48|0.060437|
| Normal offset |43/48|0.030672|46/48|0.080296|
| Raw synthetic background |45/48|0.022975|45/48|0.055935|
| Exact black exterior |47/48|0.037200|48/48|0.089282|
| Imperfect brush |36/48|0.013929|44/48|0.040824|

The12/16 full-map failures in imperfect-brush observations identify a meaningful
mode-dependent limitation; oracle support weighting improves but does not remove
all failures. Oracle visible support is diagnostic ground truth, not an available
inference mask or a requirement for automatic segmentation.

At the supplied truth frame, mean centre residuals were3.27e-11µm(full) and
2.85e-11µm(visible); mean finite-slab residuals were0.007710/0.013482µm, with
maximum per-observation means0.012789/0.023632µm. The truth feature costs remained
0.300819/0.473214: a nonzero cross-modal feature cost is not itself a geometry
error. These geometry values re-average saved distance maps, not a new render.

## Consequence for the next readout investigation

The unweighted directional differences are measurable and of similar order
across all three probed axes. The evidence supports investigating how the
recurrent/readout path uses this information, rather than diagnosing absent
feature direction from the failed learned updates. It does **not** establish
that a particular gain, normalization or pooling change will fix learning.
Per-axis signed negative central differences δ/(2h) are retained separately in
the analysis (cost/radian for normal directions, cost/µm for offset); mixing
those units into one aggregate would be misleading.

This oracle, single-axis, TRAIN-only result does not establish joint pose/shape
optimization, robustness to wrong reflection or PSF, learned convergence,
held-out biological accuracy, calibration or model qualification. No coarse-run
output or operative source was accessed, and no GPU/inference/training was run
for this independent remeasurement.

## Reproduction and pins

- Frozen run: `I:/AnatomyTracker/runs/joint_v6_train_pose_cost_direction_001`.
  Completion SHA256: `556fc13ee2f6dfc1db059c3ce8e5a3b00eca429b2de23c394be19e552cdc5ed5`.
- Flat analysis source:
  `I:/AnatomyTracker/runs/joint_v6_train_pose_cost_direction_001_independent_analysis_source.py`;
  SHA256 `604b47211ad5ea928af55ae3e382658e52be072afc764e085d9b2efde3456a75`.
- Result:
  `I:/AnatomyTracker/runs/joint_v6_train_pose_cost_direction_001_independent_analysis/analysis.json`;
  SHA256 `c208a48baf72d1939c659723144dd5d993dea90619cc34b7f84455d54c352580`.
- Recomputed rows in the same output folder: `recomputed_comparisons.npz`;
  SHA256 `c59460ad2288c7c54d4fa1b2a3adea8279b98c84f71e40e1b4903605af368d0f`.

All completion-listed raw artifacts, the saved failed-native audit and selected
cohort source files passed their recorded hashes. Float64 cost-map averages
matched the original FP32 readouts within2e-7; geometry averages within1e-8µm.
The initial analysis attempt reached JSON serialization, then required a
NumPy-scalar-to-native-bool correction; the completed analysis exited0 without
changing any diagnostic input. Full per-axis/per-mode/per-subject distributions,
truth means and provenance hashes remain in the result, with no row exclusions.
