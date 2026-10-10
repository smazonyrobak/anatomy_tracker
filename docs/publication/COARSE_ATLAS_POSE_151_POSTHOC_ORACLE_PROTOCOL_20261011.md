# 151 posthoc correspondence oracle on the frozen 150 synthetic DEV panel

This is a **diagnostic, not a new qualification gate**. It asks why the scratch-trained 150 matcher failed its prespecified fresh-plan gate. No checkpoint, architecture, threshold, example, or subgroup is selected using this panel. It does not train anything. In particular, an oracle fit is not an implementable inference result.

The sole inputs are the frozen 150 TRAIN completion (`07f9852ddc7176dcb2e77db8510be671fe8ee04e6919c87397d57fa5785031fe`), 150 fresh synthetic DEV panel completion (`ffdd678cad121ff86b0e88ea582dafb8cb6497da86e5803a78c89fbef0656031`), and 150 evaluation completion (`02e83c4c0910d933021a1045659a865d29e1f1f39a11e85d869341a6e1050224`). The 150 full-intensity checkpoint is the already-selected step 6,000. The frozen 132 step-2,000 pose checkpoint supplies the same blind 16-candidate beam as the 150 evaluator. Only the panel's 243 eligible sections from eight synthetic development deformation plans are processed. The exact true plane and the truth-best beam branch are analyzed separately; the latter is selected using ground truth and is never claimed as blind inference.

At each model 24×24 source query, sample the frozen native-256 valid mask and exact observed-to-CCF map bilinearly. A query is oracle-valid only if sampled validity exceeds 0.99. Use the same candidate-local 3 mm reachability limit, supported 24×24×9 atlas key lattice, finite-thickness atlas renderer, learned fit, and ridge as 150. A true key means the **nearest supported and candidate-reachable key to the exact observed CCF target**; queries without such a key receive zero key-oracle weight. Record the reachable fraction and nearest-key distance among oracle-valid queries, so discrete-key limitations are not hidden.

For both roles, fit the same rigid plane from exactly five correspondence/weight combinations:

1. Learned correspondence + 150 effective learned fit weights (must replay the frozen 150 evaluation).
2. True nearest supported key + the same learned weights, zeroed where the key is unreachable.
3. True nearest supported key + oracle-valid query weights, zeroed where unreachable.
4. Continuous exact observed-to-CCF target + oracle-valid query weights.
5. True rigid plane CCF target + oracle-valid query weights.

Report mean Euclidean rigid CCF error over all valid native-256 pixels, not the 24×24 fitting queries. Store one row per eligible physical section and role with animal/specimen/experiment and plan receipt IDs, five errors, initial error, query counts, key reachability and distance. Summarize section-equal and synthetic-plan-equal errors and reachability for all cases and the initially-near ≤1.5 mm subset, crossed with input mode (raw, exact black, imperfect brush) and nearest-cardinal angle bin (<15°, 15–30°, 30–45°, ≥45°). Empty strata remain explicit. Write `config.json`, `rows.jsonl`, `summary.json`, and hash-bound `completed.json` only under `I:/AnatomyTracker/runs/coarse_atlas_pose_151_posthoc_oracle` after source and this protocol are reviewed and frozen.

This panel contains stylized atlas-derived synthetic slides, not expert-labelled biological all-angle sections. These comparisons cannot validate calibrated uncertainty, electrode-region probabilities, local nonrigid alignment, the GUI, or a DeepSlice benchmark. Any architecture change inferred here requires new TRAIN-only development and a different held-out test; do not retune 150 on this DEV panel.
