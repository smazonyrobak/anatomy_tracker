# Proposal-head precision failure and focused correction check

2026-09-28. These are numerical diagnostics on completed synthetic development
runs, not model-quality, calibration or benchmark claims. Original checkpoints,
predictions and metrics remain unchanged historical results; they must not be
silently relabelled as corrected evaluations.

The initial CPU diagnostic inspected completed `joint_v6_proposal_substantive_001`
step 10000 on 16 evenly spaced development rows. The eight mixture components
had eight distinct peak cells per image, mean pairwise Jensen-Shannon divergence
0.564 nats, mean total-variation distance 0.855 and effective gate count 4.71.
Thus small query initialization had not left identical components. Context was
also nonconstant (mean coordinate variance 37.42, effective rank 5.39).

After `joint_v6_proposal_capacity_002` exited successfully, its steps 3000 and
10000 were inspected on the same fixed indices:
`[0,43,85,128,170,213,256,298,341,383,426,469,511,554,596,639]` of 640 rows.
Only proposal inference was used, with no atlas rendering. TF32 was disabled.

| Comparison against full FP32 | Log-probability RMS error | Top-cell agreement | Subset NLL difference |
|---|---:|---:|---:|
| Original AMP, step 3000 | 0.18159 | 6/16 | +0.03327 |
| Original AMP, step 10000 | 3.14676 | 0/16 | +1.70878 |
| AMP encoder + FP32 head, step 10000 | 0.00151 | 16/16 | +0.0000082 |
| Patched AMP versus patched FP32, step 10000 | 0.00140 | 16/16 | +0.0001112 |

At step 10000, normal/offset score terms had cell-constant RMS 3151.7/3342.0,
while their across-cell standard deviations were only 4.60/11.06. Half precision
lost meaningful contrasts. The components still had eight distinct peaks and
mean pairwise JS 0.652 nats in FP32; context variance had grown to 1663.6.

The correction computes geometry, context, queries and scores outside encoder
autocast and subtracts each geometry embedding's cell mean before dot products.
Subtracting this per-component, cell-constant score preserves the exact
categorical density, including cell prior mass. In finite FP32, original versus
centered step-10000 log probabilities differed by up to 0.00275 (RMS 0.000401),
so a proposed maximum-log-error tolerance of 1e-4 was **not** met; maximum
probability difference was 8.71e-6, with all top cells unchanged.

The actual patched head, tested on the same unchanged checkpoint, returned
float32 scores under AMP. Its AMP/FP32 posterior JS was 8.00e-8 and NLLs were
9.9730189/9.9729077. One backward pass produced 34 gradient-bearing parameter
tensors, all finite (global L2 77.56). No optimizer was constructed; all model
parameters and buffers remained exactly equal to the loaded checkpoint.
This confirms numerical repair on the inspected batch, not that it alone
explains every training failure or establishes generalization. Fresh stable
optimization and internal animal-level evaluation remain necessary.

## Exact audit references

Diagnostic paths are under `I:/AnatomyTracker/tmp/`; capacity checkpoint paths
are under `I:/AnatomyTracker/runs/joint_v6_proposal_capacity_002/`. SHA-256:

| Artifact | SHA-256 |
|---|---|
| `proposal_mixture_diagnostic_20260928.py` | `a4cf48b69b367061f914f6e752dbbf0443ea96387668a3e68693c5949f00a242` |
| `proposal_mixture_diagnostic_20260928.json` | `47795ed7f258134b94739d076f95be5bbbe9d7d197d40a970d3e93a42cb13ceb` |
| `proposal_capacity_precision_20260928.py` | `1e8652c46b2de5972487c9eae32c1da6e266afbe7aa88cc3f3dcb7cf45151275` |
| `proposal_capacity_precision_20260928.json` | `a3bd9d9e08ad737f645bbc030c1c2cb90a5f632df8a1bab434b83592b46542b2` |
| `proposal_precision_patch_check_20260928.py` | `6158cfc42321da136c12c36cf12918f49478775b408dc56394d88bc8dd5c3451` |
| `proposal_precision_patch_check_20260928.json` | `04e2e8f2719aa34d1c388f8c5a712fbebe5618c8d2813c9ed3ac68a77e0bb1c2` |
| `joint_model_step_03000.pt` | `2447989285e9ff386b37a43fd5dc7dd7ce8ca13fba925a3d0c933a9289cc2322` |
| `joint_model_step_10000.pt` | `12453700f57d840432828e8ee1e1828fc1d7958ac5ff002daa92d9579779f362` |

Original `training/arbitrary_plane_coarse_proposal_v6.py` SHA-256:
`0becbb0df56b9b3470dc1489502fe7ecc3e2294ee70c669e2099f7635a72b5d0`.
Patched source tested here:
`676bc3ea8704b534aefef4e2ead7ff6e51d87cb5e61157ef0ac4a637205a7cd3`.
