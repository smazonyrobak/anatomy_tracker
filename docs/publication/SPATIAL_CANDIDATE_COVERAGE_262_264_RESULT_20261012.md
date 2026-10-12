# 262–264: geometric pose-bank reachability, not blind localization

Three short, preregistered **read-only** assays used the exact frozen 260
batch-3,000 model and the same 227 eligible physical sections from panel 248.
No new image, learned weight or synthetic training presentation was created.
Each assay asked whether an inference-available candidate grid had some plane
within all of the proposed local updater's generous bounds: unsigned normal
≤15°, centre ≤3 mm and both *unordered* physical span log-ratios ≤0.7.

| Candidate FOV from image/atlas | Correctable cases | <10% visible tissue | Frozen gate |
| --- | ---: | ---: | --- |
| 262: 260-predicted ×{1,1.8} | 158/227 (69.6%) | 4/27 | Failed |
| 263: predicted ×{1,1.8,3} | 164/227 (72.2%) | 8/27 | Failed |
| 264: fixed 16×24-mm atlas-size pair | **214/227 (94.3%)** | **16/27** | Passed |

All eight synthetic plans exceeded 85% correctability in 264 (range
89.7–100%). Its 13 failures included two outside the fixed span tolerance
and seven outside the centre tolerance; these counts can overlap and are not
an exhaustive mutually exclusive breakdown. The 64 fixed antipodal normals
put an axis within 15° of every saved true normal, with median nearest angle
7.09° in 262. Merely adding a 3× multiplier to the image-predicted FOV only
rescued six cases because many predictions were nearly zero: all 65 of 262's
span failures had a predicted minor FOV below 40% of truth, and 47 were below
20%. The frozen 262 low-visibility stratum had median predicted/true minor FOV
2.73/16.13 mm. The result supports retaining an absolute atlas-size FOV
hypothesis rather than letting 260's collapsed span erase true-plane proposals.

**Necessary correction to the 264 protocol wording:** the assay compared
*sorted singular values*, so a later candidate bank must explicitly include
both 16×24 and 24×16 mm raster-axis assignments. Rotating or reflecting only
one rectangular render does **not** exchange its U/V edge lengths. With both
assignments in the bank, the unordered span test is a realizable necessary
condition. Neither assay checked exact roll-bin error, shear, finite atlas
support or image-based selection. It is deliberately optimistic: 94.3% is a
geometric ceiling, **not** 94.3% model accuracy or a guarantee that the local
updater can reach truth. These synthetic plans are not biological animals.

The scientific decision is to implement one bounded **spatial** image–atlas
candidate-scoring and residual-correction experiment with both FOV assignments,
nominal inference-available finite-thickness rendering and no inserted true
plane at inference. Its development readout must measure corrected candidate
capture separately from image-selected pose; an atlas-intensity-zeroed control
is needed to show it uses atlas anatomy rather than support shortcuts. Because
the 248 panel informed this architecture, reuse of the same synthetic plans
cannot establish independent generalization. Preserve held-out biological
animals for later calibration and validation. No GUI replacement, electrode
region probability or DeepSlice benchmark is justified now.

The three runners exited 0. Independent read-back found 227 unique physical
section IDs per run, exact raw-row and summary SHA-256 matches to each
`completed.json`, and recomputed correctable counts of 158, 164 and 214.
Frozen output roots are `I:/AnatomyTracker/runs/spatial_candidate_coverage_262_dev_001`,
`I:/AnatomyTracker/runs/spatial_candidate_fov_263_dev_001` and
`I:/AnatomyTracker/runs/spatial_candidate_atlas_fov_264_dev_001`.
