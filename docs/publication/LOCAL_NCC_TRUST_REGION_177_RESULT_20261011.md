# 177: local anatomy matching helps nearby planes, but misses the gate

The frozen TRAIN-only assay completed 128 independent full-sphere physical
section draws: one from each of 64 deformation bases with v3 appearance and
one from each with v4. It did not retry a plane to make a different-looking
copy. Of those draws, 48 v3 and 37 v4 sections had a matching-reflection
member of the frozen model's blind 16-plane beam initially 0.5–3 mm from
truth with adequate atlas support. The other 43 were excluded by the
predeclared eligibility rule (42 had no such near branch; one lacked support).
Truth selected a branch **for diagnosis only**, after blind proposals were
fixed; neither fitter received it. These numbers are not blind accuracy.

| Same near branch, mean five-point error | v3 (48) | v4 (37) |
| --- | ---: | ---: |
| Before fitting | 1.776 mm | 1.658 mm |
| Old 48-pixel global fit (176) | 1.763 mm | 1.643 mm |
| New 96-pixel local-NCC/trust fit | **1.651 mm** | **1.557 mm** |
| Matched permuted-atlas-intensity fit | 1.775 mm | 1.719 mm |

The new fit moved 39/48 v3 and 33/37 v4 near branches physically closer to
truth; the intensity-null did so in 27/48 and 11/37. Full-intensity physical
gain was **0.125 mm v3** and **0.100 mm v4**; the full-over-null advantage was
0.124 and 0.162 mm. Mean fixed-mask support retention was 99.90% and 99.87%,
so erasing or leaving the scored brain support does not explain the mean
gain. The exact 176 fitter on these same branches gained only 0.013 and
0.014 mm. The new method therefore has a genuine *conditional local*
anatomical signal. The observed 45–55° eligible strata had only five v3 and
one v4 section, and support no robust steep-oblique claim.

The [predeclared protocol](LOCAL_NCC_TRUST_REGION_177_PROTOCOL_20261011.md)
required at least 0.15 mm physical gain in **both** appearances, at least
65% branches improved, at least 0.10 mm full-over-null advantage, and at
least 99% mean support retention. The first criterion fails in both. The
continuation gate is **negative** despite the other three passing. Do not
lower the threshold after seeing this panel, enlarge this optimizer
unchanged, or call the oracle-selected near-branch result a deployable
aligner. No pose-head learning, blind candidate selector, nonrigid joint
training, real-animal reference, uncertainty calibration, GUI replacement,
or DeepSlice benchmark was evaluated here.

This narrows the next model step: prioritize image-conditioned selection and
correction *among the model's own blind candidate planes*, using spatially
grounded atlas/section comparisons and physically constrained tissue mapping.
The local NCC signal can be used only inside a credible basin. A matched
full-feedback versus detached-feedback training comparison is still needed
before saying atlas fit improves the direct pose head; donor-separated
acquired and physical oblique validation remain later gates.

Post-exit audit matched every frozen output, source, protocol, and 132 parent
checkpoint SHA-256. Independent raw-row aggregation reproduced 128 draws,
128 unique physical-section IDs, 85 eligible rows, per-appearance means, and
39/48 plus 33/37 improvement counts. Frozen output:
`I:/AnatomyTracker/runs/local_ncc_pose_177_train_001`. The summary SHA-256
is `b8d1762d72740fb65b2d8d7e03354937ef1f778308f56ae33347c82de441039c`.
