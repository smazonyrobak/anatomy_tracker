# Direct-pose atlas patch gradient 035 — frozen development result

The preregistered necessary gate **failed** on 24 fixed synthetic DEV sections from eight synthetic deformation identities. The frozen 019 direct pose model and 025 patch descriptor were not trained or altered. The correctly placed atlas plane scored above the median of eight predicted planes in **24/24** cases, so the evidence recognizes a supplied correct match. But it did not provide useful guidance from the actual predictions:

| Full-tissue rigid error | Mean |
| --- | ---: |
| 019 prior top-one | 2.902 mm |
| Highest patch-score among prior top-eight | **3.116 mm** |
| Best of those eight, truth-selected oracle | 1.294 mm |

The patch-score choice improved on the prior in only 8/24 cases. More decisively, a bounded ascent step along the differentiable patch-score gradient improved physical error for the prior top-one in only **11/24** cases. In normalized full-frame update coordinates, mean cosine of patch-score ascent with physical-error descent was 0.031 at the prior top-one, 0.044 at the oracle-best of eight, and 0.244 when starting near the true plane (a controlled 0.1-rad/1-mm perturbation). The latter suggests a narrow local signal, not capture from present predictions. The score and physical error were evaluated independently; the true pose was never in the inference pool.

This directional test is distinct from 026's wrong-proposal rank correlation but agrees with its conclusion. Simply connecting the current 025 diagonal patch cosine to the 019 pose head would supply a gradient without reliable anatomical direction. Do **not** train that unchanged route or present nonzero gradients as fitting feedback. The next trainable design must widen global pose capture and learn a structured, spatial correspondence/fit signal that is supervised on actual off-plane candidates, then compare its pose movement against a matched atlas-detached control. Keep constrained local deformation separate enough that it cannot hide a wrong plane. These synthetic DEV identities have now been consulted often; use a new identity-disjoint synthetic development panel for any future checkpoint selection, and reserve independent animal-level real validation.

The independent read-only verifier passed frozen file hashes, row counts, identity uniqueness, selected-branch arithmetic, mean errors, gradient fractions and the gate recomputation. Frozen raw run: `I:/AnatomyTracker/runs/atlas_pose_patch_fit_035`. Source: `training/diagnose_atlas_pose_patch_fit_035.py`, `training/atlas_pose_patch_fit_035.py`; verifier: `training/verify_atlas_pose_patch_fit_035.py`. This is neither calibrated uncertainty nor real-animal accuracy, a deployed model, or a DeepSlice comparison.
