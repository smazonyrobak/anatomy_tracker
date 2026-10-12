# 255: frozen direct-head model misses good planes as well as choosing badly

The read-only 255 audit evaluated all 64 pose anchors under both raster reflections for each of the same 227 eligible synthetic DEV sections. It used full-resolution surviving-tissue coordinate targets and matched 254's selected branch, selected coordinate error, reflection-oracle error and normal-axis error case by case. The privileged oracle columns below use truth **only after** model proposals exist; they are not available to a researcher using the GUI.

| 254 branch choice | Subject-equal rigid coordinate error |
|---|---:|
| Image-selected top branch | 4.410 mm |
| Best of top four normal-mass anchors × two reflections (truth chosen) | 3.116 mm |
| Best of all 64 anchors × two reflections (truth chosen) | 2.290 mm |
| Best of three anchors nearest the true plane normal × two reflections (truth chosen) | 2.767 mm |

The best of all 128 branches is still worse than 1 mm in **227/227** sections. It is within 2 mm in 74/227 and within 3 mm in 200/227. The best candidate's median image-prior rank is 45 of 128; the image-selected branch is 2.120 mm worse than the all-branch oracle, while even the top-four oracle is 1.293 mm better than selection. All-branch oracle normal-axis error averages 39.90°, so the candidate set itself remains geometrically poor. Both proposal generation **and** selection need improvement; a better scorer alone cannot make this checkpoint accurate enough for electrode-region claims.

The next controlled intervention should supervise full pose factors explicitly, especially physical in-plane span and centre, instead of relying on the mean point-distance loss across several nearby normal anchors. That loss can reward a short predicted basis when roll or position is uncertain; this is a plausible mechanism for the observed span collapse, **not yet a proven cause**. A multimodal full-pose objective should retain multiple valid candidates rather than regress incompatible poses to one mean. [Implicit-PDF](https://arxiv.org/abs/2106.05965) provides primary evidence that single-image rotation prediction may need a multimodal distribution under symmetry and ambiguity; it does not establish the right architecture for histology. [DeepSlice](https://www.nature.com/articles/s41467-023-41645-4) demonstrates the importance of training scale but was trained on whole coronal sections and recommends grouped sections, so its success does not validate single arbitrary-plane localization here. Keep independent physical planes, provenance, optional brush/background variability, and synthetic DEV monitoring; do not scale the unchanged 254 loss.

The audit exited successfully and independent read-back verified 227 unique physical IDs across eight synthetic plans, exact source/protocol/row/summary SHA-256s, and no CAL/FINAL or public benchmark use. Its frozen output is `I:/AnatomyTracker/runs/pose_direct_capture_255_001`; all SHA-256s are bound in `completed.json`.
