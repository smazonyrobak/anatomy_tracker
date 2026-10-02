# Global oriented-patch capture 030 — frozen development result

The preregistered necessary gate **passed**, and the independent verifier passed all bank/output hashes, raw-row CCF distances, provenance and group-weighted metrics. The atlas-only bank contained 4,027 positions × 512 frames = 2,061,824 rendered patches. Its construction used the pinned Allen atlas and scratch-trained frozen 025 checkpoint, but no DEV image, coordinate, pose or derived label. Bank construction took 345 seconds. The later 185-section lookup took 6.9 seconds after loading. The bank stores 128-component float16 descriptors (about 528 MB plus small geometry arrays), so memory/runtime still require eventual deployment measurement.

| Fixed synthetic DEV endpoint | Result |
| --- | ---: |
| Top-1 atlas point within 500 µm | 1.86% |
| Any top-16 atlas point within 500 µm | **15.42%** |
| Sections with ≥3 such matches separated by ≥1 mm | **67.49%** |
| Top-16 point recall, raw / black / imperfect brush | 13.95% / 18.42% / 13.81% |

Eight synthetic deformation identities were weighted equally. The preregistered thresholds were ≥10% top-16 point recall, ≥20% sections with three separated matches, and no appearance mode below half the overall recall. All passed. The bank-grid **geometric ceiling** on these sampled truth points was 49.22% within 250 µm and 99.85% within 500 µm. Among the 512 frames, 29.73% of true frames had a nearest frame within 10° and 88.65% within 20°. These ceilings were computed from known DEV geometry only for diagnosis, not to construct the bank or select matches. Median observed pixel scale was about 67–68 µm/pixel; its 10th–90th percentile was roughly 53–77 µm/pixel, while the bank used fixed 67 µm/pixel.

This is a materially wider global capture signal than the previous **single invariant 3D-cube descriptor** approach in 024, but the experiments used different banks and cannot be read as a matched effect size. A true match merely appears *somewhere* among 16; the top match is correct only 1.86% of the time. The queried pixels were sampled from the generator's known valid-tissue mask. A real inference system would have to choose query pixels from the image/optional brush and find a geometrically consistent plane without truth. The fixed 031 consensus experiment is therefore the next decision: select matches by their agreement on one image-to-atlas plane and test resulting physical error. This result establishes no real-animal accuracy, joint fit-feedback learning, calibrated uncertainty, GUI usability or DeepSlice superiority.

Frozen bank: `I:/AnatomyTracker/runs/atlas_global_patch_030`. Frozen evaluation: `I:/AnatomyTracker/runs/atlas_global_patch_030_development_eval`. Verifier: `training/verify_atlas_global_patch_030.py`.
