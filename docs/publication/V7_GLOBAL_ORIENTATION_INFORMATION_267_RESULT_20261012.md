# Global cutting-angle information in synthetic sections

The frozen 267 assay found that a whole-slice atlas comparison can distinguish the correct cutting angle from distant alternatives **when the true centre, in-plane framing, reflection and slice thickness are supplied**. It does not locate a slice blindly. On 64 preselected v7 synthetic DEV sections, 54 had at least 16 support-comparable wrong angles. In those 54, the correct render beat every eligible wrong render in 53 cases; the eight-plan-equal success was **98.21%**, versus **1.79%** after atlas intensities were shuffled within their tissue support. The predeclared conditional-information gate passed.

| Frozen readout | Value |
| --- | ---: |
| Independently sampled physical sections | 64 from eight synthetic deformation plans |
| Fixed candidate axes per section | 112, excluding angles <30° from truth |
| Sections with ≥16 support-comparable distant axes | 54/64 |
| Eligible true-versus-wrong comparisons | 4,260 |
| Correct render beat every eligible distant angle, equal-plan mean | 98.21% |
| Shuffled-intensity control, same measure | 1.79% |

Each wrong plane used the exact true physical centre and raster span/shear; its in-plane axes were carried by the smallest 3-D rotation that changed the true normal to the candidate normal. The true reflection and through-plane point-spread function were also supplied. Scores were absolute image–atlas correlation on eroded common atlas support, with the same support-overlap rules as 257. Candidate selection and exclusions were fixed before scores were read. The independent verifier matched all output and input hashes, the first-eight-per-plan selection, every one of 7,168 axis records, summary arithmetic, and 140 independently rerendered scores across eight cases.

Ten sections failed the minimum support-comparison coverage rule. Nine of these had <10% valid visible tissue; **none** of the nine very-low-visibility sections qualified. Among qualifying sections there was one intact failure, a raw section with 64 eligible distant axes whose hardest wrong angle was 51.36° and exceeded the true score by only 0.00114. The result is therefore evidence of global angle information for sufficiently visible *atlas-derived synthetic* sections under oracle nuisance values, not evidence of flawless behavior for slivers, damaged tissue, acquired histology, unknown framing, or arbitrary physical oblique cuts. Eight synthetic plans share one Allen atlas and are not eight biological animals.

**Development decision:** do not scale the unchanged 266 orientation-agnostic atlas-voxel codebook or repeat the short 64-bin image-only normal experiment. The full-angle failure so far is in blind pose capture/selection, not a demonstrated absence of atlas angle information. The next model must represent each candidate's actual plane orientation and physical frame, predict a multimodal full pose from the image, and couple its local tissue map to pose-specific atlas fit. Train a scratch direct joint branch first; retain a render-and-update branch only if a matched comparison improves blind physical mapping without moving a supplied correct plane away. Candidate losses must compare anatomy at similar support to avoid the measured 090 support shortcut. Keep acquisition constraints optional and probability outputs uncalibrated until held-out animal-level calibration. No public DeepSlice benchmark is justified yet.

The frozen output is `I:/AnatomyTracker/runs/v7_global_orientation_information_267_dev_001`; `completed.json` SHA-256 is `65490105a449205affd42a16739da47f80d45b0e686ceeb8e982152695cc4263`. Reproduce the audit with `training/verify_v7_global_orientation_information_267.py` in the pinned I: environment. No earlier model weights, acquired DEV/CAL/FINAL animals, or public benchmark data entered this assay.
