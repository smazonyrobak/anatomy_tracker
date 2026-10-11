# 221 fixed-case diagnosis: selection, not warp gauge, is the main error

The completed read-only diagnostic replayed all 64 frozen framed 166→217→v3/v4 TRAIN-inner cases and verified their full normalized provenance and input-image SHA-256 against the 221 readout. It scored every one of the 160 pose/reflection branches from the 187 step-800 parent and 221 framed step-400 checkpoint at the same 128 surviving tissue pixels. The two targets were (1) the generator's known rigid affine plane and (2) the observed pixel-to-CCF coordinates after synthetic deformation and artifacts. The parent and adapted checkpoint bytes, frozen case file, diagnostic source, and output files are pinned in the completion receipt; both output hashes were rechecked. These are synthetic TRAIN-inner cases, not independent animals or a calibration set.

| Median of 64 cases | 187 parent | 221 framed 400 |
| --- | ---: | ---: |
| Known rigid plane to observed deformed coordinates | 0.122 mm | 0.122 mm |
| First-choice error to rigid plane | 3.629 mm | 3.424 mm |
| First-choice error to deformed coordinates | 3.673 mm | 3.450 mm |
| Best among top 16 to deformed coordinates | 1.634 mm | 1.590 mm |
| Best among all 160 to deformed coordinates | 1.515 mm | 1.474 mm |
| Median probability mass on candidates within 2 mm of rigid plane | 0.066 | 0.080 |
| First choices from 16 free modes | 64/64 | 64/64 |
| Truth-best deformed candidate from 64 anchor modes | 29/64 | 23/64 |
| Total anchor-family probability mass | 0.247 | 0.259 |

For the framed-400 checkpoint, the best rigid-truth and best deformed-coordinate branch is the *same* in 58/64 cases. The 0.122-mm rigid-to-dense residual is the discrepancy of this specific generator affine gauge; it is **not** a mathematical lower bound over all possible rigid fits. Its small size nevertheless rules out that gauge discrepancy as an explanation for the 3.450-mm first-choice error. The nearest fixed anchor normal is 7.74° from truth at the median and the nearest learned anchor normal is 6.36°, whereas the selected normal is 39.44° away. Thus the model has useful directional proposals but assigns the wrong branches the largest probabilities. It still has a proposal problem: only 53/64 cases have *any* branch within 2 mm, and the best-160 median remains 1.474 mm.

An exploratory constant bonus of 1, 2, 3, or 4 log units to every anchor branch on these same 64 cases did not rescue selection: the median deformed-coordinate error was 3.535, 3.450, 3.664, or 3.368 mm, versus 3.450 mm without a bonus, and <2-mm first-choice cases fell from 16/64 to as low as 8/64. This is a within-panel mechanism check, **not** a fitted method or a new validation result. The problem includes within-family ordering, not just the anchor-family prior offset.

**Decision:** do not promote, calibrate, publicly benchmark, or GUI-install this checkpoint. The next bounded model change should train pose-branch *selection* more directly against known physical plane errors while retaining explicit anchor proposal supervision and older TRAIN rehearsal. Evaluate on a newly drawn TRAIN-inner panel as well as the frozen 221 panel, and keep the existing parent/221 checkpoints as controls. Do not add a more flexible deformation scorer to compensate for a wrong rigid plane. The arbitrary-angle objective stays active, but no synthetic panel can establish accuracy on unseen physical steep-oblique sections.

Frozen diagnostic: `I:/AnatomyTracker/runs/pose_framing_221_affine_dense_diagnostic_001`. Its `rows.jsonl` retains per-case IDs, input hashes, full branch errors and prior scores, family-specific best errors, and tissue visibility. No DEV/CAL/FINAL data, calibration, prior external weights/features/pseudolabels, public benchmark, or GUI installation was used.
