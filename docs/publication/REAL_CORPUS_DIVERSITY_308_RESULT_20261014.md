# 308: more coronal donors at fixed exposure did not help

**Decision: the predeclared data-diversity gate failed.** Two continuations began from the same randomly initialized-project-lineage 305 checkpoint and AdamW state. Each received 4,000 matched four-image batches. The only planned difference was the coronal training slot: 2,000 physical-section presentations from 20 donors in the control, versus 2,000 presentations spanning all 1,727 eligible TRAIN donors in the broad arm. Synthetic, Lupori and sagittal draws were matched. The 100 entirely new coronal readout donors and the 58 previously used coronal donors were excluded from both arms' coronal gradients.

| Frozen batch-4,000 readout | 20-donor control | 1,727-donor arm |
| --- | ---: | ---: |
| New 100-donor coronal median centre gap to weak Allen registration | **0.718 mm** | 0.737 mm |
| New coronal first-choice dominant axis | 100/100 | 100/100 |
| Held-out 20-donor sagittal first-choice dominant axis | 20/20 | 20/20 |
| `AL1A` Lupori mean first-choice gap to approximate human plane | **1.239 mm** | 1.258 mm |
| Independent v9 synthetic first-choice mean site gap | **2.106 mm** | 2.108 mm |
| Independent v10 narrow-FOV synthetic first-choice mean site gap | **2.732 mm** | 2.733 mm |
| v9 / v10 top-16 ≤1.5-mm candidate captures | 122/128 / **105/128** | 122/128 / 104/128 |
| v10 `<15%` **valid-pixel** top-16 captures | **18/32** | 17/32 |

The broad arm needed to improve the new-coronal median by at least 0.15 mm, while retaining the other readouts. It was instead 0.019 mm worse; all eight other gate clauses passed. On the 100 matched coronal cases, broad was individually better on 57 and worse on 43, but its mean gap was 0.782 versus 0.771 mm and its empirical 90th-percentile gap was 1.119 versus 1.043 mm. This is no evidence of a useful donor-breadth gain **at fixed 2,000 coronal presentations**. It does not show that the larger corpus is useless with greater exposure, nor that 20 donors suffice for a final model.

Both arms' new-coronal median moved from 1.642 mm at batch 0 toward approximately 0.7 mm, and `AL1A` moved from 1.756 toward 1.2–1.3 mm. Those are continuation gains under a shared mixed training schedule, not an isolated effect of coronal donor diversity. The first-choice synthetic v9 and v10 errors remain about 2.1 and 2.7 mm, with roughly 4.0 mm on the nominal sparse v10 panel. Candidate availability is much better than blind first choice, so neither arm is suitable for electrode-site localization, calibrated probabilities, a DeepSlice comparison, or GUI replacement.

The sparse panel has a protocol wording error: the runner's `<15%` test used `valid_mask.mean()`, which excludes invalid overlap, folds, bubbles and other pixels, rather than `visible.mean()` for all visible tissue. Its exact frozen test is therefore **less than 15% valid pixels**, not demonstrated less than 15% visible tissue. This secondary guard passed numerically but cannot support a physical sparse-tissue claim. The primary donor-diversity failure is unaffected.

The Allen coordinates are authored registrations, not independent measured cut planes; the coronal metric is a centre gap to that weak reference and dominant-axis agreement, not a full physical-plane or deformation accuracy measurement. The Lupori plane is also approximate. No local field, atlas-fit feedback, acquisition constraints or uncertainty head was trained in 308. This was a TRAIN-inner mechanism test, not an animal-level qualification. No DEV/CAL/FINAL or public benchmark image entered the run.

**Process consequence:** stop this fixed-exposure donor-breadth recipe rather than extending it or adding another small candidate head. The observed 305 2,000→26,000-batch learning curve still improved first-choice synthetic pose, while 307 rejected extra unranked guesses and 308 shows donor count alone is not the present bottleneck. The next bounded study should target *first-choice* pose and a directly supervised local map in the existing one-pass model, with independently drawn full-sphere synthetic sections and acquired rehearsal, explicit learning-curve checkpoints, and the same strict animal/donor separation. Scale presentations only while held-out first-choice and mapped-site errors improve. Keep atlas-fit feedback disabled unless it beats support-only and shuffled-intensity controls. Independent physically cut oblique animals with measured 3-D planes remain necessary for any arbitrary-plane accuracy claim.

Frozen output: `I:/AnatomyTracker/runs/real_corpus_diversity_308_001`. Root completion receipt SHA-256 `8c5e777c8714ee35820972744315c53b7a9f09bed208683c3a8239c16075c06e`. `training.verify_real_corpus_diversity_308` exited 0 with `verified: true`; it checked exact source/protocol/parent/split and output hashes, matching noncoronal draw identity, readout rows and the failed gate. The raw readouts, per-batch losses/draws, panel and all eight checkpoints remain frozen on `I:`.
