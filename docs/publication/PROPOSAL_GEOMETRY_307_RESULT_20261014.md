# 307: more full-pose guesses recover some candidates but fail blind use

**Decision: fail; do not promote or extend the two-slot anchor recipe unchanged.** The preregistered primary v10 natural-panel gain was at least six captured sections over the trained one-slot control. After 800 two-image batches per arm, capture within 1.5 mm was **57/64 versus 54/64**: a gain of three. The two-slot arm also worsened v9 first-choice mean rigid-site error from **2.623 to 2.837 mm**, exceeding the allowed 0.20-mm regression by 0.014 mm. These are frozen TRAIN-inner synthetic results, not physical oblique validation or an electrode-localization claim.

| Batch 800, fresh independent sections | One-slot control | Two-slot arm |
|---|---:|---:|
| v10 natural, any candidate ≤1.5 mm | 54/64 of 160 branches | 57/64 of 288 branches |
| v10 natural, top 16 ≤1.5 mm | 48/64 | 44/64 |
| v10 natural, first-choice mean | 2.658 mm | 2.755 mm |
| v9 natural, any candidate ≤1.5 mm | 56/64 | 57/64 |
| v9 natural, top 16 ≤1.5 mm | 55/64 | 54/64 |
| v9 natural, first-choice mean | 2.623 mm | 2.837 mm |
| v10 sparse `<15%` tissue, any candidate | 8/16 | 11/16 |
| v10 fuller `≥35%` tissue, any candidate | 15/16 | 16/16 |

The two-slot arm started at 53/64 v10 natural captures from the extra perturbed guesses and reached 57/64 by batch 400, staying there at 800. Its average within-anchor five-point separation on the 64 natural v10 cases was 2.42 mm at batch 800. Among the 64 first choices, 58 came from the old free-mode family, six from the first anchor slot, and zero from the second. **This zero is expected by design:** the geometric-only loss gave no ranking gradient to the initially tied slot scores, so a deterministic argmax cannot prefer the second slot. It is not evidence that a properly trained selector could never use it. The top-16 decline shows that merely adding unranked oracle branches is not sufficient. The first-choice guard failure partly reflects the mass split and changes to the original slot, not a fair test of learned two-slot probabilities.

The preceding read-only 305 parent decomposition found all 13 v10 and all six v9 all-branch misses nevertheless had at least one predicted normal within 15° of the true unsigned normal. The normal anchors themselves were also within 15° in every miss. This supports targeting the paired position/roll/field-of-view prediction, but does not prove that those parameters are identifiable from every image. The 307 result only weakly supports additional within-anchor hypotheses; it rejects this particular simple head-only recipe as a solution.

The run used 192 new, frozen TRAIN-inner panel sections (96 v9 and 96 v10, each including 64 natural, 16 sparse, 16 fuller) and 1,600 accepted independently drawn TRAIN-gradient sections per arm from 56 separate synthetic deformation bases. No sample was deliberately replayed with different artifacts/backgrounds. The parent descends solely from project random initialization. No acquired images entered 307 gradients, and no DEV/CAL/FINAL animal, DeepSlice label, probability calibration, or GUI model was touched. Raw frozen output: `I:/AnatomyTracker/runs/proposal_geometry_307_001`; parent SHA-256 `56d10febb1a6e6592adf18b9b4c7fe1fe0beae2a9179d9ddaffb78eefb82e692`.

There was one implementation failure: after the control finished, the intervention halted **before its first batch** because initialization jitter was on CPU rather than GPU. The corrected script resumed the preserved panel, schedule and control without repeating them. The completion receipt binds the original control source hash and corrected intervention source hash separately. The independent post-exit verifier passed all frozen output hashes, parent/protocol/current-source bindings, unique section IDs and counts, reconstructed readout means/capture counts, and the exact failed gate. The original pre-correction source bytes were not separately committed, so its hash is a recorded binding rather than a source file that can be re-read from this checkout. This provenance limit does not alter the raw paired readout or failed gate.

**Process decision.** Stop stacking tiny candidate heads and synthetic-only rerankers on the same weak proxy. The next high-value work is a bounded audit of the real training/validation material and label fidelity, with hard physical-oblique evidence sought explicitly. Then choose one coherent joint pose-and-fit training study only if its input/atlas signal and proposal-capture mechanism are credible. Retain the best random-initialization lineage and every failed gate; do not buy another long run, calibrate electrode-region probabilities, benchmark against DeepSlice, or replace the GUI from 307.
