# Fitted atlas image-information diagnostic 022 — result

The frozen 019 batch-18,000 top-eight planes were re-rendered at 96 and 256 pixels for the same 185 synthetic DEV sections. Candidate identities exactly matched the earlier 019 diagnostic. The independent verifier passed checkpoint, panel, control-row, source, output and summary hashes. No network was trained, no image grid was inspected, and no public benchmark or final-test animal was touched.

Lower local contrast disagreement was used to select a candidate. The first mask in each pair is the **known synthetic surviving-tissue mask**, an impossible oracle at inference; the second is the parent's predicted tissue reliability. Errors below are identity-equal mean physical mapping errors for the *same* eight fitted planes.

| Selector | Selected mapping error | Physical-best candidate picked |
| --- | ---: | ---: |
| Existing learned 019 fitted score | **2.591 mm** | **43.2%** |
| 96-pixel local contrast, oracle tissue mask | 2.979 mm | 30.3% |
| 96-pixel local contrast, predicted tissue support | 2.966 mm | 31.4% |
| 256-pixel local contrast, oracle tissue mask | 3.034 mm | 28.1% |
| 256-pixel local contrast, predicted tissue support | 2.981 mm | 27.0% |
| Physical best of eight, not observable at inference | 1.143 mm | 100% |

Thus neither access to the perfect tissue mask nor simply rendering at higher resolution makes the present proposed planes rankable by local image agreement. This is not a claim that the slice contains no localization signal. As a sanity check, the atlas rendered at each synthetic section's **exact observed-pixel-to-CCF surface** has much lower disagreement than even the most image-similar proposed plane: mean 0.323 versus 0.685 at 96 pixels, and 0.458 versus 0.786 at 256 pixels. The exact surface beats the best proposed-plane score in 97.8% and 98.4% of cases, respectively. That exact surface and mask are available only as synthetic truth; this check establishes an information/capture-range gap in the experiment, not a deployable selector. It does not prove that every real section is identifiable from one image.

Interpretation: the 019 model usually proposes planes too far from the exact tissue surface for a narrow local similarity basin to rank them reliably. The best-eight physical mapping error of about 1.14 mm is still far above the 0.129 mm exact-pose mapping error. At that distance, anatomy can differ substantially and a wrong plane may coincidentally have lower local contrast cost than a *closer but still wrong* one. The failed 020 scalar scorer and 021 dense scorer are consistent with this explanation. Increasing comparison resolution or substituting an oracle tissue mask did not rescue them; adding another scorer of the same candidates is low priority.

The next development target is **proposal capture**, not further tweaking local fit metrics. Quantify the true-pose neighbourhood in which atlas evidence is monotonic, then improve the randomly initialized global pose/retrieval stage so that it supplies candidates in that neighbourhood across all brain-intersecting orientations, including partial-tissue cases. A learned coarse-to-fine global matcher or a much stronger direct pose head may be justified, but prior image-key and 32-branch catalogue failures mean a broad gallery alone is not enough: it must be tested for *near-truth physical capture* and final selected error. Keep 019 batch 18,000 as the internal checkpoint, do not deploy or claim calibrated electrode-region probabilities, and defer DeepSlice benchmarking.
