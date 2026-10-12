# Independent v7 synthetic DEV panel and pose-mechanism readout (248–252)

## Frozen evidence

The 248 panel contains 256 independently drawn physical planes from eight new, independently sampled 3D synthetic deformation plans (indices 2000–2007). There are no deliberately paired versions of a section. All attempts were retained: 227 meet the predetermined tissue-visibility rule and 29 do not. Appearance modes are 187 raw/background-variable, 33 exact-black, and 36 imperfect-brush. The angle to the nearest principal axis, in 15° bins, is 27/64/132/33 attempted sections. The 227 eligible sections contribute 27/58/116/26 to those bins. All eight inverse-map grids passed their separate 4,096-point sampled error check at 100 µm spacing; their maximum sampled error is 0.721 µm (the limit was 5 µm). These are eight synthetic variations of **one** Allen atlas, not eight biological animals.

The public read-only `python -m training.verify_one_shot_v7_heldout_dev_panel_248` completed successfully. It rechecked panel and source hashes, section and subject identities, all 256 file hashes and arrays, eligibility, and each inverse-map accuracy record. Panel completion SHA-256: `55c4d8c5080cfc1bf8e191729ef02f9a9dfe634211f986de75f4f90aa80a4e4a`. The fixed 247 checkpoint was chosen before this panel existed. The 249, 250, and 251 checkpoints were likewise fixed before their same-case 252 DEV readout. No CAL/FINAL data or public DeepSlice benchmark were used.

The 248 baseline readout (`abd1bf3c9cafa5de0c60387119c8d7a5dbf608da37d6ad0eb4992be1ec3765d1`), 249 readout (`541ab5ed80605eced36ef599438c9047ef6e38f009922d334117870a73067d84`), and 252 readout (`6c29f76abf1eb1aa36a5e071aa0e1371434485b065e52f9f85c16827b39d7ced`) are their completed-file SHA-256 values. Each exited successfully; their 227-row and summary hashes, source/protocol bindings, panel binding, and unique case IDs were independently rechecked. All displayed means below give each of the eight synthetic subjects equal weight. Units are degrees for plane-normal axis error and millimetres for tissue-to-atlas coordinate error.

## Result

| Frozen model/readout | Normal-axis error | Tissue-to-atlas error | Interpretation |
|---|---:|---:|---|
| 247 image-only top choice, before fitting | 54.66° | 4.247 mm | Poor plane capture. |
| 247 atlas-fit selected choice, after refinement | 54.52° | 4.151 mm | Fitting barely changes pose. |
| 247 shifted-atlas-intensity null, after fitting | — | 4.185 mm | Real atlas intensity improves the selected coordinate error by only 0.034 mm over this null. |
| 249 shared-anchor head, image-only | 56.68° | 4.320 mm | Worse than 247. |
| 250 shared-anchor CE-only head, image-only | 56.25° | — | Removing the rigid loss did not rescue orientation. |
| 251 direct 64-anchor CE-only head, image-only | 52.78° | — | Small gain, inconsistent across plans; four of eight plans improve over 247 and four worsen. |

The same image-only top-anchor axis metric used for all four models in 252 gives 247/249/250/251 errors of 54.62°/56.70°/56.25°/52.78°, respectively. These are the directly comparable 252 figures; 247's 54.66° above uses the earlier reflection-aware readout. The last 500 training-batch normal-classification losses for 249/250/251 were 4.15784/4.15680/4.07617 nats versus 4.15888 nats for uniform mass over 64 anchors. The prespecified 249 continuation gate and both 252 head-comparison gates failed.

With the *true* pose supplied to 247's mapping head, its 192-pixel coordinate error was 0.2105 mm. The same true pose with **no learned warp** gave 0.2138 mm. Thus the model's learned local warp added only about 0.0033 mm in this favorable condition; it has not demonstrated useful deformation recovery. This exact-pose result is an oracle diagnostic, not performance obtainable from the image-only model.

These numbers support a narrow conclusion: the current models do not localize arbitrary planes reliably on new synthetic deformations, and the present atlas-fit feedback and local map do not repair that. The data do **not** establish that arbitrary-plane recovery is impossible. A single cut with little visible tissue or near-symmetric anatomy can be intrinsically ambiguous; successful deployment will need to represent ambiguity and exploit optional acquisition constraints or multiple sections when available. No calibrated probability, electrode-region claim, GUI replacement, or benchmark-beating claim is justified.

The four fixed-stratum visual examples in `I:/AnatomyTracker/previews/v7_dev_253_true_input_joint247_fit.png` compare the true atlas plane, actual v7 generator output, and 247's fitted atlas plane. Their sidecar records the physical field of view (singular values of the in-plane basis). Three of four guesses have a roughly 1 mm minor field-of-view axis, near the model's lower span limit, while the corresponding true axes are 12–20 mm. One example has only 7.05° plane-normal error yet 3.24 mm coordinate error because its field of view and centre are wrong. These four examples diagnose a specific failure mode; they do not establish its prevalence across all 227 eligible sections.

## Development decision

Do not scale or deploy the unchanged 247/249/250/251 architecture. First inspect actual v7 source/augmented/guessed-plane examples for anatomical and artifact plausibility and target correctness. Then run a focused pose-learning intervention: retain the one-pass joint pose-and-map objective, but give the pose pathway a stronger direct full-pose supervised signal and adequate presentations, with the 248 same-case synthetic DEV panel monitored at coarse checkpoints. An atlas-render/fit-feedback path may be retained only if a matched predicted-plane comparison shows a gain; the current null and exact-pose results do not justify assuming it works. Keep all optimization on I:, preserve the untouched biological CAL/FINAL split, and delay full public benchmarking until independent internal and biological gates pass.
