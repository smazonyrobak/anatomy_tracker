# A measured cutting-plane direction could reduce synthetic ambiguity; it is not a current GUI constraint

This was a nine-second, read-only **hypothetical metadata** diagnostic on the already inspected 256-case v10 TRAIN-inner panel and frozen 309 batch-20,000 checkpoint. It was not training, independent validation or a model improvement. Each approximate hint was generated from the synthetic true post-artifact plane: AP-centre noise had 0.75-mm standard deviation, cut-normal angular noise had 15° standard deviation, and the score used broad 1.8-mm and 25° Gaussian penalties. Candidates, images and model weights stayed fixed. The median realized hint errors were 0.505 mm and 11.0°. These hints use ground truth to simulate a measurement; they must **never** enter an image-only result.

| Frozen candidate choice | Mean surviving-site error | Within 1.5 mm | Over 5 mm |
| --- | ---: | ---: | ---: |
| Image-only, original first choice | 2.469 mm | 109/256 | 17/256 |
| Noisy AP-centre hint, original top 16 | 2.425 mm | 112/256 | 17/256 |
| Noisy cut-normal hint, original top 16 | 1.647 mm | 175/256 | 6/256 |
| Both noisy hints, original top 16 | 1.587 mm | 178/256 | 5/256 |
| Both noisy hints, all 160 branches | 1.551 mm | 181/256 | 5/256 |
| Both hints, twice the angular noise, top 16 | 1.783 mm | 158/256 | 7/256 |

The top-16 truth oracle is 0.931 mm and the all-160 truth oracle is 0.826 mm; neither is selectable at inference. A separate exploratory stress arm made about 10% of hints grossly wrong by adding 60° and 3.5 mm. Its overall mean was 1.674 mm, but on the 27 deliberately wrong-hint cases it worsened the image-only 2.621-mm mean to 2.716 mm. A simple 10% floor on the hint factor worsened this subset further to 3.003 mm, so that ad hoc robustness rule is rejected. The hint penalties were chosen for an information probe, not fitted or calibrated posteriors.

**Interpretation:** approximate knowledge of the *cutting-plane normal* could be valuable because much of 309's error is orientation selection, whereas AP alone helps little under this setup. But the existing Anatomy Tracker GUI records **probe insertion attack angle**, not the tissue cutting-plane normal; the two cannot be substituted. Existing AP bounds likewise have legacy coronal semantics, not the synthetic true centre used here. Therefore none of these numbers measures the gain from the current surgical controls, supports a new GUI requirement, rescues blind arbitrary-plane accuracy, or licenses an electrode-region probability. Continue the image-only problem and native integration of actual marked-track and surgical constraints on their own valid geometry; use an optional cut-normal measurement only if it really exists in acquisition metadata or is explicitly supplied, with uncertainty and wrong-hint handling verified on held-out animals.

Reproducible flat script: `training/diagnose_first_choice_309_acquisition_hints.py`. Its run wrote no artifact and used only `I:` model files. This adaptive TRAIN-inner probe is deliberately not a qualification gate or a DeepSlice comparison.
