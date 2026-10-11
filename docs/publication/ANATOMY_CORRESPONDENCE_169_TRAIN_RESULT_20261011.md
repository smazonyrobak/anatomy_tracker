# 169: synthetic anatomy correspondence learned, localization unproven

The TRAIN-only stage exited successfully after 6,000 optimizer batches and 6,442 independently drawn section attempts; 442 attempts were blank. Its frozen receipt and all listed output SHA-256 hashes were checked after exit. The 96 fixed inner attempts came from four other synthetic TRAIN deformation bases; 88 had usable tissue. No section was deliberately repeated with only changed appearance.

| Batch | Correct atlas intensity beats fixed-support wrong intensity | True plane beats displaced plane | Mean local rigid error before → after |
| ---: | ---: | ---: | ---: |
| 0 | 44.3% of 70 pairs | 56.8% of 88 | 0.700 → 0.700 mm |
| 1,500 | 76.1% | 75.0% | 0.700 → 0.649 mm |
| 3,000 | 73.2% | 69.3% | 0.700 → 0.680 mm |
| 4,500 | 83.1% | 71.6% | 0.700 → 0.639 mm |
| 6,000 | 85.7% | 75.0% | 0.700 → 0.631 mm |

The dense correspondence cross-entropy fell from 4.782 to 4.422. These are useful signs that the newly initialized image-to-atlas descriptors learned anatomical content and a modest near-plane correction. The fixed-support comparison is specifically harder to pass by using tissue outline alone. It does **not** establish that the fitter selects the right plane from the direct model's blind candidates; the displaced-plane decision still loses one in four cases. The local correction is also much too small to rescue a several-millimetre first-choice pose error.

Run 170 therefore compares every frozen checkpoint on the *same fresh blind-16 candidate beams*, against the direct prior, native fitter and atlas-intensity-null controls. The 169 checkpoint must not be selected from this TRAIN panel for deployment. All inner subjects still share one atlas, the acquired-image domain and physically cut oblique specimens remain untested, and no posterior or electrode-region probability has been calibrated. This is not a GUI replacement or a DeepSlice comparison.

Frozen run: `I:/AnatomyTracker/runs/anatomy_correspondence_169_train_001`; `completed.json` (SHA-256 `e0b02f62938ca64c28ab5c74eb407d0468dfead9f440070172f9784e307099ea`) binds `config.json`, `training.jsonl`, five per-case inner reports and all five checkpoints.
