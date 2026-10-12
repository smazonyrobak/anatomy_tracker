# 283: frozen mapper fits seen sections only weakly

The preregistered read-only diagnostic completed at `I:/AnatomyTracker/runs/image_map_train_check_283_002`. A first attempt (`..._001`) stopped before scoring because a Python tuple/list provenance comparison did not normalize the JSON-recorded form; it is preserved and excluded. The corrected runner was frozen at `15e1a73`. The completed receipt's source, protocol, parent-282, row, and draw hashes match; it contains 64 distinct seen training sections and 64 independently drawn new sections from the same 56 TRAIN bases, each scored at 512 surviving sites. The fresh tier required 71 attempts. No weights were updated, and the 282 step-1,500 smooth16/full-atlas checkpoint was unchanged.

| Tier | Sections | No-warp error (mm) | Mapped error (mm) | Gain (mm) | Nonnegative gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seen during 282 training | 64 | 0.208756 | 0.205379 | +0.003377 | 48/64 sections |
| Fresh planes, same TRAIN bases | 64 | 0.203645 | 0.203158 | +0.000486 | 29/64 sections |
| New bases 56–63 from verified 282 readout | 64 | 0.191819 | 0.192047 | -0.000229 | 4/8 bases |

The selected seen sections were the latest eligible ≥512-pixel section per training base plus eight order-spaced extras, chosen without inspecting map error. Each fresh section was an independent new plane, virtual-variant draw, appearance, and artifact realization from its paired base—not a background variant of the seen slice. The third row is the already verified 282 inner-TRAIN panel from different bases and is shown for context, not a paired error comparison.

The observed +0.00338-mm seen-section gain is only about 1.6% of its 0.209-mm no-warp error and far below the preregistered +0.05-mm useful-gain threshold from 282. It falls to +0.00049 mm on new sections from the same bases and becomes slightly negative on new bases. This supports weak fitting plus poor transfer, not a map head that learned useful anatomy-conditioned correction. It does not distinguish insufficient exposure from inadequate image information or learning signal. The 281 coordinate-fit oracle still demonstrates an optimization target exists under the same physical cap, but only when the coordinates themselves are available to fit.

Next run one bounded repeated-image overfit/gradient probe on a few fixed independently sampled TRAIN sections using the same true-plane map path. If the mapper cannot recover a substantial fraction of the coordinate-oracle gain even on those repeated images, fix its gradient, conditioning, or dense supervision before any larger training. If it can, change the inductive bias/supervision and require new-base gains before scaling. No biological, arbitrary-oblique, GUI, electrode-probability, or DeepSlice claim follows from this diagnostic.
