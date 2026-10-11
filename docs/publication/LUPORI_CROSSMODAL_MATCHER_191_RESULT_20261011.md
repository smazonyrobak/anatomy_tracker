# 191: a small real-trained atlas comparator did not improve animal-held-out ranking

The [paired TRAIN-only pilot](LUPORI_CROSSMODAL_MATCHER_191_INNER_PROTOCOL_20261011.md) completed 600 matched batches on 188 sections from `CC2B,CC4B,CC4A`. It froze the 187 step-800 pose, image encoder and deformation mapper, and updated only two independent copies of its spatial global plane matcher. The `full` arm received Allen atlas intensity plus support; the `support_only` arm received the same support and geometry but zero atlas intensity. They saw the same blind candidate IDs and approximate human-plane supervision. The `AL1A` inner TRAIN-animal guard remained absent from gradients. No DEV/CAL/FINAL animal or public benchmark was read.

On all **52** valid `AL1A` sections, the fixed step-600 readout failed its predeclared gate:

| selector | five-site mean / median against approximate human plane | ≤1.5 mm | >5 mm |
|---|---:|---:|---:|
| unchanged direct pose prior | **1.904 / 1.344 mm** | 28/52 | **2/52** |
| learned full atlas matcher | 1.986 / 1.373 mm | 28/52 | 3/52 |
| matched support-only matcher | 1.959 / **1.323 mm** | **31/52** | 4/52 |

The full arm changed 30/52 prior choices, improving 15 and worsening 15; support-only changed 34/52, improving 19 and worsening 15. Both training losses decreased over the 600 batches (first→last 100: full 2.043→1.669, support-only 2.023→1.601), so optimization occurred, but that did **not** produce a useful held-animal anatomy-specific ranking gain. The blind top-16 included a ≤1.5-mm branch for 37/52 sections. This is an inner TRAIN-animal result, not an independent final test, and the labels are approximate registrations.

The first launch at `I:/AnatomyTracker/runs/lupori_crossmodal_matcher_191_inner_AL1A` exited before its first optimizer update: cloning a frozen parent left both matcher copies with gradients disabled. It is preserved as a failed technical attempt. The committed correction enabled gradients in the copies and used a new `..._retry1` run; the read-only verifier confirmed all 600 finite-gradient updates, exactly the three allowed gradient animals, 188 distinct source sections, 0/300/600 checkpoint hashes, source image hashes, and all 52 guard rows and arithmetic. Only `..._retry1` is scientific evidence.

**Decision:** do not promote this 191 comparator or use its score as pose feedback. Together 189–191 show a local atlas-intensity correction direction but no robust blind selector. Adding more epochs to the same three-animal weak-label set is not justified by the support-only control. The next step must improve the *evidence*, not merely the optimizer: determine whether acquired fluorescence provides reliable spatially matched landmarks against the Allen template, and whether human displacement fields can supervise a constrained local map without allowing the map to absorb global pose. A larger, specimen-diverse acquired training set and independent steep-oblique sections are still needed before arbitrary-plane and electrode-region claims. The direct 188 pose checkpoint remains the stronger starting point but is not a shipped joint model.

Frozen `I:/AnatomyTracker/runs/lupori_crossmodal_matcher_191_inner_AL1A_retry1/completed.json` SHA-256 is `ee151bca0e587c0ab1a4d06ccde20b553949c2b835b89bc191cca5d4a77d8649`; frozen guard completion SHA-256 is `23dffab20b094afe5fb6e85813636213c2b8b1f68d823ba04bdd9b9251445b37`.
