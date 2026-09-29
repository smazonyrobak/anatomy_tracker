# Prepared canonical rehearsal 001

Source reviewed and ready for a committed launch; no completed training, qualification or benchmark. One flat driver: `training/run_joint_v6_canonical_rehearsal.py`; its launch guard is enabled following root and independent source review.

## Frozen lineage and purpose

Continue the **whole experimental B8000** checkpoint from canonical bridge 001. Its strong real-development improvement did not qualify the model: the no-brush synthetic-capture retention gate failed. This run tests recovery without changing that threshold, selecting development cases, merging encoders or loading another model's learned features.

- Bridge completion SHA-256: `81257a308b0d6e10c70a53de72cafa932fd546c0ec2a6cd13aa82f1edfac3d2f`.
- Independent audit SHA-256: `d53664d3999054fe37c292b9ba080420384dbd499d235562dc4fdbb088462392`.
- Whole B8000 checkpoint SHA-256: `c6aab521d86eee312f12327965b71c6da1d88f184b7e3427dd7ba2782c88c9f8`.
- Original A6000 retention-reference checkpoint SHA-256: `280836b65fb6db22930c8ee268eb4a880997c7858fb91a1e31ad6c7c94937eb2`; only its pinned metric receipts are read, not its weights.

## Fixed continuation

Train 2,000 applied updates, 8000 to 10000. Strictly load B's complete model, deep-copy/restore its complete AdamW state and RNG, then change **only learning rate** from `0.001` to `0.00025`; betas, moments, weight decay `1e-4` and gradient cap five remain unchanged. Use FP32, no AMP/TF32. Train the same histology/atlas stems, shared encoder and descriptor; all nonretrieval tensors remain frozen exactly.

Each batch retains eight real queries and eight synthetic queries, with separately normalized losses combined as `(2/3)*synthetic + (1/3)*real`. Synthetic data remain four generated plus four frozen rows and the unchanged three presentation modes. Use generated schedule `[40000:48000]` and frozen steps `[6000:8000, :4]`. No new mode weighting or development-example selection.

Real queries are unchanged frozen pixels from 58 TRAIN donors. Donor permutation, within-donor row, negative and chart seeds are respectively `2026092921`, `2026092922`, `2026092923`, `2026092924`. Exactly 8,000 of the 16,000 scheduled real presentations use each chart before eligibility weighting. Preserve the exact continuous physical plane and roll, original matched/canonical geometry, shared two-chart support eligibility, physical-plane false-negative mask, invalid-real-key exclusion, fixed 50-µm engineering PSF and fresh learned key features. Synthetic candidate selection and prior arithmetic remain unchanged. Full arbitrary-plane catalogue coverage and optional-brush inputs remain intact.

Archive current source/input/schedule hashes and full identities. The three model files acquiring a separately reviewed **disabled** signed-pose option are explicitly listed as source exceptions; all other non-driver source files must match the bridge snapshot. Rehearsal asserts the option is disabled and strict whole-model parameter equality. No signed evidence, extra encoder or ribbon training is enabled here.

## Final-only evaluation and unchanged thresholds

At step 10000 rebuild the complete 98,304-cell × two-representation bank and evaluate all 256 TRAIN, 640 synthetic development and 64 real development observations. Preserve raw scores, matched/canonical anchor diagnostics, finite-frame RMS diagnostics, exact identities and donor/group aggregation. There are no interim development-based selections.

All gates must pass:

- Real donor-macro normal error no more than **B8000 +2°**, and top-32 physical-plane capture no less than **B8000 −0.02**.
- Synthetic normal error no more than **original A6000 +2°**, and capture no less than **original A6000 −0.02**, overall eligible and separately for every populated eligible presentation mode.
- Real performance must still beat **original A6000** by at least **5°** normal error and **0.10** absolute capture.

This is a single experimental recovery continuation, not an A/B attribution of learning rate versus mixture weight, untouched final validation, calibrated uncertainty, native joint qualification or deployment. No threshold is relaxed because the previous failure was small.
