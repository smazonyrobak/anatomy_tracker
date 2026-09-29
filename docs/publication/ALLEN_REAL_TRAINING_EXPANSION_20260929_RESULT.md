# Allen real TRAIN expansion — frozen CPU audit

**Integrity passed.** The completed acquisition adds exactly 1,024 prespecified
sections to the original 256, linking a 1,280-section union without copying or
changing the original array. These remain **58 TRAIN donors**, not 1,024 new
animals. The six development donors (14452, 15219, 15336, 15439, 15447, 15935)
are excluded. Thirty-eight TRAIN donors contribute 18 additional sections each;
20 contribute 17. No selection retry or replacement was introduced.

The independent CPU audit reconstructs the donor-round-robin selection from
frozen metadata, authenticates all 1,280 raw JPEG files and their IDs, and checks
every source/output hash, planned new row, and union link. The original 256
metadata records, row order, source array indices and image paths match exactly.
The 1,024 new downloads total **82,656,170 bytes**, with zero failed downloads.

The new array is finite `float32[1024,1,96,96]`, range
**[0, 0.9992483258247375]**. Independently reconstructed full-resolution,
downloaded-pyramid and AP/DV/ML affine matrices agree exactly for all 1,280 rows
(maximum absolute difference 0). The audit also checks the saved frame centre,
normal, native coverage and antialias scale. The existing convention remains:
downloaded pixel centres satisfy `full=(downloaded+.5)*32-.5`, model coordinates
use x/96 and y/96, and **12.5 um is added once** to each atlas-coordinate origin
component. No coordinate convention or image normalization was tuned.

This authenticates the unchanged preprocessing source and its saved products;
it does not replay red-channel filtering/interpolation. No network requests,
atlas rerender, model inference, GPU work, training, or benchmark occurred in
the audit. These are acquired appearances and weak upstream-affine references,
not dense deformation or arbitrary-plane ground truth. The historical-exposure
caveat remains: no complete joinable historical benchmark exclusion list was
available. Donor-disjoint development is not an untouched final benchmark claim.

## Frozen receipts

Data: `I:/AnatomyTracker/data/allen_real_training_expansion_20260929`.
Audit: `I:/AnatomyTracker/runs/allen_real_training_expansion_20260929_independent_audit`;
`audit_source.py` is the flat executable source and `audit.json` retains every
authenticated raw-image/source/artifact SHA256. Execution exited 0.

- Acquisition `summary.json`:
  `dfe59bde72f55ade32c8b2b6d144fb4ce0064d9fa27bd74c98dd16362def3334`.
- Independent `audit.json`:
  `454eb800f058f77b0f45a8302b06e7caed11b4888a8f786789f28b16f97c2631`.
- Independent `audit_source.py`:
  `c42877c44f03c09747927a8cf34e9270a8b543202a64eb6466328eb1bff81806`.
- New `raw_model_input.npy`:
  `e539be1410eb2868a606e0bc3823262a93cee1120a282fd8c2063da19ae386b2`.
- New `image_geometry.jsonl`:
  `41002b7685378ccd77e7a93f468a915aa9d9beb98af3e55823d94b1b130d4f40`.
- `union_training_index.jsonl`:
  `361f0bd8e1381033205ca6db08d30b3e31cb8a53472a63f3ae9086e99c2a0d8f`.

This is a data-readiness result, not evidence that the larger corpus improves
the model or qualifies it for deployment.
