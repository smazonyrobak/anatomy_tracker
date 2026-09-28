# Complete-catalogue mixed curriculum: prepared, not launched

`training/run_joint_v6_proposal_curriculum.py` is a flat experiment draft, not a
trained deliverable. Priority remains the same-row precision-recovery run, which
isolates the corrected FP32, centered proposal head before changing the data.
The curriculum must not run concurrently with that experiment. Only source
parsing/static review has been performed; GPU behavior and learning remain
unverified. The root agent owns the launch decision and any small GPU check.

## Planned intervention

- Fresh random **whole joint model**; no checkpoint loading or legacy learned
  dependencies. Only its proposal branch is optimized at this stage. Capacity is
  F64/H128, proposal embedding 16/eight mixtures, spatial bins 8×8, native 96×96.
- 20,000 updates, batch 16: eight new rendered observations plus eight authenticated
  frozen complex training rows. AdamW starts at 0.001; this is provisional until
  the precision-recovery evidence informs the next experiment.
- Shuffled permutations of all 98,304 catalogue cells cover every one of 384 RP2
  normals, 16 offsets and 16 rolls. The 160,000 planned renders are **observations**,
  not 160,000 distinct planes: all 98,304 canonical cells appear at least once, with
  new appearance/PSF draws when a cell repeats. No subcell continuous jitter yet.
- The pinned Allen intensity/support volume is integrated over nine through-plane
  positions spanning 25–100µm, with normalized trapezoid weights
  `(1,2,2,2,2,2,2,2,1)/16`. Gain, gamma, inversion, noise and synthetic backgrounds
  vary. Exact-black and imperfect-brush modes derive masks from known atlas
  annotation support; raw mode needs no mask. Inputs remain image, mask boundary
  and availability—not a filled-mask channel. Horizontal image flips leave the
  canonical catalogue label unchanged and are recorded as raster representations.
  Appearance transforms act on PSF intensity divided by finite support, then
  restore support mass and mix background before adding noise/masking. This is
  post-PSF conditional-tissue augmentation, not a calibrated acquisition model.
- Every plane is retained. Finite-slab and post-mask visible support below 64
  native pixel-equivalents censor point-pose loss, with per-observation masses
  and weights logged. This is not an uncertainty or abstention training target.

## Provenance and interpretation

Schedules record every catalogue index, observation seed, mode, reflection,
physical thickness, augmentation parameters and unique synthetic
animal/specimen/experiment/section IDs. These IDs are **organizational groups
from one atlas**, not independently acquired or coherently deformed biological
animals. Source hashes, raw-atlas binding, prepared-input hashes, exact cell
states through the authenticated catalogue, and the shared physical PSF rule
allow reconstruction. Frozen row identities/receipts remain intact.

The same 640 development rows/40 synthetic organizational groups stay excluded
from training. Every 1,000 updates retain full raw proposal probabilities,
animal-group macro metrics, support/mode subsets, and a joint checkpoint; no
per-step checkpoint machinery. All run artifacts and caches stay on I:.

This intervention addresses global categorical coverage and observation volume,
not continuous-plane accuracy, consistent 3D subject variation, recurrent joint
alignment/deformation, calibrated electrode probabilities, or deployment.
Complex frozen rows remain half the input samples. Public/DeepSlice benchmarking
and final-test animals remain untouched until internal performance is convincing.
