# Predicted-coordinate recurrent evidence — 2026-09-29

Status: opt-in implementation and one CPU geometry/compatibility check only. No
training driver enables this option yet. No accuracy, calibration, global-capture,
or deployment claim follows from this change.

## Narrow architectural hypothesis

The existing spatial GRU is followed by spatial mean pooling and a nine-component
canonical-frame update. An affine residual field contains position-dependent
terms that can cancel under a spatial mean. Convolutions, image boundaries and
anatomical appearance can still encode position implicitly, so this is a possible
learning bottleneck, not proof of the cause of the local pose failure. Reflection
also changes the relationship between observed-raster displacement and canonical
frame correction; the previous implementation is not thereby a demonstrated sign
bug.

Add known, predicted geometry before the nonlinear GRU instead of replacing its
readout: [CoordConv](https://arxiv.org/pdf/1807.03247) motivates explicit coordinate
channels for position-dependent mappings. [RAFT](https://arxiv.org/pdf/2003.12039)
conditions recurrent updates on current flow, correlation and context; its dense
update head is different from our pooled full-frame head. These papers motivate
the control, not its expected numerical success on histology.

## Exact implementation

`coordinate_evidence_conditioning=False` is propagated through the existing joint,
v6 pose and base recurrent constructors. When enabled, a bias-free zero-initialized
`Conv2d(7, hidden_channels, 1)` is added under an isolated CPU RNG fork. With the
current 128-channel GRU this adds 896 parameters, solely under
`pose_model.coordinate_evidence.weight`. Disabled construction creates no new
state-dictionary entry or random draw.

At every update and at the final scoring pass, seven channels are projected and
added to pair evidence immediately before the GRU:

1. Predicted CCF AP/DV/ML coordinates, minus the fixed support origin, divided by
   10,000 micrometres. No per-slice centering removes the plane offset.
2. Observed raster x/y in `align_corners=False` normalized coordinates.
3. The two known representation reflection signs.

The composition follows the actual image-rendering path:

`observed pixel -> current renderer feedback map -> representation reflection -> physical OUV`.

The loop currently supplies the decoder's `forward_map_yx_px` as its sampling map;
the helper consumes that exact `feedback_map`, not a map selected by interpreting
its name. Convert mapped y/x to x/y, reflect finite-raster coordinates with
`x_c = s_x*x + (1-s_x)*(W-1)/2` (and similarly y), then use
`p = O + x_c/W*U + y_c/H*V`. The field is the symmetric PSF's midplane/untruncated
first moment; the existing finite-thickness integration remains unchanged.
Out-of-canvas queries retain their extrapolated physical coordinates while the
image sampler still zero-pads. No truth, support mask or surgical measurement is
an input to this hook.

The source/atlas encoder centres lie at original raster indices `0,4,8,...`, not
the `1.5,5.5,...` centres of an `align_corners=False` 96-to-24 resize. The helper
therefore samples the current full-resolution map at `[..., ::4, ::4]`. Cell and
representation contexts remain separate at this injection; existing later
averaging and the common per-cell deterministic pose/deformation remain unchanged.
The separate optional centre-offset observation hook is unaffected.

## Focused completed CPU check

One script used two rows of the already-completed frozen training pack and the
completed proposal catalogue. Frozen states/maps were numerical fixtures only;
the model helper receives current predictions. No GPU, optimizer, live run tree,
or atlas decoding was used. This is not a complete-refinement performance test.

- Against source commit `5d6a4a2`, default base state keys, all parameters and RNG
  state were exact. Enabling the option in the full joint model preserved all old
  parameters and RNG state; only the 896-parameter projection was new.
- Initial source features and zero-conditioned pair evidence were bitwise equal.
  Both a supplied map and the no-map path were checked.
- For 2,264 in-bounds queries, the maximum physical-coordinate discrepancy against
  the existing CCF/reflection/map samplers was **0.0005642573 micrometres**.
- Projection-gradient L1 through the existing GRU and nine-component readout was
  **0.0008040259**, finite and nonzero. Coordinate-field gradients through state
  and map were finite/nonzero (L1 **0.07568654** and **0.002211276**). At zero
  projection, its additional route to state/map gradients is initially zero by
  construction; learning the projection opens that route.

Source: [CPU check](I:/AnatomyTracker/tmp/check_coordinate_evidence_20260929.py),
SHA-256 `d54deefe93de09b59891673dc30174e8b3736cb4864226299cd9980e62f6f40e`.
Result: [receipt](I:/AnatomyTracker/tmp/check_coordinate_evidence_20260929.json),
SHA-256 `468bb66141798072dbf03969ab77029d2579616c5bf760054d5a59165c637bbc`.
The receipt records the three implementation-source hashes.

A bounded learning control must still demonstrate genuine full-frame and dense
mapping improvement. Better reflection classification alone does not qualify
joint alignment/deformation, and this local hook does not solve global capture
or establish calibrated uncertainty.
