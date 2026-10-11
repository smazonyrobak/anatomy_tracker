# 172: frozen local-warp fit energy is not reliable pose feedback

The [predeclared TRAIN-only assay](COHERENT_FIT_ENERGY_172_PROTOCOL_20261011.md)
finished in 204 s without new optimizer updates. It froze 240 independent
draw attempts from four synthetic TRAIN deformation bases; 175/192
unconditional and 46/48 deliberately 45–55°-conditioned attempts had usable
tissue. The 132 pose/map model, 171 descriptor matcher and 121 native fitter
were frozen. Synthetic truth was used only after each blind beam and score
was fixed. This is still one atlas with simulated subjects, not independent
biological validation.

On 143 unconditional pairs where a <=1.5-mm branch and a branch >=1 mm worse
had closely matched pre-fit atlas support and at least 64 identical comparison
sites, the lower-error branch won **76/143 (53.1%)** under coherent mapped
descriptor agreement plus displacement/strain cost. Rigid descriptor agreement
won 66/143 (46.2%); a complete zero-atlas-intensity map-and-score rerun won
67/143 (46.9%). Crucially, **zeroing atlas intensity at the final score while
retaining the actual-intensity-induced map won 79/143 (55.2%)**. The actual
intensity score therefore did not outperform this matched score-stage null.
Even among 31 pairs whose best branch was <0.5 mm from truth, it won only 18.
The >=75% and >=10-point pair gate fails; the signal should not teach pose.

Blind 16-way selection on the 175 unconditional nonblank sections also failed:

| Selector | Mean initial rigid error | Capture <=1.5 mm |
| --- | ---: | ---: |
| Direct pose prior | **2.575 mm** | **70/175** |
| Native 121 fitter | 2.621 mm | 68/175 |
| 172 coherent energy alone | 2.669 mm | 71/175 |
| Direct prior + 172 coherent energy | 2.667 mm | 66/175 |
| Direct prior + complete intensity-null energy | 2.718 mm | 65/175 |
| Atlas-support-only | 3.811 mm | 2/175 |
| Truth-chosen best of 16, diagnostic only | 0.939 mm | 156/175 |

The fused coherent score changed 54 decisions versus the direct prior: 22
lowered and 32 raised rigid error. Its selected tissue-map error was 2.638 mm
versus 2.590 mm for the native fitter's selected rigid map. In the separate
46-section steep-oblique stratum, fused error was 3.088 mm versus 2.549 mm for
the direct prior, and capture 16/46 versus 22/46. In 24 sparse (<10% valid
target) cases pooled for descriptive analysis, fused error was 4.331 mm
versus the prior's 3.871 mm. Neither stratum is pooled into the unconditional
gate. The beam oracle's 0.939-mm mean shows substantial selection headroom,
but it uses truth and is not an operational model.

The physical-derivative bound of the selected predicted 132 fields had median
2.96, far above the 0.35 sufficient bound used by the separate constrained
ribbon geometry. This is a *whole-raster conservative bound*, not proof that
the tissue actually folded; it does mean the existing map carries no such
certificate. The scored-site mean squared gradient cost had median 4.54 in
units of (gradient/0.2)^2. A poor pointwise descriptor, background/damage
visibility, synthetic-to-atlas appearance mismatch, inaccurate local fields
and overly coarse pose candidates remain possible contributors. This assay
does not separate them causally or refute coherent fitting as a general idea.

The post-exit audit recomputed the raw unconditional means, captures, pair
wins and decision changes; checked 240 rows, 221 unique scored physical IDs,
all 16 finite candidate scores/choices per scored case and every matched-pair
criterion. All four output hashes, all source hashes, frozen checkpoint and
parent-receipt hashes matched. `completed.json` SHA-256 is
`4dcd23f3c7d3a1443abefc2b8da705b6f06a54cfccbdd151fb5b4c3dbec863ce`.
Frozen output: `I:/AnatomyTracker/runs/coherent_fit_energy_172_train_001`.

**Decision:** do not scale, deploy, or backpropagate this 172 energy into the
pose head; no calibrated posterior, real-animal accuracy, GUI replacement or
DeepSlice comparison follows. The next targeted check should ask whether a
simple *global, contrast-normalized image-versus-rendered-atlas* comparison
can distinguish the same support-matched alternatives. If not, the problem is
deeper than this learned descriptor; if it can, use that evidence to build a
spatially coherent, tissue-aware fit objective inside the eventual joint model
and then test an actual fitting-to-pose gradient before substantial scaling.
