# 130: crossed evidence did not establish anatomical plane selection

Both predeclared 2,000-batch arms completed. Each saw 4,000 independently
sampled physical TRAIN sections with v4 appearance; accepted draw records and
all source/configuration fields except arm were identical. The full arm saw
atlas intensity, support and coordinates; the matched control saw support and
coordinates with intensity zeroed. Both continued the same randomly
initialized proprietary 128 lineage. Completion receipts, all frozen
checkpoints, training logs, evaluator source, and raw DEV output hashes passed
the post-exit audit. There was no external model, pseudolabel, calibration or
public benchmark input.

The original-branch evaluation reused 243 eligible sections on eight held-out
synthetic deformation plans from the frozen 128 panel, including 151
support-matched near/wrong *original-branch* pairs. It compared the same
candidate identities and frozen tissue-to-atlas mappings across arms. These
are not physical animals or fresh biological test cases.

| Batch | Atlas-intensity selected mapping error | Intensity-zeroed error | Atlas-intensity near capture | Intensity-zeroed capture | Matched-pair wins, full / control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 3.221 mm | 3.185 mm | 31.3% | 31.8% | 69.5% / 66.9% |
| 1,000 | 3.154 mm | 3.262 mm | 33.4% | 31.3% | 70.9% / 69.5% |
| 2,000 | **3.140 mm** | **3.136 mm** | **34.3%** | **34.0%** | **66.2% / 64.9%** |

The common direct-prior baseline on the same beam was 3.300 mm selected
mapping error and 26.3% within 1.5 mm. Thus the final full arm improved by
only 0.160 mm and 8.0 percentage points against that prior, below the
predeclared 0.30-mm and ten-point localization gates. More importantly, it
was 0.004 mm *worse* than the matched intensity-zeroed arm. The largest
conditional full-arm pair-win rate, 70.9% at batch 1,000, missed the 75% gate
and exceeded its same-batch control by only 1.3 points, not ten. At batch
2,000, zeroing atlas intensity within the full arm reduced pair wins by 6.6
points and swapping source features by 3.3 points, both below the ten-point
ablation gate. Across the eight plans, the full final arm improved over the
control on four and worsened on four. Raw-input error was 3.229 versus 3.219
mm (175 sections), exact-black 3.021 versus 2.904 (39), and imperfect-brush
2.839 versus 3.031 (29); these unequal small subgroups are descriptive only.

The crossed training interaction increased in both arms: final-500-batch
means were 15.30 (full) and 15.59 (intensity zeroed). The exact-plane
contrast can therefore be learned without atlas intensity in this design.
Only 557/4,000 training draws per arm had a candidate within the training
script's strict five-point-plus-normal 1.5-mm criterion, yielding 549
support-matched hard pairs. That criterion is **not** the DEV panel's
valid-tissue mapped-error criterion; its 13.9% fraction must not be compared
as though it were the DEV beam's ~80% mapped capture. A matched-metric pose
capture diagnosis on independent v4 sections is needed before assigning the
cause to exposure, candidate generation or plane sampling.

**Decision: fail the 130 gate.** Do not feed this evidence head into the
direct probabilistic pose head, scale this exact crossed loss, deploy it in
the GUI, or claim improved anatomy-based fitting feedback. The control still
receives support and coordinates and both arms inherit an intensity-trained
matcher; source-row thickness also prevents exact cancellation of all atlas
terms in the 2×2 algebra. Those limitations make a positive result harder to
interpret, but they do not rescue the absent intensity advantage. Because
the gate failed, no fresh v4 DEV, weak-real, shared-PSF, physical-oblique,
uncertainty-calibration or DeepSlice benchmark promotion was performed.

Next, measure frozen direct-pose beam capture on independently drawn v3 and
v4 sections under the *same* valid-tissue physical error, stratified by
exposure, mode and cutting angle. If v4 capture deteriorates, target the
appearance/domain gap and direct pose head before another selector. Separately
review tissue texture, exterior variability and complex damage in the
generator; TRAIN uses 64 distinct local deformation maps (not eight), while
synthetic DEV still relies on eight maps and one Allen anatomy template.

Frozen outputs: `I:/AnatomyTracker/runs/crossed_plane_evidence_130_full`,
`I:/AnatomyTracker/runs/crossed_plane_evidence_130_support`, and
`I:/AnatomyTracker/runs/crossed_plane_evidence_130_dev_eval`. Completion
receipt SHA-256 values are `64441ebda174d88043081824fcbfe81fcadbfead8a4edd514b069a75cac696db`,
`aa79cc515890ba1573fb17598e58e1d1620b217b97d35409b9cb63ffa6ce385b`,
and `2544178eb04d8b9a2c627d30c655634f289e0441c9d4acfb0a161765a949c3b4`,
respectively.
