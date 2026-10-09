# 127: the wrong plane is usually selected, not merely pruned

The frozen diagnostic ran the 122 step-2000 model once on each of the 241 eligible physical sections in the completed 125 synthetic DEV panel. Every all-160 original-branch mapped and rigid error, prior score/rank, current 16-beam ID, corrected-beam error and selected action is preserved per section. The run made no weight or frozen-input changes. Its evaluator reproduced the 125 selected IDs and errors section by section; the output hashes match its receipt. These are fresh cuts on eight **reused synthetic deformation plans**, not independent biological animals.

| Available choice | Mean best mapped error | Sections with a ≤1.5-mm mapped plane |
| --- | ---: | ---: |
| Prior top 16, original | 1.289 mm | 174/241 |
| Prior top 32, original | 1.214 mm | 179/241 |
| Prior top 64, original | 1.172 mm | 183/241 |
| All 160, original | **1.126 mm** | **188/241** |
| Current blind 16, original | 1.356 mm | 170/241 |
| Current blind 16, corrected | 1.357 mm | 170/241 |
| Actual selected original/corrected action | **3.364 mm** | **67/241** |

All 160 rescue only 18/241 (7.47 percentage points) over the current blind 16 original branches, below the predeclared 10-point threshold for a material *beam-pruning* bottleneck. The all-160/beam-16 best-error difference of 0.230 mm is real, but the selected-versus-best-beam gap is about **2.01 mm**. The selected branch ID equals the direct prior's top-1 branch on **all 241 sections**; every top-1 branch is among the 16 unconstrained base modes, never an anchor. The prior's top two already contain a ≤1.5-mm plane in 111/241 sections and its top three in 137/241, versus only 67/241 actually selected. Thus stronger discrimination among existing plausible planes is the immediate priority. The frozen global matcher did not change the selected branch on this panel.

The raw/no-brush subgroup had a near original blind-beam plane in 99/149 sections but selected one in only 36; exact-black had 41/50 available and 21 selected; imperfect-brush had 30/42 available and 10 selected. The 45–55° nearest-cardinal-angle subgroup had 20/24 near candidates in both the beam and all 160, yet selected only eight. These unequal, unpaired strata diagnose poor selection and possible raw-input vulnerability; they do not establish a causal background effect or prove reliable acquired steep-oblique localization.

**Decision:** do not spend the next substantial run merely increasing beam size. Train a selection mechanism within the same deployment-intended pose/atlas/map architecture and compare it against a same-draw continuation, while retaining direct proposal and real-donor guardrails. Fitting-to-coordinate feedback remains unproven because 125 failed; this diagnostic does not authorize using its score. No calibrated uncertainty, expert physical truth, GUI replacement or DeepSlice result follows.

Frozen output: `I:/AnatomyTracker/runs/joint_pose_capture_127_diag`. SHA-256: rows `30859fcd34dded419f548c17c478e8cc61cd7392f2f073205718d597575773d1`, summary `ae17d480b71e33ba142f47133fc25950c9e584f6da72942f005d0844d408816d`, evaluator `1eab211893a096b476b147aca6a24f66d98216678b496bc0f6fd222dbca84fa2`; parent 122 checkpoint `07dc707bcb56fd75c7bfb05c5a37e93ce5d8460c6b5a85cc27a39ae8b3aaee75`.
