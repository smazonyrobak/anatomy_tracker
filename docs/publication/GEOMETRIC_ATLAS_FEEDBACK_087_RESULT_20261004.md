# 087 frozen geometric-feedback readout (4 October 2026)

The fixed correspondence-to-rigid-pose update **failed**. On the same 246 held-out arbitrary-plane synthetic sections and same blind 14-branch beam, it worsened both candidate capture and selected alignment. There is no basis to add this untrained solve to the deployed model or tune its gain on this development panel.

| Eight-plan-equal error | Frozen 085 | 087 fixed geometry |
| --- | ---: | ---: |
| Best of the same 14, rigid 3-D tissue placement | 1.022 mm | 1.254 mm |
| Best of the same 14, mapped 96-grid tissue | 0.992 mm | 1.234 mm |
| Score-selected mapped 96-grid tissue | 2.587 mm | 2.776 mm |
| Score-selected mapped 256-grid tissue | 2.591 mm | 2.777 mm |

The selected 256-grid result is also worse than parent 059's matched 2.535 mm. Seven of eight held-out synthetic deformation plans worsened versus 085; 100 individual sections improved and 146 worsened. The selected 256-grid error worsened for exact-black (2.429→2.691 mm), raw-background (2.721→2.880 mm), and imperfect-brush sections (2.510→2.644 mm). The rigid best-of-14 and mapped best-of-14 regressions are similar in size, so this is not solely a frozen mapper reacting poorly to a better plane. The frozen score chose a different branch for 52/246 sections, but selection error alone cannot explain the worse truth-best-14 result.

The 087 head took the already-trained 083 local match posterior, converted its per-pixel mean into a confidence-weighted, damped six-DOF least-squares pose correction, re-rendered, and repeated at finer resolution. It trained no weights. The readout used the frozen 085 batch-1,000 model, identical 086 branch IDs in original beam order, identical 1024 surviving pixels for all-candidate mapped96 comparisons, and all surviving pixels for selected mapped256. No true branch was inserted. An independent raw-row read confirmed 246 unique sections/eight plans, each with 14 scores and errors, selected-branch argmax and best-14 minima, and reproduced the plan-equal means and appearance strata. Config/row/summary, both new source files and the preregistered protocol hashes all matched the completion receipt or recorded config.

This result says that **posterior averaging plus a fixed rigid solve is not a reliable plane correction with the present 083 descriptors**. It does not prove that spatial correspondence or trained geometric feedback cannot work. The next model change should learn a geometry-consistent fitting signal jointly with the direct probabilistic pose head, anchor it to known synthetic pose/map targets, and avoid the oversized native-fit gradient that undermined the earlier 008 joint training. Merely extending 085's scalar ranker or increasing 087's solve gain is not evidence-led. This remains synthetic development, not expert real-animal validation, calibrated uncertainty, a GUI replacement, or a DeepSlice comparison.

Frozen output: `I:/AnatomyTracker/runs/geometric_feedback_087_development_eval`; completed receipt config SHA-256 `0b697d1785e159abd1d7ac0250d4bfad616aaadfe5e515015f6a6d84d6f9b6d7`, rows `217c6124308c1d8738dafca55252755524e0dbe1e7761357691da1f393c25669`, summary `dbe4f5be5b28d1478fdbc92676aa5324240271d18752029fd2ba034fcd7782b7`. See the [frozen protocol](GEOMETRIC_ATLAS_FEEDBACK_087_PROTOCOL_20261004.md), [086 ranking diagnosis](ALLBEAM_CANDIDATE_RANK_DIAGNOSTIC_086_RESULT_20261004.md), and [008 joint native-fit result](ONE_SHOT_JOINT_NATIVE_FIT_008_READOUT_20261002.md).
