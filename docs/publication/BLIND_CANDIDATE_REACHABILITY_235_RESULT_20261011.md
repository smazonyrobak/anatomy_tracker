# 235: blind proposals contain near candidates, but the local-flow range is unresolved

The no-training [frozen protocol](BLIND_CANDIDATE_REACHABILITY_235_PROTOCOL_20261011.md) completed on 64 new, independently seeded TRAIN-inner sections (32 v4, 32 v7; virtual bases 60–63). All 64 first attempts were eligible; all 64 physical-section IDs and input hashes were distinct by chance, with no deduplication rule. The frozen 226 batch-1200 checkpoint made every blind 160-plane proposal. Truth was used only for post-proposal scoring, never to form the top-16 beam. No weights were updated or biological holdout/public benchmark read.

| Outcome | v4 (32) | v7 (32) | Combined (64) |
|---|---:|---:|---:|
| First-choice median tissue-site rigid error | 2.766 mm | 3.420 mm | 3.152 mm |
| Truth-best-of-160 median | 1.311 mm | 1.402 mm | 1.387 mm |
| A `<2 mm` candidate in blind top 16 | 26 | 25 | 51 |
| Support-matched `<2 mm` vs `>3 mm` pair in top 16 | 22 | 23 | 45 |
| `<2 mm` top-16 candidate meeting 0.5-mm/32-pixel local reachability | 2 | 1 | 3 |

The predeclared stage-A beam/pair gate passes: 51/64 cases have a near candidate and 45/64 expose at least one same-reflection, support-matched near/wrong pair (322 pairs). This is much more informative than the three support-matched sparse pairs in the older 162 feedback pilot. It **does not** establish that image-based matching can order them.

The local-reachability condition is unexpectedly strict. Across all 156 `<2 mm` top-16 candidates, the median 75th-percentile displacement was 24.1 pixels in plane, but **1.43 mm through plane**; only five candidates had 75% of sites within 0.5 mm of their proposed plane. Thus a small 2-D warp around a fixed plane is unlikely to rescue most apparently “near” candidates. Before treating this as a genuine proposal-angle failure, measure the same through-plane residual for the *true synthetic pose*: the synthetic deformation target itself may include substantial 3-D displacement. Do not train the proposed 32-pixel local updater or relax the threshold based on this readout alone. If the true-pose residual is small, full-pose correction (especially orientation/offset) or wider coarse search must precede local warp fitting; if it is large, normalize the fit-effort definition against that deformation floor.

Independent post-exit checks matched the completion receipt's checkpoint, protocol, audit script, ten dependency-source hashes and all three output hashes; the draw/case files each have 64 records. Frozen output: `I:/AnatomyTracker/runs/blind_candidate_reachability_235_001`; completion SHA-256 `7e73cf7c024143048bd84fd6cff073c87a0d1ea4fe79dc37bc156bc213342454`, case rows SHA-256 `1c081c4870b07af5d16cadfa1e43dafb62bbebacacc0b484a9732c03691f4e4c`. Median visible-tissue fraction was 0.247; the case rows retain each fraction and all 16 candidate metrics. This is synthetic TRAIN-inner evidence, not arbitrary-angle animal accuracy or a deployment gate.
