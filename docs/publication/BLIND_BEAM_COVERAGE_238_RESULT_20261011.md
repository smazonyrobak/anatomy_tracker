# 238: the frozen proposer, not just the top-16 beam, misses some planes

The no-training replay of the exact 64 TRAIN-inner sections from 235 completed normally. All 64 image hashes and physical-section IDs matched 235, and the old top-16 indices/errors reproduced. An independent read-only audit checked 64 distinct IDs, source/checkpoint/protocol/output hashes and every reported beam-coverage count. No reserved animal or public benchmark was used.

| Prior-ordered beam | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sections with a `<2 mm` proposal, all 64 | 19 | 31 | 36 | 47 | 51 | 53 | 58 | 58 |
| v4, 32 sections | 11 | 18 | 20 | 23 | 26 | 26 | 30 | 30 |
| v7, 32 sections | 8 | 13 | 16 | 24 | 25 | 27 | 28 | 28 |
| Median truth-best tissue error (mm), all | 3.152 | 2.101 | 1.865 | 1.666 | 1.521 | 1.500 | 1.422 | 1.387 |

Seven of the 13 top-16 misses have a `<2 mm` proposal at prior ranks 18–64, so a larger beam can recover those candidates. The other **six of 64 have no such proposal anywhere among all 160** (two v4, four v7); their best errors range from 2.25 to 3.15 mm. A 64-branch beam captures every near proposal that the 160-branch set can offer on this panel. The 16 free-pose modes produce a near proposal in 55/64 cases, the 64 normal-anchor modes in 49/64, with 46 cases overlapping; both families contribute some unique coverage. Misses include sections with moderate visible-tissue fractions, so low tissue support is not a complete explanation.

**Decision:** the protocol's 60/64 all-proposal criterion is missed (58/64). Increasing the beam is a bounded aid to candidate capture, not the primary fix, and it multiplies the cost of the currently unreliable 237 anatomy matcher. The current selected first choice has a `<2 mm` proposal in only 19/64 cases despite 58/64 being available somewhere in the full set, so both proposal coverage and anatomy-informed selection remain open. Do not train or deploy unchanged. The next architecture/data step needs to improve candidate representation/capture *and* use trustworthy candidate-specific anatomical evidence with no true-pose drift; test that on new TRAIN-inner physical draws before animal-separated qualification. These replayed cases are diagnostic, not unbiased evidence for a tuned successor.

Frozen output: `I:/AnatomyTracker/runs/blind_beam_coverage_238_001`. SHA-256: `completed.json` `a7b1bd69b55fd625906f9d40d2dc149452ade5670434bad20c22f52811427d66`; `summary.json` `b91362ce3ebe4cab01aa664e1d62c12c93d845d92adf1d8f8fe2986df7aa7775`; `rows.jsonl` `511c809651b63f0dc1145bd9482066d17078b61e5c89d63f0eea55a7c2cce598`.
