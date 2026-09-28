# Proposal baseline 001: completed run, failed development gate

The 10,000-step proposal-only run completed, but localization remains inadequate
for deployment or for opening the ordinary joint-refinement training phase with
the current candidate budget. This is an internal development decision, not a
public benchmark, final test, calibration result, or predefined superiority test.

Run: `I:\AnatomyTracker\runs\joint_v6_proposal_substantive_001`.
Training source commit: `56a09440d6ed71c2043d56e701b48590add84013`.
Fresh whole-model initialization used seed `2026092801`; only the proposal path
was optimized. Configuration: AdamW, learning rate 0.001, batch 16, 32x32 retrieval,
16 feature channels, 32 hidden channels, eight proposal mixtures, 16-dimensional
geometry embeddings, and 4x4 spatial pooling. No earlier model weights, learned
features, pseudolabels, public benchmark or real external data were used.

## Independent CPU audit

Final metrics were recomputed from the complete saved 640x98,304 log-probability
array, frozen physical truth states, catalogue frames, and animal identities.
Ties use the canonical catalogue-index ordering. Sections are averaged within
each animal before averaging across the 40 development animals.

| Metric | Initial saved result | Final independently recomputed result |
| --- | ---: | ---: |
| Catalogue target NLL, nats | 11.495820 | 9.580307304 |
| Projective plane-normal error, degrees | 56.9724 | 50.829943 |
| Absolute normal-offset error, micrometres | 6069.7318 | 2921.138467 |
| Exact target-cell top-1 recall | 0% | 0.15625% |
| Exact target-cell top-8 recall | 0.15625% | 0.78125% |
| Exact target-cell top-32 recall | 0.3125% | 2.34375% |
| Exact target-cell top-128 recall | 0.3125% | 8.90625% |

The final saved metrics agree exactly except for an NLL aggregation difference
of 3.65e-8 nats from floating-point precision. Exact-cell recall is sensitive to
catalogue discretization, but the large physical plane errors independently
support the failed gate. Probabilities remain uncalibrated. This run does not
evaluate a learned joint deformation model or electrode-site localization.

The final 100 training batches have mean weighted NLL 8.738299, while held-out
NLL plateaus around 9.55 after approximately 6,000 updates. The aggregations and
support censoring differ, so this is not a precisely matched generalization-gap
estimate; nevertheless, the learning curves indicate emerging overfitting as
well as poor localization. Limited encoding capacity is a hypothesis for the
next trial, not an established sole cause.

After plotting completed, a separate truth-selected best-of-top-32 diagnostic
reported mean animal-level normal error 27.754845 degrees. This is an oracle
candidate-capture diagnostic, not achieved model accuracy, and it ignores other
pose errors. Even that optimistic normal-only choice remains poor, so failure
is not explained merely by the strict exact-cell recall metric. Its frozen
`top32_capture_diagnostic.json` SHA-256 is
`fe3f65859b87ae71f4d88acfa1c4c8d3df678af2106b8a5d3d12b4fbbe298821`.

- All 10,000 trace rows have consecutive step IDs and applied optimizer steps;
  all recorded losses and gradient norms are finite. The checkpoint optimizer
  states independently record step 10,000.
- All final raw probabilities and checkpoint tensors are finite. Maximum
  absolute full-catalogue log-normalization error is 5.145e-7.
- Step-0 and step-10,000 checkpoint configurations match the saved experiment.
  Exactly 34 tensors changed, all in the histology stem, shared encoder or
  proposal head. Their combined parameter-change L2 norm is 33.88685548.
  The unused atlas/reranking/recurrent/deformation parameters did not change.
- Training contains 5,120 sections from 320 synthetic animals; development has
  640 sections from 40 different animals. Saved identities and row receipts
  match the frozen cache records. Intersections are empty for animal, specimen,
  experiment, synthetic-animal and section IDs.

The audit touched only completed baseline artifacts. It did not access the
active capacity experiment or invoke the GPU. The additional top-32 diagnostic
was read only after the plotting process completed. Existing baseline files
were not changed.

## Artifact receipts

SHA-256 values bind the frozen results used above:

| Artifact | SHA-256 |
| --- | --- |
| `experiment.json` | `90bde7ec17104e340f03aa71913d581a5f9373d50862f0699d4acfba1453be10` |
| `training_trace.jsonl` | `da7089d23012610f072a72e44efdf28c72a44f1ea0a1f50f4cb58c38ef6c7d4d` |
| `training_row_indices.npy` | `9748f19bdbba652687a7198ffaa484912c5d7a54d6eb340ec5dacdc911ff02cb` |
| `development_log_probability_step_00000.npy` | `0cb202c9979f7a8109393f839fca92cf2a3fdcb73b2cbc48d7065d3b5482eadc` |
| `development_log_probability_step_10000.npy` | `70c9dfc9a55617341244d60c603935f283f5dced84d689fcc996e7950765cd1b` |
| `development_metrics_step_10000.json` | `8b187796dd635280538903bc6a9dc6acb04051a81ff5e9f39f0eee6ea165f47e` |
| `joint_model_step_00000.pt` | `b35b73ad1c738de2d73fb9d74b00279328a55c5d5df3cce0eb1fc3b52293ca95` |
| `joint_model_step_10000.pt` | `92b1b4a43f927a2d96249136a706c2d9aa19525c9ed9c45695dd274bdb2f080f` |

Audit script: `I:\AnatomyTracker\tmp\audit_proposal_substantive_001.py`, SHA-256
`0eb4bd503f7b16ddd88ff16f39a2e75ef3bf6a791b422c542fec7dfcaa12fc3f`.
Full audit output, including additional catalogue/identity/prepared-data hashes:
`I:\AnatomyTracker\tmp\audit_proposal_substantive_001.json`, SHA-256
`f083c419cab076519ce687dfcca7505f7b01614370634ac2565b2235d1f1809e`.

The next experiment enlarges spatial encoding capacity while preserving this
baseline, its data split and its objective. Improved NLL alone will not open the
joint-training gate; candidate capture and physical localization must improve.
