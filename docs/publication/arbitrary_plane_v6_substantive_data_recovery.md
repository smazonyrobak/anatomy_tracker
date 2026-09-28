# Substantive synthetic data recovery — 2026-09-28

The previous generator completed. Its stdout contains the final summary and its
stderr is empty. The saved summary records a completed replay of every row through
the independent v6 finite-row loader; regeneration is unnecessary.

| Partition | Sections | Synthetic animals | Frozen manifest receipt |
| --- | ---: | ---: | --- |
| Training | 5,120 | 320 | `af778499dd77ac13f673ce0778972218519f421b97fa01c4d055e00d1dd16266` |
| Internal development | 640 | 40 | `22d6ae273f13974250d74a2f5b32446a49a1c2c3051d40ad45706300aa1159de` |

Both caches are under
`I:\AnatomyTracker\runs\arbitrary_plane_finite_v6_substantive_data_001`, in
`training_cache` and `internal_development_cache` respectively. They contain
96×96 images, nine through-plane samples, and all three smart-brush conditions.
Training contains 3,072 pose and 2,048 joint rows; development contains 384 pose
and 256 joint rows. Each synthetic animal contributes 16 sections.

**Interpretation correction:** these "synthetic animals" are organizational
16-row provenance groups, not distinct consistent 3D subject anatomies. Plane,
section deformation and appearance streams are sampled per row from one Allen
atlas. Empty ID intersections therefore establish separate generated records,
not biological animal-level generalization. See
`SYNTHETIC_GROUPING_CORRECTION_20260928.md`; frozen data and receipts are unchanged.

Recovery verification recomputed the summary receipt, authenticated both frozen
manifests using `load_frozen_row_cache_manifest_v6`, and independently recomputed
cross-partition intersections for animal, specimen, experiment, synthetic-animal
and section IDs. All five intersections were empty. Dense arrays were not replayed
a second time because the completed generation already recorded full replay.

- Summary: `substantive_finite_data_summary_v6.json`
- Summary receipt: `20d5963c4432d92cedecd0b194eb624861ffe0f916fb7fb92dce2b1e4db2a65e`
- Summary file SHA-256: `7c62a53256d24786f2f8b572acbc6566862cc3072441259b36d7acefbaafcc01`
- Generation commit: `9803ec89eae7ee8cbdd293325b0094171e2112ec`
- Launcher SHA-256: `50eea07d73b18abb4cec48b4b2776e0fe91372335cd576765fbbba7c4d951c97`
- No prior learned weights/features/pseudolabels; no public benchmark, external
  validation or final-test data accessed.

The pending runner optimization was reviewed and four focused tests passed in
7.77 seconds: incremental warm replay, rejection of a corrupted new transaction
before publication, cold rejection of corrupted historical output, and recovery
of an already completed transaction. Warm updates now replay only the appended
transaction; cold loading still replays all committed artifacts.

This improves the existing runner but does not make it suitable for unlimited
step counts: it still serializes and verifies the growing checkpoint ledger each
step, and row loading authenticates the full cache manifest each batch. The next
training experiment should load authenticated data once and use periodic
checkpoints with exact sampled IDs and raw metrics retained. Begin with fresh
random initialization and a substantive proposal-training phase around learning
rate 0.001, then use disjoint synthetic development groups to decide when to introduce
rendering/refinement and joint deformation. The four-row capacity diagnostic is
evidence that optimization can work, not evidence of generalization.
