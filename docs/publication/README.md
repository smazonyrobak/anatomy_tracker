# Unified registration publication dossier

This directory records the preregistered rationale, methods and evidence plan for the proposed joint atlas-pose and nonlinear-registration model. It is documentation, not evidence that the proposed model has been implemented, trained or qualified.

The frozen software baseline is Git commit `c6681039e0b7acf35c9cdbee43040a3dca29cdab`. Machine-readable protocol, gate and comparator definitions live in `../../publication/`.

- [`STUDY_PROTOCOL.md`](STUDY_PROTOCOL.md): claims, endpoints, splits, statistics and decision rules.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): proposed recurrent pose/registration model and rationale.
- [`COORDINATE_CONVENTIONS.md`](COORDINATE_CONVENTIONS.md): public coordinates and transform direction.
- [`DATA_AND_SYNTHETIC_GENERATOR.md`](DATA_AND_SYNTHETIC_GENERATOR.md): provenance, generation and leakage controls.
- [`TRAINING_PROTOCOL.md`](TRAINING_PROTOCOL.md): staged training and experiment controls.
- [`CONSTRAINTS.md`](CONSTRAINTS.md): deterministic AP, order, common-tilt and surgical constraints.
- [`BENCHMARKS.md`](BENCHMARKS.md): comparators, metrics, statistics, ablations and runtime.
- [`FUTURE_VALIDATION_PLAN.md`](FUTURE_VALIDATION_PLAN.md): deferred animal-level external validation and calibrated uncertainty.
- [`BASELINE_LEDGER.md`](BASELINE_LEDGER.md): immutable statement of evidence available at the baseline commit.

Machine-readable study, cold-start architecture-screen, data, comparator,
release-gate and exact synthetic-test contracts live in
[`../../publication`](../../publication).
