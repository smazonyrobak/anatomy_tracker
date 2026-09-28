"""Descriptive matched-update comparison; synthetic groups are not real animals."""

import os
from pathlib import Path
os.environ["MPLCONFIGDIR"] = r"I:\AnatomyTracker\cache\matplotlib"

import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(r"I:\AnatomyTracker\runs")
runs = [root / "joint_v6_proposal_substantive_001", root / "joint_v6_proposal_capacity_002"]
output = root / "joint_v6_proposal_comparison_001_002"
output.mkdir(exist_ok=False)
summary = {"scope": "internal synthetic-group descriptive comparison, not population inference", "runs": {}}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for run, label in zip(runs, ("16 channels / 32 pixels", "64 channels / 96 pixels")):
    evaluations = [json.loads(path.read_text()) for path in sorted(run.glob("development_metrics_step_*.json"))]
    trace = [json.loads(line) for line in (run / "training_trace.jsonl").read_text().splitlines()]
    steps = [r["step"] for r in evaluations]
    for ax, metric, ylabel in zip(axes, ("nll", "plane_angle_deg", "hit_at_128"),
                                  ("Held-out NLL", "MAP normal error (degrees)", "Nearest-cell top-128 recall")):
        ax.plot(steps, [r["animal_macro"][metric] for r in evaluations], label=label)
        ax.set_xlabel("Optimization step")
        ax.set_ylabel(ylabel)
    final = evaluations[-1]
    config = json.loads((run / "experiment.json").read_text())
    summary["runs"][run.name] = {
        "final_step": final["step"], "final_synthetic_group_macro": final["animal_macro"],
        "final_identifiable_group_macro": final["by_support"]["identifiable"]["animal_macro"],
        "final100_training_nll": float(np.mean([r["weighted_nll"] for r in trace[-100:]])),
        "applied_optimizer_steps": sum(r["optimizer_step_applied"] for r in trace),
        "finite_training_losses": bool(np.isfinite([r["weighted_nll"] for r in trace]).all()),
        "all_elapsed_seconds": final["elapsed_seconds"], "model_kwargs": config["model_kwargs"],
        "config_sha256": hashlib.sha256((run / "experiment.json").read_bytes()).hexdigest(),
        "final_checkpoint_sha256": hashlib.sha256((run / f"joint_model_step_{final['step']:05d}.pt").read_bytes()).hexdigest(),
    }
for ax in axes:
    ax.legend(fontsize=8)
fig.suptitle("Same synthetic rows and training order; both configurations fail coarse capture")
fig.tight_layout()
fig.savefig(output / "comparison.png", dpi=160)
summary["identical_training_row_order"] = bool(np.array_equal(*[np.load(run / "training_row_indices.npy") for run in runs]))
summary["identical_development_identities"] = (runs[0] / "internal_development_identities.json").read_bytes() == (runs[1] / "internal_development_identities.json").read_bytes()
summary["decision"] = "Neither configuration qualifies; increased width/resolution alone did not resolve failure. Investigate late optimization deterioration before scaling compute."
(output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
