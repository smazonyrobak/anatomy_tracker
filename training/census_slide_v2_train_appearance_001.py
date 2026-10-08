"""TRAIN-only input-appearance census of independently drawn v2 synthetic planes."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(r"I:\AnatomyTracker")
os.environ["TEMP"] = os.environ["TMP"] = str(root / "tmp")
os.environ["TORCH_HOME"] = str(root / "cache/torch")
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_one_shot_slide_artifacts_v2 import sample_one_shot_slide_artifacts_v2
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64


out = root / "runs/slide_v2_train_appearance_census_001"
seed, target = 2026100906, 256
rng = np.random.default_rng(seed)
context = load_streaming_synthetic_v7_64(device="cuda")
out.mkdir(parents=True, exist_ok=False)
rows, attempts = [], []
draw_seed = seed + 1
with torch.inference_mode():
    while len(rows) < target:
        virtual = rng.integers(len(context["subjects"]), size=8).tolist()
        sample = sample_one_shot_slide_artifacts_v2(context, virtual, draw_seed, side=256)
        image = sample["inputs"][:, 0].cpu().numpy()
        available = sample["inputs"][:, 2, 0, 0].cpu().numpy()
        for i, meta in enumerate(sample["provenance"]):
            used = bool(sample["eligible"][i]) and len(rows) < target
            attempts.append({"draw_seed": draw_seed, "row": i,
                             "physical_section_id": meta["physical_section_id"], "used": used})
            if not used:
                continue
            pixels = image[i]
            border = np.concatenate((pixels[0], pixels[-1], pixels[1:-1, 0], pixels[1:-1, -1]))
            rows.append({"physical_section_id": meta["physical_section_id"],
                         "virtual_subject_id": meta["virtual_subject_id"],
                         "mode": meta["mode"], "outline_available": bool(available[i] > .5),
                         "events": meta["one_shot_slide_artifacts_v2"]["events"],
                         "raw_exterior_code": meta["one_shot_slide_artifacts_v2"]["parameters"]["raw_exterior_code"],
                         "valid_fraction": float(sample["valid_mask"][i].float().mean()),
                         "exact_zero_fraction": float((pixels == 0).mean()),
                         "near_zero_fraction_below_0_03": float((pixels < .03).mean()),
                         "border_fraction_above_0_03": float((border > .03).mean()),
                         "saturated_fraction_at_least_0_99": float((pixels >= .99).mean()),
                         "q50": float(np.quantile(pixels, .5)),
                         "q95": float(np.quantile(pixels, .95))})
        draw_seed += 1

with (out / "rows.jsonl").open("w", encoding="utf-8") as stream:
    for row in rows:
        stream.write(json.dumps(row) + "\n")
with (out / "attempts.jsonl").open("w", encoding="utf-8") as stream:
    for row in attempts:
        stream.write(json.dumps(row) + "\n")
names = ("valid_fraction", "exact_zero_fraction", "near_zero_fraction_below_0_03",
         "border_fraction_above_0_03", "saturated_fraction_at_least_0_99", "q50", "q95")
groups = {"all": rows, **{mode: [row for row in rows if row["mode"] == mode]
                         for mode in ("raw", "exact_black", "imperfect_brush")}}
summary = {"version": "slide-v2-train-appearance-census-001", "seed": seed,
           "scope": "256 independently drawn eligible full-sphere TRAIN synthetic sections; whole canvas, not tissue-segmented; one atlas with virtual subject deformations",
           "attempts": len(attempts), "eligible": len(rows),
           "groups": {name: {"n": len(group), "outline_available": sum(row["outline_available"] for row in group),
                             **{metric: float(np.mean([row[metric] for row in group])) for metric in names}}
                      for name, group in groups.items()},
           "event_counts": {event: sum(row["events"][event] for row in rows)
                            for event in ("fragment", "fold", "bubble", "tile_seam")},
           "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                             for name in ("census_slide_v2_train_appearance_001.py",
                                          "arbitrary_plane_one_shot_slide_artifacts_v2.py",
                                          "arbitrary_plane_streaming_synthetic_v7.py",
                                          "arbitrary_plane_streaming_synthetic_v7_64.py")}}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
(out / "completed.json").write_text(json.dumps({"attempts": len(attempts), "eligible": len(rows),
    "output_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                      for name in ("rows.jsonl", "attempts.jsonl", "summary.json")}}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "groups": summary["groups"],
                  "event_counts": summary["event_counts"]}, indent=2), flush=True)
