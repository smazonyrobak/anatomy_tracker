import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_propagation_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)

COORDINATE_COLUMNS = ("ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz")


def test_shared_angle_propagation_matches_official_deepslice_exactly():
    inputs = np.load(ROOT / "tests" / "data" / "deepslice_shared_angles_input.npy")
    expected = np.load(ROOT / "tests" / "data" / "deepslice_shared_angles_official.npy")
    records = [
        {
            "Filenames": f"slice_{index}.png",
            "width": 299,
            "height": 299,
            **{column: float(value) for column, value in zip(COORDINATE_COLUMNS, row)},
        }
        for index, row in enumerate(inputs)
    ]
    original_records = [dict(record) for record in records]

    propagated = TRACKER.propagate_deepslice_shared_angles(records)
    actual = np.asarray(
        [[record[column] for column in COORDINATE_COLUMNS] for record in propagated]
    )

    assert np.array_equal(actual, expected)
    assert records == original_records
    assert [record["Filenames"] for record in propagated] == [
        record["Filenames"] for record in records
    ]
    assert all(record["width"] == 299 and record["height"] == 299 for record in propagated)
