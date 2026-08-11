import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).parents[1] / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


def test_quicknii_adapter_recovers_tracker_plane_exactly():
    shape = (528, 320, 456)
    for index, tilt_lr, tilt_dv in [(216, 0, 0), (156, 10, -6), (276, -8, 7)]:
        lr_slope = np.tan(np.deg2rad(tilt_lr))
        dv_slope = np.tan(np.deg2rad(tilt_dv))
        prediction = {
            "ox": 0.0,
            "oy": 528 - (index - lr_slope * 227.5 - dv_slope * 159.5),
            "oz": 320.0,
            "ux": 456.0,
            "uy": -lr_slope * 456,
            "uz": 0.0,
            "vx": 0.0,
            "vy": -dv_slope * 320,
            "vz": -320.0,
            "width": 456,
            "height": 320,
        }
        recovered_index, recovered_lr, recovered_dv, matrix = TRACKER.quicknii_to_tracker_alignment(
            prediction,
            shape,
        )
        assert recovered_index == pytest.approx(index, abs=1e-9)
        assert recovered_lr == pytest.approx(tilt_lr, abs=1e-9)
        assert recovered_dv == pytest.approx(tilt_dv, abs=1e-9)
        assert matrix == pytest.approx(np.eye(3), abs=1e-9)
        assert TRACKER.volume_to_stereotaxic_um(
            np.array([recovered_index, 0, 0]),
            np.array([216, 0, 0]),
        )[0] == pytest.approx(-(index - 216) * 25)
