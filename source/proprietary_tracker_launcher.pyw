import os
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYQTGRAPH_QT_LIB"] = "PySide6"
env["TRAJECTORY_ATLAS_FOLDER"] = str(APP_DIR / "data" / "Allen Brain Atlas 25um")
env["TRAJECTORY_SLICES_FOLDER"] = ""
env["TRAJECTORY_RUN_FOLDER"] = ""
tracker = APP_DIR / "tools" / "TrajectoryTracker" / "TrajectoryTracker.exe"
subprocess.Popen([str(tracker)], cwd=str(tracker.parent), env=env)
