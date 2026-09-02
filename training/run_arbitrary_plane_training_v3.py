"""Resume one fully prepared I:-drive arbitrary-plane development run."""

from __future__ import annotations

import json
import sys

from training.arbitrary_plane_training_runner_v3 import run_training_until_target_v3


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m training.run_arbitrary_plane_training_v3 I:\\path\\to\\prepared-run")
    state = run_training_until_target_v3(sys.argv[1])
    print(json.dumps(state, sort_keys=True, separators=(",", ":")))
