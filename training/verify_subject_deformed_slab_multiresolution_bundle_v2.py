"""Independent verifier for a frozen fixed-case multiresolution raw bundle."""

import json
import os
from pathlib import Path

import training.subject_deformed_slab_multiresolution_bundle_v2 as bundle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ENVIRONMENT = "ANATOMY_TRACKER_SUBJECT_SLAB_MULTIRESOLUTION_OUTPUT"
ATLAS_FOLDER = Path(
    os.environ.get(
        "ANATOMY_TRACKER_ATLAS_FOLDER", str(ROOT / "data" / "Allen Brain Atlas 25um")
    )
).resolve()


def main() -> None:
    value = os.environ.get(OUTPUT_ENVIRONMENT)
    if not value or not Path(value).is_absolute():
        raise ValueError(f"{OUTPUT_ENVIRONMENT} must name the frozen absolute directory")
    output = Path(value).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("frozen multiresolution output must be outside the repository")
    if not output.is_dir():
        raise FileNotFoundError("frozen multiresolution output directory does not exist")
    context, allen_inputs = bundle.load_pinned_allen_context(ATLAS_FOLDER)
    verified = bundle.verify_frozen_bundle_v2(output, context, allen_inputs)
    print(
        json.dumps(
            {"event": "subject-slab-fixed-case-multiresolution-verified", **verified},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
