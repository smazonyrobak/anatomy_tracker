"""Opt-in Allen CCF runner for the small subject-deformed slab qualification."""

import hashlib
import json
import os
from pathlib import Path

import nrrd

from training.arbitrary_plane_acquisition_v2 import (
    prepare_arbitrary_plane_acquisition_context_v2,
)
from training.arbitrary_plane_support import build_annotation_support_index
from training.slab_refinement_gate_status_v2 import legacy_gate_contract
from training.subject_deformed_slab_qualification_v2 import (
    evaluate_subject_deformed_slab_qualification_v2,
    save_subject_deformed_slab_qualification_v2,
    verify_subject_deformed_slab_qualification_v2,
)


ROOT = Path(__file__).resolve().parents[1]
ATLAS_FOLDER = Path(
    os.environ.get(
        "ANATOMY_TRACKER_ATLAS_FOLDER", str(ROOT / "data" / "Allen Brain Atlas 25um")
    )
).resolve()
OUTPUT = Path(
    os.environ.get(
        "ANATOMY_TRACKER_SUBJECT_DEFORMED_SLAB_QUALIFICATION_OUTPUT",
        str(ROOT / "build" / "subject_deformed_slab_qualification_v2.json"),
    )
).resolve()
OPT_IN_ENVIRONMENT = "ANATOMY_TRACKER_RUN_SUBJECT_DEFORMED_SLAB_QUALIFICATION_V2"
TEMPLATE_SHA256 = "e4a2b483e842b4c8c1b5452d940ea59e14bc1ebaa38fe6a9c3bacac6db2a8f4b"
ANNOTATION_SHA256 = "c620cbcc562183e4dcd40250d440130501781f74b41de35b1c1bdabace290c42"
PYNRRD_VERSION = "1.1.3"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if os.environ.get(OPT_IN_ENVIRONMENT) != "1":
        raise PermissionError(
            f"set {OPT_IN_ENVIRONMENT}=1 only after the paired diagnostic exits"
        )
    if legacy_gate_contract()["qualification_eligible"] is False:
        raise RuntimeError(
            "subject-deformed slab qualification is disabled until the fixed-case "
            "multiresolution replacement gate is predeclared"
        )
    if nrrd.__version__ != PYNRRD_VERSION:
        raise ValueError("pynrrd runtime does not match the frozen decoder version")
    template_path = ATLAS_FOLDER / "average_template_25.nrrd"
    annotation_path = ATLAS_FOLDER / "annotation_25.nrrd"
    if (
        _file_sha256(template_path) != TEMPLATE_SHA256
        or _file_sha256(annotation_path) != ANNOTATION_SHA256
    ):
        raise ValueError("Allen CCF inputs do not match publication/data.lock.yaml")
    template = nrrd.read(str(template_path), index_order="F")[0]
    annotation = nrrd.read(str(annotation_path), index_order="F")[0]
    support = build_annotation_support_index(
        annotation,
        atlas_id="Allen CCFv3",
        atlas_version="2017 25um",
        source_uri="data/Allen Brain Atlas 25um/annotation_25.nrrd",
        source_sha256=ANNOTATION_SHA256,
        source_entity_type="atlas-annotation",
        voxel_size_um=(25.0, 25.0, 25.0),
        origin_um=(0.0, 0.0, 0.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    context = prepare_arbitrary_plane_acquisition_context_v2(
        template,
        annotation,
        support,
        scalar_source_uri="data/Allen Brain Atlas 25um/average_template_25.nrrd",
        scalar_source_sha256=TEMPLATE_SHA256,
        scalar_source_entity_type="atlas-template",
        template_decoder=f"pynrrd {PYNRRD_VERSION}",
        template_index_order="F",
        annotation_decoder=f"pynrrd {PYNRRD_VERSION}",
        annotation_index_order="F",
    )
    report = evaluate_subject_deformed_slab_qualification_v2(context)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_subject_deformed_slab_qualification_v2(OUTPUT, report)
    verify_subject_deformed_slab_qualification_v2(report, context)
    print(
        json.dumps(
            {
                "event": "subject-deformed-slab-qualification-v2-passed",
                "animal_count": report["animal_count"],
                "case_count": report["case_count"],
                "qualification_receipt_sha256": report[
                    "qualification_receipt_sha256"
                ],
                "output": str(OUTPUT),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
