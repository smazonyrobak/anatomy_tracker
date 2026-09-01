"""Opt-in execution of the threshold-free fixed-case multiresolution replacement."""

import json
import os
from pathlib import Path

import training.slab_refinement_gate_status_v2 as gate_status
import training.subject_deformed_slab_multiresolution_assessment_v2 as assessment
import training.subject_deformed_slab_multiresolution_bundle_v2 as bundle
import training.subject_deformed_slab_qualification_v2 as legacy


ROOT = Path(__file__).resolve().parents[1]
OPT_IN_ENVIRONMENT = "ANATOMY_TRACKER_RUN_SUBJECT_SLAB_MULTIRESOLUTION_V2"
OUTPUT_ENVIRONMENT = "ANATOMY_TRACKER_SUBJECT_SLAB_MULTIRESOLUTION_OUTPUT"
ATLAS_FOLDER = Path(
    os.environ.get(
        "ANATOMY_TRACKER_ATLAS_FOLDER", str(ROOT / "data" / "Allen Brain Atlas 25um")
    )
).resolve()


def _external_output() -> Path:
    value = os.environ.get(OUTPUT_ENVIRONMENT)
    if not value or not Path(value).is_absolute():
        raise ValueError(f"{OUTPUT_ENVIRONMENT} must be an explicit absolute directory")
    output = Path(value).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("multiresolution output must be outside the repository")
    if output.exists() or (output.parent / f".{output.name}.partial").exists():
        raise FileExistsError("frozen output or its partial sibling already exists")
    return output


def main() -> None:
    if os.environ.get(OPT_IN_ENVIRONMENT) != "1":
        raise PermissionError(
            f"set {OPT_IN_ENVIRONMENT}=1 only after the paired diagnostic exits"
        )
    rejected = gate_status.legacy_gate_contract()
    if (
        rejected["decision"] != "reject_legacy_universal_gate"
        or rejected["qualification_eligible"] is not False
    ):
        raise RuntimeError("legacy gate status does not authorize the replacement path")
    output = _external_output()
    context, allen_inputs = bundle.load_pinned_allen_context(ATLAS_FOLDER)
    failed_report = legacy.evaluate_subject_deformed_slab_qualification_v2(context)
    plan = assessment.make_fixed_case_multiresolution_plan_v2(
        failed_report, context
    )
    subject_plan = legacy._make_subject_plan(
        context, plan["selected_first_failure"]["animal_manifest"]
    )
    rendered = assessment.render_fixed_case_multiresolution_v2(
        context, plan, subject_plan
    )
    result = assessment.assemble_fixed_case_multiresolution_assessment_v2(
        plan, rendered
    )
    staging = bundle.write_staged_bundle_v2(
        output,
        allen_inputs=allen_inputs,
        failed_report=failed_report,
        plan=plan,
        subject_plan=subject_plan,
        rendered=rendered,
        result=result,
    )
    verified = bundle.verify_frozen_bundle_v2(
        staging, context, allen_inputs
    )
    bundle.publish_staged_bundle_v2(staging, output)
    print(
        json.dumps(
            {
                "event": "subject-slab-fixed-case-multiresolution-recorded",
                "output": str(output),
                **verified,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
