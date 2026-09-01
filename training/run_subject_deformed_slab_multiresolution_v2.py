"""Opt-in execution of the threshold-free fixed-case multiresolution replacement."""

import json
import os
import subprocess
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
EXPECTED_COMMIT_ENVIRONMENT = "ANATOMY_TRACKER_EXPECTED_SOURCE_COMMIT"
EXPECTED_BRANCH = "codex/arbitrary-plane-joint-model"
UPSTREAM_REF = f"origin/{EXPECTED_BRANCH}"


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_state_v2() -> dict[str, object]:
    expected_commit = os.environ.get(EXPECTED_COMMIT_ENVIRONMENT, "").strip()
    repository_root = Path(_git_output("rev-parse", "--show-toplevel")).resolve()
    head = _git_output("rev-parse", "HEAD")
    branch = _git_output("branch", "--show-current")
    upstream_head = _git_output("rev-parse", UPSTREAM_REF)
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    if (
        not expected_commit
        or repository_root != ROOT
        or head != expected_commit
        or branch != EXPECTED_BRANCH
        or upstream_head != head
        or status
    ):
        raise RuntimeError(
            "multiresolution execution requires the explicit expected pushed commit "
            "and a clean tracked, staged, and untracked worktree"
        )
    return {
        "branch": branch,
        "head": head,
        "expected_commit": expected_commit,
        "upstream_ref": UPSTREAM_REF,
        "upstream_head": upstream_head,
        "clean_tracked_staged_untracked": True,
    }


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
    repository = repository_state_v2()
    context, allen_inputs = bundle.load_pinned_allen_context(ATLAS_FOLDER)
    (
        failed_report,
        live_report_capability,
    ) = legacy._evaluate_subject_deformed_slab_qualification_with_capability_v2(
        context
    )
    plan = assessment._make_fixed_case_multiresolution_plan_from_live_report_v2(
        failed_report,
        context,
        live_report_capability=live_report_capability,
    )
    subject_plan, subject_to_ccf_mapper = legacy._make_subject_plan_with_mapper(
        context, plan["selected_first_failure"]["animal_manifest"]
    )
    rendered = assessment._render_fixed_case_multiresolution_with_mapper_v2(
        context,
        plan,
        subject_plan,
        subject_to_ccf_mapper=subject_to_ccf_mapper,
    )
    result = assessment.assemble_fixed_case_multiresolution_assessment_v2(
        plan, rendered
    )
    if repository_state_v2() != repository:
        raise RuntimeError("repository state changed before bundle write")
    staging = bundle.write_staged_bundle_v2(
        output,
        repository=repository,
        allen_inputs=allen_inputs,
        failed_report=failed_report,
        plan=plan,
        subject_plan=subject_plan,
        rendered=rendered,
        result=result,
    )
    verified = bundle.verify_frozen_bundle_v2(
        staging, context, allen_inputs, repository=repository
    )
    if repository_state_v2() != repository:
        raise RuntimeError("repository state changed before bundle publish")
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
