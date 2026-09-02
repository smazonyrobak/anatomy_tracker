import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import training.arbitrary_plane_joint_curriculum_v3 as joint_curriculum
import training.arbitrary_plane_pose_curriculum_v3 as pose_curriculum
import training.arbitrary_plane_row_cache_v3 as row_cache
import training.arbitrary_plane_training_row_v3 as training_row
import training.arbitrary_plane_training_runner_v3 as training_runner
import training.run_arbitrary_plane_authentic_development_v3 as run
from training.arbitrary_plane_rendered_generator import prepare_finite_render_context
from training.arbitrary_plane_support import build_annotation_support_index


def test_first_authentic_development_run_config_is_exact_and_i_only():
    assert all(
        path.drive.upper() == "I:"
        for path in (
            run.ANATOMY_ROOT,
            run.REPOSITORY_ROOT,
            run.TEMPLATE_PATH,
            run.ANNOTATION_PATH,
            run.OUTPUT_ROOT,
            run.TRAIN_CACHE,
            run.DEVELOPMENT_CACHE,
            run.TRAINING_RUN,
            run.TRAIN_CAPTURE_AUDIT,
            run.DEVELOPMENT_CAPTURE_AUDIT,
            run.TEMP_ROOT,
        )
    )
    assert run.REPOSITORY_ROOT == Path(run.__file__).resolve().parents[1]
    assert (run.TRAIN_POSE_ROWS, run.TRAIN_JOINT_ROWS) == (3072, 2048)
    assert (run.DEVELOPMENT_POSE_ROWS, run.DEVELOPMENT_JOINT_ROWS) == (384, 256)
    assert run.SECTIONS_PER_ANIMAL == 16
    assert run.CACHE_CHUNK_SIZE == 48
    assert run.CACHE_GENERATION_WORKERS == 4
    assert run.CATALOGUE_CONFIG == {
        "normal_count": 384,
        "offset_count": 16,
        "roll_count": 16,
        "raster_shape_h_w": (160, 160),
        "raster_physical_span_y_x_um": (12000.0, 12000.0),
    }
    assert 384 * 16 * 16 == 98304
    assert training_runner._complete_model_kwargs(run.MODEL_KWARGS) == {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in sorted(run.MODEL_KWARGS.items())
    }
    assert run.TRAINING_CONFIG["pose_warmup_steps"] == 1000
    assert run.TRAINING_CONFIG["refinement_steps"] == 3
    assert run.TRAINING_CONFIG["joint_pose_only_steps"] == 2
    assert run.TRAINING_CONFIG["retrieval_shape_h_w"] == (48, 48)
    assert run.TRAINING_CONFIG["catalogue_chunk_size"] == 512
    assert run.TRAINING_CONFIG["top_k"] == 4
    assert run.TRAINING_CONFIG["learning_rate"] == 1.0e-3
    assert run.TRAINING_CONFIG["weight_decay"] == 1.0e-4
    assert run.TRAINING_CONFIG["gradient_clip_norm"] == 5.0
    runner_config = pose_curriculum.single_plane_curriculum_runner_config_v3(
        run.RUNNER_CONFIG
    )
    assert training_runner._validate_runner_config(
        runner_config,
        catalogue_cell_count=98304,
        cache_row_count=5120,
        training_top_k=run.TRAINING_CONFIG["top_k"],
        pose_warmup_steps=run.TRAINING_CONFIG["pose_warmup_steps"],
        refinement_steps=run.TRAINING_CONFIG["refinement_steps"],
        joint_pose_only_steps=run.TRAINING_CONFIG["joint_pose_only_steps"],
    ) == runner_config
    assert runner_config["axial_offsets_um"] == [0.0]
    assert runner_config["axial_weights"] == [1.0]
    assert runner_config["candidate_bank_size"] == 512
    assert runner_config["batch_size"] == 4
    assert runner_config["target_applied_steps"] == 4000
    assert not inspect.signature(run.main).parameters


@pytest.fixture(scope="module")
def prepared_context():
    annotation = np.zeros((17, 15, 13), dtype=np.uint16)
    annotation[2:15, 3:13, 1:11] = 7
    annotation[6:11, 6:10, 4:8] = 19
    ap, dv, ml = np.indices(annotation.shape)
    template = (100 + 3 * ap + 5 * dv + 7 * ml).astype(np.uint16)
    support = build_annotation_support_index(
        annotation,
        atlas_id="fixture-ccf",
        atlas_version="fixture-v1",
        source_uri="file:///fixture/annotation.nrrd",
        source_sha256="3" * 64,
        source_entity_type="atlas-annotation",
        voxel_size_um=(11.0, 17.0, 29.0),
        origin_um=(-71.0, 23.0, 107.0),
        coordinate_axis_directions=("posterior", "inferior", "right"),
    )
    return prepare_finite_render_context(
        template,
        annotation,
        support,
        scalar_source_uri="file:///fixture/template.nrrd",
        scalar_source_sha256="4" * 64,
        template_decoder="pynrrd 1.1.3",
        template_index_order="F",
        annotation_decoder="pynrrd 1.1.3",
        annotation_index_order="F",
    )


def _configs(prepared_context):
    source_commit = "a" * 40
    pose_config = pose_curriculum.pose_curriculum_generation_config_v3(
        prepared_context,
        root_seed=2**63 + 55,
        start_index=0,
        row_count=4,
        output_shape_h_w=(47, 53),
        identity_prefix="parallel-pose-v3",
        sections_per_animal=2,
        finite_parent_generator_source_commit=source_commit,
    )
    joint_config = joint_curriculum.joint_curriculum_generation_config_v3(
        prepared_context,
        root_seed=2**63 + 155,
        start_index=0,
        row_count=4,
        output_shape_h_w=(47, 53),
        identity_prefix="parallel-joint-v3",
        sections_per_animal=2,
        finite_parent_generator_source_commit=source_commit,
    )
    composite = joint_curriculum.composite_curriculum_generation_config_v3(
        pose_config, joint_config
    )
    binding = joint_curriculum.composite_curriculum_generator_binding_v3(
        composite
    )
    return pose_config, joint_config, composite, binding, source_commit


def _assert_rows_byte_exact(left, right):
    assert [row["training_row_id"] for row in left] == [
        row["training_row_id"] for row in right
    ]
    assert [row["receipt_sha256"] for row in left] == [
        row["receipt_sha256"] for row in right
    ]
    for serial_row, parallel_row in zip(left, right):
        assert all(
            np.array_equal(
                serial_row["arrays"][name], parallel_row["arrays"][name]
            )
            and serial_row["arrays"][name].dtype
            == parallel_row["arrays"][name].dtype
            for name in training_row._ARRAY_KEYS
        )


def test_ordered_parallel_generation_is_byte_exact(prepared_context):
    pose_config, joint_config, _, _, _ = _configs(prepared_context)
    for module, config in (
        (pose_curriculum, pose_config),
        (joint_curriculum, joint_config),
    ):
        serial = run._component_rows(module, prepared_context, config, 0, 4)
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel = run._ordered_component_rows(
                executor,
                module,
                prepared_context,
                config,
                0,
                4,
            )
        _assert_rows_byte_exact(serial, parallel)


def test_parallel_cache_append_order_and_crash_resume_are_exact(
    tmp_path, prepared_context, monkeypatch
):
    configs = _configs(prepared_context)
    serial_cache = tmp_path / "serial-cache"
    parallel_cache = tmp_path / "parallel-cache"
    interrupted_cache = tmp_path / "interrupted-cache"
    serial = run._resume_composite_cache(
        serial_cache,
        prepared_context,
        row_cache,
        pose_curriculum,
        joint_curriculum,
        *configs[:-1],
        configs[-1],
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = run._resume_composite_cache(
            parallel_cache,
            prepared_context,
            row_cache,
            pose_curriculum,
            joint_curriculum,
            *configs[:-1],
            configs[-1],
            executor=executor,
        )
    assert parallel == serial

    original = run._ordered_component_rows

    def fail_after_generation(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated crash before ordered append")

    monkeypatch.setattr(run, "_ordered_component_rows", fail_after_generation)
    with pytest.raises(RuntimeError, match="simulated crash"):
        with ThreadPoolExecutor(max_workers=4) as executor:
            run._resume_composite_cache(
                interrupted_cache,
                prepared_context,
                row_cache,
                pose_curriculum,
                joint_curriculum,
                *configs[:-1],
                configs[-1],
                executor=executor,
            )
    open_manifest = row_cache.load_training_row_cache_manifest_v3(
        interrupted_cache,
        expected_generator_binding=configs[3],
    )
    assert open_manifest["row_count"] == 0
    monkeypatch.setattr(run, "_ordered_component_rows", original)
    with ThreadPoolExecutor(max_workers=4) as executor:
        resumed = run._resume_composite_cache(
            interrupted_cache,
            prepared_context,
            row_cache,
            pose_curriculum,
            joint_curriculum,
            *configs[:-1],
            configs[-1],
            executor=executor,
        )
    assert resumed == serial
