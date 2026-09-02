import inspect

import training.arbitrary_plane_pose_curriculum_v3 as pose_curriculum
import training.arbitrary_plane_training_runner_v3 as training_runner
import training.run_arbitrary_plane_authentic_development_v3 as run


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
    assert (run.TRAIN_POSE_ROWS, run.TRAIN_JOINT_ROWS) == (3072, 2048)
    assert (run.DEVELOPMENT_POSE_ROWS, run.DEVELOPMENT_JOINT_ROWS) == (384, 256)
    assert run.SECTIONS_PER_ANIMAL == 16
    assert run.CACHE_CHUNK_SIZE == 48
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
