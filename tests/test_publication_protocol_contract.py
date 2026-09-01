from pathlib import Path


PROTOCOL = (
    Path(__file__).parents[1] / "publication" / "protocol.yaml"
).read_text(encoding="utf-8")


def test_standalone_training_excludes_historical_product5_and_mixed_real_regimen():
    assert (
        'release_training_data_policy: "Allen CCF-derived synthetic '
        'arbitrary-plane views only"'
    ) in PROTOCOL
    assert (
        'role: ["historical_real_pose_supervision", '
        '"historical_real_appearance", "historical_development_selection"]'
    ) in PROTOCOL
    assert "historical_product5_mixed_real_training:" in PROTOCOL
    assert 'status: "legacy_record_only"' in PROTOCOL
    assert PROTOCOL.count("standalone_arbitrary_plane_release_eligible: false") == 2


def test_landmark_tre_is_the_unique_anatomical_primary_endpoint():
    assert 'primary_anatomical_endpoint: "brain_level_median_landmark_tre_um"' in PROTOCOL
    assert "primary_endpoints:" not in PROTOCOL
    assert "secondary_pose_track_metrics:" in PROTOCOL
    assert '- "brain_level_mean_physical_plane_distance_um"' in PROTOCOL


def test_three_dof_plane_covariance_cannot_propagate_downstream_confidence():
    assert "current_covariance_dof: 3" in PROTOCOL
    assert "full_finite_frame_uncertainty_represented: false" in PROTOCOL
    assert "deformation_uncertainty_represented: false" in PROTOCOL
    assert "downstream_confidence_propagation_allowed: false" in PROTOCOL
    assert "downstream_propagation_prerequisite:" in PROTOCOL
