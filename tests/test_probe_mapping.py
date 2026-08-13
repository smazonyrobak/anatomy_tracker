import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).parents[1] / "source" / "proprietary_trajectory_tool.py"
SPEC = importlib.util.spec_from_file_location("trajectory_tracker_probe_mapping_tests", SOURCE)
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)
VOXEL_UM = TRACKER.VOXEL_UM
probe_mapping_coordinates = TRACKER.probe_mapping_coordinates


def test_deepest_mark_tip_places_contacts_toward_surface():
    entry = np.array([20.0, 10.0, 30.0])
    deepest = np.array([20.0, 50.0, 30.0])
    surface_direction = np.array([0.0, -1.0, 0.0])
    tip, y0, contacts, tip_depth, observed_depth = probe_mapping_coordinates(
        entry,
        deepest,
        surface_direction,
        "deepest_mark_is_tip",
        None,
        np.array([0.0, 20.0, 1000.0]),
        "Neuropixels 1.0",
    )
    assert np.allclose(tip, deepest)
    assert tip_depth == observed_depth == 40.0 * VOXEL_UM
    assert np.allclose(y0, tip + np.array([0.0, -220.0 / VOXEL_UM, 0.0]))
    assert np.all(np.diff(contacts[:, 1]) < 0.0)


def test_known_depth_places_unobserved_tip_below_deepest_mark():
    entry = np.array([20.0, 10.0, 30.0])
    deepest = np.array([20.0, 50.0, 30.0])
    tip, _, contacts, tip_depth, observed_depth = probe_mapping_coordinates(
        entry,
        deepest,
        np.array([0.0, -1.0, 0.0]),
        "known_insertion_depth",
        2000.0,
        np.array([0.0, 1000.0]),
        "Neuropixels 2.0 single-shank",
    )
    assert observed_depth == 40.0 * VOXEL_UM
    assert tip_depth == 2000.0
    assert tip[1] > deepest[1]
    assert contacts[0, 1] < tip[1]
    assert contacts[1, 1] < contacts[0, 1]


def test_known_depth_cannot_be_shallower_than_observed_trace():
    with pytest.raises(ValueError, match="shallower than the deepest marked"):
        probe_mapping_coordinates(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 80.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            "known_insertion_depth",
            1000.0,
            np.array([0.0]),
            "Neuropixels 1.0",
        )


def test_channel_geometry_cannot_extend_beyond_physical_shank():
    with pytest.raises(ValueError, match="outside the 10000 um physical shank"):
        probe_mapping_coordinates(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 40.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            "deepest_mark_is_tip",
            None,
            np.array([9900.0]),
            "Neuropixels 1.0",
        )


def test_regions_are_sampled_at_the_expected_contact_voxels():
    entry = np.array([2.0, 4.0, 2.0])
    tip = np.array([2.0, 84.0, 2.0])
    _, _, contacts, _, _ = probe_mapping_coordinates(
        entry,
        tip,
        np.array([0.0, -1.0, 0.0]),
        "deepest_mark_is_tip",
        None,
        np.array([0.0, 500.0, 1000.0]),
        "Neuropixels 1.0",
    )
    annotation = np.zeros((5, 100, 5), dtype=np.uint16)
    expected_ids = [11, 22, 33]
    for coordinate, region_id in zip(contacts, expected_ids, strict=True):
        ap, dv, ml = np.rint(coordinate).astype(int)
        annotation[ap, dv, ml] = region_id
    sampled_ids = [annotation[tuple(np.rint(coordinate).astype(int))] for coordinate in contacts]
    assert sampled_ids == expected_ids


def test_unit_regions_are_inherited_from_peak_channel_with_composite_probe_key():
    channels = TRACKER.pd.DataFrame(
        {
            "probe_name": ["imec0", "imec1"],
            "probe_channel_number": [7, 7],
            "probe_horizontal_position": [0.0, 0.0],
            "probe_vertical_position": [500.0, 500.0],
            "structure_id": [101, 202],
            "structure_name": ["Region A", "Region B"],
            "structure_acronym": ["A", "B"],
        }
    )
    units = TRACKER.pd.DataFrame(
        {
            "unit_key": ["imec0:1", "imec1:1"],
            "probe_name": ["imec0", "imec1"],
            "probe_channel_number": [7, 7],
        }
    )
    mapped = TRACKER.attach_peak_channel_metadata(channels, units)
    assert mapped["structure_acronym"].tolist() == ["A", "B"]
    assert mapped["structure_id"].tolist() == [101, 202]


def test_probe_anatomy_view_uses_physical_sites_atlas_colors_and_unit_counts():
    channels = TRACKER.pd.DataFrame(
        {
            "probe_name": ["imec0", "imec0", "imec0", "imec1"],
            "probe_channel_number": [0, 1, 2, 0],
            "probe_horizontal_position": [11.0, 59.0, 27.0, 11.0],
            "probe_vertical_position": [0.0, 20.0, 40.0, 0.0],
            "trajectory_distance_um": [220.0, 240.0, 260.0, 220.0],
            "structure_id": [101, 101, 202, 303],
            "structure_name": ["Region A", "Region A", "Region B", "Region C"],
            "structure_acronym": ["A", "A", "B", "C"],
        }
    )
    units = TRACKER.pd.DataFrame(
        {
            "unit_key": ["imec0:1", "imec0:2", "imec0:3", "imec1:1"],
            "probe_name": ["imec0", "imec0", "imec0", "imec1"],
            "probe_channel_number": [0, 0, 2, 0],
        }
    )

    sites, summary = TRACKER.probe_anatomy_view_data(
        channels, units, "imec0", {101: "112233", 202: "AABBCC"}
    )

    assert sites["probe_channel_number"].tolist() == [0, 1, 2]
    assert sites["structure_color_hex"].tolist() == ["112233", "112233", "AABBCC"]
    assert sites["unit_count"].tolist() == [2, 0, 1]
    assert summary["structure_acronym"].tolist() == ["A", "B"]
    assert summary["channels"].tolist() == [2, 1]
    assert summary["units"].tolist() == [2, 1]
    assert summary["depth_min_um"].tolist() == [220.0, 260.0]
    assert summary["depth_max_um"].tolist() == [240.0, 260.0]


def test_probe_anatomy_dialog_renders_mapped_sites():
    app = TRACKER.QtWidgets.QApplication.instance() or TRACKER.QtWidgets.QApplication([])
    channels = TRACKER.pd.DataFrame(
        {
            "probe_name": ["imec0", "imec0"],
            "probe_channel_number": [0, 1],
            "probe_horizontal_position": [11.0, 59.0],
            "probe_vertical_position": [0.0, 20.0],
            "trajectory_distance_um": [220.0, 240.0],
            "structure_id": [101, 202],
            "structure_name": ["Region A", "Region B"],
            "structure_acronym": ["A", "B"],
        }
    )
    units = TRACKER.pd.DataFrame(
        {
            "unit_key": ["imec0:1"],
            "probe_name": ["imec0"],
            "probe_channel_number": [0],
        }
    )
    sites, summary = TRACKER.probe_anatomy_view_data(
        channels, units, "imec0", {101: "112233", 202: "AABBCC"}
    )
    dialog = TRACKER.ProbeAnatomyDialog("imec0", sites, summary)
    dialog.show()
    app.processEvents()
    image = dialog.grab().toImage()
    assert image.width() >= 900
    assert image.height() >= 700
    assert "imec0" in dialog.windowTitle()
    dialog.close()
    app.processEvents()
