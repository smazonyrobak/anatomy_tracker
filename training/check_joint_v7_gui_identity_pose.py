"""Offscreen identity/manual-pose roundtrip on one already-completed real-image result."""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ.update(TEMP=str(ROOT / 'tmp'), TMP=str(ROOT / 'tmp'), QT_QPA_PLATFORM='offscreen',
                  CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))

import json
import numpy as np
import cv2
from copy import deepcopy
import proprietary_trajectory_tool as tracker

OUT = ROOT / 'runs/joint_v7_gui_identity_pose_smoke_004'
OUT.mkdir(exist_ok=False)
PARENT = ROOT / 'runs/joint_v7_gui_workflow_smoke_002/real_train_joint_v7.attracker'
with (ROOT / 'data/allen_real_training_expansion_20260929/union_training_index.jsonl').open() as stream:
    record = json.loads(next(stream))
expected_ids = {name: str(record[name]) for name in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')}
errors = []
for name in ('critical', 'warning'):
    setattr(tracker.QtWidgets.QMessageBox, name,
            lambda parent, title, message, *args: errors.append({'title': title, 'message': message}))
app = tracker.QtWidgets.QApplication.instance() or tracker.QtWidgets.QApplication([])
window = tracker.TrajectoryTrackerWindow(default_atlas_folder=ROOT / 'data/Allen Brain Atlas 25um',
                                         default_slices_folder=OUT, default_run_folder=OUT)
window.load_session_file(PARENT)
session = window.current_session()
assert tracker.file_sha256(Path(session.path)) == record['actual_image_sha256']
assert all(getattr(session, name) == '' for name in expected_ids), 'Missing old-archive IDs must remain unknown'
assert 'Missing animal ID' in window.probe_fit_summary.text()


def accept_dialog(values):
    dialog = app.activeModalWidget()
    for name, value in values.items():
        editor = dialog.findChild(tracker.QtCore.QObject, name)
        if isinstance(editor, tracker.QtWidgets.QLineEdit):
            editor.setText(value)
        elif isinstance(editor, tracker.QtWidgets.QComboBox):
            editor.setCurrentText(value)
        else:
            editor.setValue(value)
    dialog.accept()


tracker.QtCore.QTimer.singleShot(0, lambda: accept_dialog(expected_ids))
window.slice_identifiers_btn.click()
assert {name: getattr(session, name) for name in expected_ids} == expected_ids
print('Actual identifier dialog stored the four explicit source IDs', flush=True)
original_arrays = {key: value.copy() for key, value in session.joint_model_arrays.items()}
original_surface = session.atlas_surface_ccf_um.copy()
original_ouv = np.asarray(session.atlas_ouv_ap_dv_ml_um).reshape(3, 3)
original_points = np.asarray(session.probe_traces['smoke_mapping_only'].volume_points) * 25 + 12.5
translation = np.array([125., -50., 75.])
rotation = cv2.Rodrigues(np.array([0., 0., np.deg2rad(2.)]))[0]
height, width = session.atlas_raster_shape_h_w
centre = original_ouv[0] + (width - 1) / (2 * width) * original_ouv[1] + (height - 1) / (2 * height) * original_ouv[2]
offset = centre + translation - rotation @ centre
window.alignment_tabs.setCurrentIndex(0)
tracker.QtCore.QTimer.singleShot(0, lambda: accept_dialog({
    'translation_AP': 125., 'translation_DV': -50., 'translation_ML': 75.,
    'rotation_axis': 'ML', 'rotation_angle': 2.}))
window.native_pose_btn.click()
assert not errors, errors
expected_surface = original_surface.astype(np.float64) @ rotation.T + offset
np.testing.assert_array_equal(session.atlas_surface_ccf_um, expected_surface)
expected_ouv = original_ouv @ rotation.T
expected_ouv[0] += offset
np.testing.assert_array_equal(session.atlas_ouv_ap_dv_ml_um, expected_ouv)
np.testing.assert_allclose(expected_ouv[0] + (width - 1) / (2 * width) * expected_ouv[1]
                           + (height - 1) / (2 * height) * expected_ouv[2], centre + translation)
corrected_points = np.asarray(session.probe_traces['smoke_mapping_only'].volume_points) * 25 + 12.5
point_error = np.max(np.abs(corrected_points - (original_points @ rotation.T + offset)))
assert point_error < .005
for key, value in original_arrays.items():
    np.testing.assert_array_equal(session.joint_model_arrays[key], value)
assert session.auto_alignment_diagnostics['joint_model']['uncertainty_invalidated_by_manual_edit']
assert session.manual_refined_from_run_id
archive = OUT / 'corrected_identified.attracker'
window.save_session_file(archive)
restored = tracker.TrajectoryTrackerWindow(default_atlas_folder=ROOT / 'data/Allen Brain Atlas 25um',
                                           default_slices_folder=OUT, default_run_folder=OUT)
restored.load_session_file(archive)
loaded = restored.current_session()
assert {name: getattr(loaded, name) for name in expected_ids} == expected_ids
np.testing.assert_array_equal(loaded.atlas_surface_ccf_um, session.atlas_surface_ccf_um)
np.testing.assert_array_equal(loaded.atlas_ouv_ap_dv_ml_um, session.atlas_ouv_ap_dv_ml_um)
np.testing.assert_array_equal(loaded.probe_traces['smoke_mapping_only'].volume_points,
                              session.probe_traces['smoke_mapping_only'].volume_points)
assert loaded.auto_alignment_diagnostics == session.auto_alignment_diagnostics
for key, value in original_arrays.items():
    np.testing.assert_array_equal(loaded.joint_model_arrays[key], value)
restored.native_pose_restore_btn.click()
assert not errors, errors
np.testing.assert_array_equal(loaded.atlas_surface_ccf_um, original_surface)
np.testing.assert_array_equal(loaded.atlas_ouv_ap_dv_ml_um, original_ouv)
np.testing.assert_array_equal(np.asarray(loaded.probe_traces['smoke_mapping_only'].volume_points) * 25 + 12.5,
                              original_points)
restored.load_slice(Path(session.path))
assert all(getattr(restored.current_session(), name) == '' for name in expected_ids)
second = restored.current_session()
second.probe_traces = deepcopy(loaded.probe_traces)
restored._switch_slice(1)
assert 'Missing animal ID' in restored.probe_fit_summary.text()
assert not any(isinstance(item, tracker.gl.GLScatterPlotItem) for item in restored.dynamic_gl_items)
blocked_calls = {
    'volume_points': lambda: restored.all_probe_volume_points('smoke_mapping_only'),
    'signal_values': lambda: restored.all_probe_signal_values('smoke_mapping_only'),
    'by_slice': lambda: restored.probe_observations_by_slice('smoke_mapping_only'),
    'weights': lambda: restored.probe_regression_weights('smoke_mapping_only', 4),
    'regression': lambda: restored.probe_regression('smoke_mapping_only'),
    'brain_geometry': lambda: restored.probe_brain_geometry('smoke_mapping_only'),
    'line_geometry': lambda: restored.probe_line_geometry('smoke_mapping_only'),
    'manifest': lambda: restored._write_manifest(OUT / 'must_not_exist', 'smoke_mapping_only',
                                                'deepest_mark', *([np.zeros(3)] * 5), 0., 0.),
}
for name, call in blocked_calls.items():
    try:
        call()
    except tracker.InfeasibleProbeConstraint as exc:
        assert 'Missing animal ID' in str(exc), (name, str(exc))
    else:
        raise AssertionError(f'{name} silently pooled unknown animal')
assert not (OUT / 'must_not_exist').exists()
restored.map_btn.click()
assert len(errors) == 1 and errors.pop()['title'] == 'Animal identity required'
restored.save_session_file(OUT / 'unresolved_identity.attracker')
for ids, reason in (({'animal_id': 'different-animal'}, 'Conflicting animal IDs'),
                    ({'animal_id': expected_ids['animal_id'], 'specimen_id': 'different-specimen'}, 'Conflicting specimen IDs')):
    tracker.QtCore.QTimer.singleShot(0, lambda values=ids: accept_dialog(values))
    restored.slice_identifiers_btn.click()
    assert reason in restored.probe_fit_summary.text()
    restored.map_btn.click()
    assert len(errors) == 1 and reason in errors.pop()['message']
tracker.QtCore.QTimer.singleShot(0, lambda: accept_dialog({'specimen_id': ''}))
restored.slice_identifiers_btn.click()
assert 'IDs' not in restored.probe_fit_summary.text() and 'Missing animal ID' not in restored.probe_fit_summary.text()
assert len(restored.all_probe_volume_points('smoke_mapping_only')) == 4
assert len(restored.probe_observations_by_slice('smoke_mapping_only')) == 2
# Probe names do not supply an animal grouping: a different probe is still the same workspace.
second.probe_traces = {'other_probe': second.probe_traces['smoke_mapping_only']}
second.animal_id = 'different-animal'
restored.slice_list.setCurrentIndex(0)
assert 'Conflicting animal IDs' in restored.probe_fit_summary.text()
# Legacy-only, all-unknown behavior stays unchanged, using copies of these real-image observations.
native_sessions = restored.sessions
restored.sessions = [tracker.replace(item, animal_id='', specimen_id='', atlas_ouv_ap_dv_ml_um=None,
                                     atlas_surface_ccf_um=None, joint_model_arrays=None) for item in native_sessions]
assert len(restored.all_probe_volume_points('smoke_mapping_only')) == 2
assert len(restored.all_probe_volume_points('other_probe')) == 2
restored.sessions = native_sessions
restored.slice_list.setCurrentIndex(1)
tracker.QtWidgets.QMessageBox.question = lambda *args: tracker.QtWidgets.QMessageBox.StandardButton.Yes
restored.remove_selected_slice_btn.click()
assert len(restored.sessions) == 1 and len(restored.all_probe_volume_points('smoke_mapping_only')) == 2
assert 'Conflicting animal IDs' not in restored.probe_fit_summary.text()
assert not errors, errors
receipt = {'scope': 'offscreen actual GUI identity and rigid-pose editing; no new inference or accuracy validation',
           'parent_archive': str(PARENT), 'parent_archive_sha256': tracker.file_sha256(PARENT),
           'source_identity': expected_ids, 'source_image_sha256': record['actual_image_sha256'],
           'missing_old_archive_ids_stayed_unknown': True, 'new_duplicate_image_ids_stayed_unknown': True,
           'actual_identifier_dialog_passed': True, 'actual_pose_dialog_passed': True,
           'rotation_axis': 'CCF ML', 'rotation_deg': 2., 'translation_ccf_um': translation.tolist(),
           'rotation_pivot': 'O + (W-1)/(2W)*U + (H-1)/(2H)*V',
           'pixel_centre_midpoint_moves_only_by_translation': True,
           'identity_blocked_aggregation_paths': list(blocked_calls),
           'missing_conflicting_animal_conflicting_specimen_mapping_buttons_blocked': True,
           'single_animal_blank_specimen_allowed': True, 'different_probe_not_animal_grouping': True,
           'legacy_only_all_unknown_unchanged': True, 'unresolved_identity_archive_saved': True,
           'identity_edit_switch_remove_refresh_summary_immediately': True,
           'full_surface_rigid_transform_exact': True, 'plane_rigid_transform_exact': True,
           'probe_rigid_transform_max_abs_error_um': float(point_error),
           'original_model_arrays_unchanged': True, 'uncertainty_invalidated': True,
           'corrected_ids_surface_plane_probes_metadata_arrays_archive_roundtrip_exact': True,
           'restore_predicted_pose_button_exact': True, 'archive_sha256': tracker.file_sha256(archive),
           'source_sha256': {name: tracker.file_sha256(Path(__file__).resolve().parents[1] / name)
                             for name in ('source/proprietary_trajectory_tool.py', 'training/check_joint_v7_gui_identity_pose.py')}}
(OUT / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
restored.close()
window.close()
app.processEvents()
