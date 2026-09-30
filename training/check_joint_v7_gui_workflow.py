"""One real TRAIN image through the actual offscreen GUI worker and archive path.

Engineering smoke only: the two marked pixels are mapping checks, not biological
electrode annotations or a validation of anatomical prediction accuracy.
"""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ.update(TEMP=str(ROOT / 'tmp'), TMP=str(ROOT / 'tmp'), QT_QPA_PLATFORM='offscreen',
                  CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))

import hashlib
import json
import time
import zipfile

import numpy as np
import torch
import proprietary_trajectory_tool as tracker

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
OUT = ROOT / 'runs/joint_v7_gui_workflow_smoke_002'
OUT.mkdir(exist_ok=False)
CHECKPOINT = ROOT / 'runs/joint_v7_direct_joint_001/joint_step_06000.pt'
with (ROOT / 'data/allen_real_training_expansion_20260929/union_training_index.jsonl').open() as stream:
    record = json.loads(next(stream))
source = Path(record['source_image_path'])
assert tracker.file_sha256(source) == record['actual_image_sha256']
errors = []
for name in ('critical', 'warning'):
    setattr(tracker.QtWidgets.QMessageBox, name,
            lambda parent, title, message, *args: errors.append({'title': title, 'message': message}))
app = tracker.QtWidgets.QApplication.instance() or tracker.QtWidgets.QApplication([])
print('Instantiating the actual GUI offscreen and loading the pinned atlas and one TRAIN image', flush=True)
window = tracker.TrajectoryTrackerWindow(default_atlas_folder=ROOT / 'data/Allen Brain Atlas 25um',
                                         default_slices_folder=OUT, default_run_folder=OUT)
window.load_slice(source)
session = window.current_session()
h, w = session.raw_display.shape
points = np.array([[.5 * (w - 1), .4 * (h - 1)], [.5 * (w - 1), .6 * (h - 1)]])
session.probe_traces['smoke_mapping_only'] = tracker.ProbeTrace(slice_points=points.tolist())
window.probe_name.addItem('smoke_mapping_only')
window.probe_name.setCurrentText('smoke_mapping_only')
window.joint_checkpoint.setText(str(CHECKPOINT))
window.joint_device.setCurrentText('CPU')
window.joint_thickness.setValue(float(record['section_thickness_um']))
window.alignment_tabs.setCurrentIndex(2)
assert window.joint_run_btn.isEnabled(), errors
start = time.monotonic()
window.joint_run_btn.click()
assert window.auto_alignment_busy, errors
print('Clicked the actual experimental joint-model button; CPU worker active', flush=True)
while window.auto_alignment_busy and time.monotonic() - start < 600:
    app.processEvents()
    time.sleep(.025)
assert not window.auto_alignment_busy, 'CPU inference did not finish within ten minutes'
assert not errors, errors
assert session.atlas_surface_ccf_um is not None, window.status.text()
assert session.transformed_overlay.shape[:2] == session.atlas_surface_ccf_um.shape[:2]
assert window.current_atlas_image.shape == session.atlas_surface_ccf_um.shape[:2]
assert window._coordinate_registration(session)['kind'] == 'joint-native'
before = np.asarray(session.probe_traces['smoke_mapping_only'].volume_points)
metadata = session.auto_alignment_diagnostics['joint_model']
direct, valid = tracker.joint_raw_points_to_ccf(points, metadata['raw_to_model_xy'],
                                              session.atlas_surface_ccf_um, session.raw_display.shape)
assert valid.all()
np.testing.assert_allclose(before * 25. + 12.5, direct, rtol=0, atol=1e-5)
assert not metadata['probabilities_calibrated'] and not metadata['constraints_used']
inference_seconds = time.monotonic() - start
print(f'Native result installed and two marked pixels mapped; elapsed {inference_seconds:.2f}s', flush=True)
archive = OUT / 'real_train_joint_v7.attracker'
window.save_session_file(archive)
restored = tracker.TrajectoryTrackerWindow(default_atlas_folder=ROOT / 'data/Allen Brain Atlas 25um',
                                           default_slices_folder=OUT, default_run_folder=OUT)
restored.load_session_file(archive)
loaded = restored.current_session()
after = np.asarray(loaded.probe_traces['smoke_mapping_only'].volume_points)
np.testing.assert_array_equal(before, after)
np.testing.assert_array_equal(session.atlas_surface_ccf_um, loaded.atlas_surface_ccf_um)
for key in session.joint_model_arrays:
    np.testing.assert_array_equal(session.joint_model_arrays[key], loaded.joint_model_arrays[key])
np.testing.assert_array_equal(session.transformed_overlay, loaded.transformed_overlay)
assert not errors, errors
with zipfile.ZipFile(archive) as saved:
    archive_version = json.loads(saved.read('session.json'))['version']
receipt = {'scope': 'one-image offscreen GUI engineering smoke; not accuracy or desktop visual validation',
           'source_image': str(source), 'source_image_sha256': record['actual_image_sha256'],
           'source_identity': {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id', 'split')},
           'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': metadata['checkpoint_sha256'],
           'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'inference_install_seconds': inference_seconds,
           'test_points_role': 'arbitrary pixel-coordinate engineering checks, not true probe annotations',
           'raw_points_xy': points.tolist(), 'before_archive_ccf_voxels': before.tolist(),
           'after_archive_ccf_voxels': after.tolist(), 'archive_version': archive_version,
           'archive_sha256': tracker.file_sha256(archive), 'saved_prediction_array_count': len(session.joint_model_arrays),
           'max_probe_roundtrip_error_um': float(np.max(np.abs(before - after)) * 25),
           'whole_curved_surface_exact': True, 'whole_prediction_arrays_exact': True, 'overlay_exact': True,
           'atlas_image_shape': list(window.current_atlas_image.shape), 'legacy_inference_used': False,
           'constraints_used': False, 'uncertainty_calibrated': False,
           'source_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                             for name in ('source/joint_slice_worker.py', 'source/joint_slice_input.py',
                                          'source/proprietary_trajectory_tool.py', 'training/check_joint_v7_gui_workflow.py')}}
(OUT / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
restored.close()
window.close()
app.processEvents()
