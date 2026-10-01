"""Paired direct-pose readout on six real development donors with weak Allen affines."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v8 import JointSliceFeedbackModel

REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
RESERVATIONS = ROOT / 'data/joint_v7_reserved_train_metadata_001/proposed_donor_reservations.jsonl'
PARENT = ROOT / 'runs/joint_v8_feedback_pilot_001/joint_step_36000.pt'
TAIL = ROOT / 'runs/joint_v8_million_direct_tail_001'
DIRECT = TAIL / 'joint_step_161000.pt'
OUTPUT = ROOT / 'runs/joint_v8_million_direct_real_dev_001'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

done = json.loads((TAIL / 'completed.json').read_text())
assert done['stage_updates'] == 125000 and done['unique_eligible_optimizer_synthetic_planes'] == 1000000
assert done['replayed_tail_updates'] == 5000 and done['max_reference_loss_gap'] < .01
real_done = json.loads((REAL / 'completed.json').read_text())
assert real_done['development_images'] == 64 and real_done['development_donors'] == 6
records = [json.loads(line) for line in (REAL / 'records.jsonl').read_text().splitlines()]
dev = [row for row in records if row['training_split'] == 'development']
reservations = [json.loads(line) for line in RESERVATIONS.read_text().splitlines()]
reserved_train = {int(row['animal_id']) for row in reservations if row['proposed_split'] == 'train'}
assert len(dev) == 64 and len({row['animal_id'] for row in dev}) == 6
assert not {row['animal_id'] for row in dev} & reserved_train
assert all(row['label_role'].startswith('weak Allen') for row in dev)
images = np.load(REAL / 'images.npy', mmap_mode='r')
with np.load(REAL / 'geometry.npz') as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    is_train = arrays['is_train'].copy()
assert images.shape == (1344, 1, 192, 192)
assert all(not is_train[row['array_row_index']] for row in dev)

hashes = {}
for path in (PARENT, DIRECT, TAIL / 'completed.json', REAL / 'images.npy', REAL / 'geometry.npz',
             REAL / 'records.jsonl', REAL / 'preparation_source.py',
             REAL / 'completed.json', RESERVATIONS, Path(__file__)):
    with path.open('rb') as stream:
        hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
for name, expected in real_done['output_sha256'].items():
    assert hashes[str(REAL / name)] == expected

OUTPUT.mkdir(parents=True, exist_ok=False)
protocol = {'input_sha256': hashes, 'checkpoints': {'parent': str(PARENT), 'direct': str(DIRECT)},
            'readout': 'paired whole-model direct prediction; maximum prior mode/reflection versus best-of-16 reference oracle',
            'metric': 'mean physical error at five full-canvas pixels, and antipodal normal angle',
            'aggregation': 'donor-equal means over six development donors; parent/direct paired by section',
            'reference': 'upstream Allen affine, weak and not blinded expert anatomical truth',
            'scope': 'internal real development only; not calibration, final test, or public benchmark'}
(OUTPUT / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
pixels = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.], [95.5, 95.5]])
reflected = pixels.expand(2, -1, -1).clone()
reflected[1, :, 0] = 191 - reflected[1, :, 0]
uv = reflected / 192 - .5
rows = []
for stage, path, expected_step in (('parent', PARENT, 36000), ('direct', DIRECT, 161000)):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['step'] == expected_step and not checkpoint['calibrated']
    model = JointSliceFeedbackModel(modes=checkpoint['config']['modes']).cuda().eval()
    model.load_state_dict(checkpoint['model_state'], strict=True)
    del checkpoint
    for index, record in enumerate(dev):
        row_index = record['array_row_index']
        image = np.concatenate((images[row_index].astype(np.float32),
                                np.zeros((4, 192, 192), dtype=np.float32)))
        with torch.inference_mode():
            result = model.predict(torch.from_numpy(image[None]).cuda())
        state = result['state'][0].cpu()
        centre, frame, basis = full_frame_state_to_components(state)
        tangent = frame[..., :2] @ basis
        predicted = centre[:, None, None] + torch.einsum('mij,rpj->mrpi', tangent, uv)
        affine = torch.as_tensor(affines[row_index], dtype=predicted.dtype)
        reference = affine[:, 2] + pixels[:, :1] * affine[:, 0] + pixels[:, 1:] * affine[:, 1]
        error = (predicted - reference).norm(dim=-1).mean(-1)
        prior = result['log_mass'][0].cpu()[:, None] + torch.stack((
            F.logsigmoid(-result['reflection_logit'][0].cpu()),
            F.logsigmoid(result['reflection_logit'][0].cpu())), -1)
        selected_mode, selected_reflection = divmod(int(prior.flatten().argmax()), 2)
        reference_normal = torch.linalg.cross(affine[:, 0], affine[:, 1])
        reference_normal = F.normalize(reference_normal, dim=0)
        angle = torch.rad2deg((frame[..., 2] * reference_normal).sum(-1).abs().clamp(0, 1).acos())
        save = OUTPUT / f'prediction_{stage}_{index:02d}.npz'
        np.savez_compressed(save, state=state.numpy(), prior=prior.numpy(),
                            coordinate_error_um=error.numpy(), normal_angle_deg=angle.numpy(),
                            std=result['std'][0].cpu().numpy(),
                            concentration=result['concentration'][0].cpu().numpy())
        with save.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        rows.append({'stage': stage, 'animal_id': record['animal_id'],
                     'specimen_id': record['specimen_id'], 'experiment_id': record['experiment_id'],
                     'section_id': record['section_id'], 'prediction_file': save.name,
                     'prediction_sha256': digest,
                     'selected_component': [selected_mode, selected_reflection],
                     'selected_five_point_um': float(error[selected_mode, selected_reflection]),
                     'oracle_five_point_um': float(error.min()),
                     'selected_normal_deg': float(angle[selected_mode])})
    print(json.dumps({'stage': stage, 'development_sections': len(dev)}), flush=True)
    del model
(OUTPUT / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
summary = {}
for stage in ('parent', 'direct'):
    donors = {}
    for donor in sorted({row['animal_id'] for row in dev}):
        sample = [row for row in rows if row['stage'] == stage and row['animal_id'] == donor]
        donors[str(donor)] = {'sections': len(sample), **{
            key: float(np.mean([row[key] for row in sample])) for key in
            ('selected_five_point_um', 'oracle_five_point_um', 'selected_normal_deg')}}
    summary[stage] = {'donors': donors, 'donor_equal': {key: float(np.mean([
        value[key] for value in donors.values()])) for key in
        ('selected_five_point_um', 'oracle_five_point_um', 'selected_normal_deg')}}
with (OUTPUT / 'rows.jsonl').open('rb') as stream:
    rows_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
(OUTPUT / 'completed.json').write_text(json.dumps({'rows': len(rows), 'rows_sha256': rows_sha,
    'summary': summary, 'scope': protocol['scope'], 'calibrated': False}, indent=2), encoding='utf8')
print(json.dumps({name: item['donor_equal'] for name, item in summary.items()}, indent=2), flush=True)
