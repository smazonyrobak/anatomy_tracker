"""Frozen arbitrary-plane and weak-real readout of the 16-mode pose continuation."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

run = root / 'runs/one_shot_pose_capture_003'
panel = root / 'data/one_shot_native256_synthetic_dev_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_pose_capture_003_development_eval'
steps = (0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 40000
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert len(records) == 128 and len({record['panel_physical_section_id'] for record in records}) == 128
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(real_records) == 64 and len({record['animal_id'] for record in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
corner = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                       [127.5, 127.5]], device='cuda') / side
flags = torch.tensor([0, 1], device='cuda')[None].expand(16, -1)


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


out.mkdir(parents=True, exist_ok=False)
rows = []
checkpoint_hashes = {}
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        path = run / f'joint_step_{step:05d}.pt'
        checkpoint_hashes[str(step)] = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True).cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in records:
            row = {'step': step, 'set': 'synthetic', 'eligible': record['eligible'],
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                                    'section_id', 'panel_physical_section_id',
                                                    'appearance_mode', 'valid_pixels', 'sha256')}}
            if record['eligible']:
                with np.load(panel / record['file'], allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                    target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                    valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                    true_reflection = int(arrays['reflection'])
                prediction = model.predict(image)
                states = prediction['state'][0, :, None].expand(-1, 2, -1)
                indices = valid.flatten().nonzero().flatten()
                xy = torch.stack((indices.remainder(side),
                                  indices.div(side, rounding_mode='floor')), -1).float() / side
                reference = target.reshape(-1, 3)[indices]
                tissue = (points(states, flags, xy) - reference).norm(dim=-1).mean(-1)
                true_five = points(truth, torch.tensor([true_reflection], device='cuda'), corner)[0]
                five = (points(states, flags, corner) - true_five).norm(dim=-1).mean(-1)
                mass = prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)
                chosen = int(mass.flatten().argmax())
                truth_normal = full_frame_state_to_components(truth)[1][0, :, 2]
                normals = full_frame_state_to_components(prediction['state'][0])[1][..., :, 2]
                angles = torch.rad2deg(torch.acos((normals * truth_normal).sum(-1).abs().clamp(0, 1)))
                row.update(selected_mode=chosen // 2, selected_reflection=chosen % 2,
                           true_reflection=true_reflection,
                           selected_tissue_um=float(tissue.flatten()[chosen]),
                           oracle_tissue_um=float(tissue.min()),
                           selected_five_point_um=float(five.flatten()[chosen]),
                           selected_normal_deg=float(angles[chosen // 2]),
                           oracle_normal_deg=float(angles.min()),
                           branch_tissue_um=tissue.cpu().tolist(), branch_log_mass=mass.cpu().tolist())
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in real_records:
            i = record['array_row_index']
            image = np.concatenate((real_images[i].astype(np.float32),
                                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + 192 * corner[:, :1] * affine[:, 0] + 192 * corner[:, 1:] * affine[:, 1]
            prediction = model.predict(image)
            states = prediction['state'][0, :, None].expand(-1, 2, -1)
            five = (points(states, flags, corner) - reference).norm(dim=-1).mean(-1)
            mass = prediction['log_mass'][0, :, None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'][0]),
                F.logsigmoid(prediction['reflection_logit'][0])), -1)
            chosen = int(mass.flatten().argmax())
            row = {'step': step, 'set': 'real_weak_allen',
                   **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                   'selected_five_point_um': float(five.flatten()[chosen]),
                   'oracle_five_point_um': float(five.min()),
                   'selected_mode': chosen // 2, 'selected_reflection': chosen % 2,
                   'branch_five_point_um': five.cpu().tolist(), 'branch_log_mass': mass.cpu().tolist()}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'rows': len(rows)}), flush=True)
        del model
summary = []
for step in steps:
    for split in ('synthetic', 'real_weak_allen'):
        group = [row for row in rows if row['step'] == step and row['set'] == split]
        scored = [row for row in group if split != 'synthetic' or row['eligible']]
        identities = sorted({row['animal_id'] for row in scored})
        metrics = (('selected_tissue_um', 'oracle_tissue_um', 'selected_five_point_um',
                    'selected_normal_deg', 'oracle_normal_deg') if split == 'synthetic'
                   else ('selected_five_point_um', 'oracle_five_point_um'))
        summary.append({'step': step, 'set': split, 'rows': len(group), 'scored': len(scored),
                        'identities': len(identities),
                        'identity_equal_mean': {name: float(np.mean([
                            np.mean([row[name] for row in scored if row['animal_id'] == identity])
                            for identity in identities])) for name in metrics},
                        'pooled_median': {name: float(np.median([row[name] for row in scored]))
                                          for name in metrics}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'steps': steps, 'rows': len(rows),
    'checkpoint_sha256': checkpoint_hashes,
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
