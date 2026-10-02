"""Matched, animal-equal DEV readout for real-weak spatial-head adaptation."""
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

run = root / 'runs/one_shot_real_coordinate_pose_014'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/one_shot_real_coordinate_pose_014_development_eval'
steps = tuple(range(0, 12001, 2000))
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((run / 'completed.json').read_text())['updates'] == 12000
synthetic_records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
real_records = [row for row in map(json.loads, (real / 'records.jsonl').open())
                if row['training_split'] == 'development']
assert len(synthetic_records) == 256 and len({r['animal_id'] for r in synthetic_records}) == 8
assert len(real_records) == 64 and len({r['animal_id'] for r in real_records}) == 6
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               candidate_ranking=True, fitted_ranking=True,
                               dense_coordinate=True).cuda().eval()
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
cells = torch.arange(16, device='cuda').float() * 16 + 7.5
yy, xx = torch.meshgrid(cells / side, cells / side, indexing='ij')
chart16 = torch.stack((xx, yy), -1).reshape(256, 2)


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    coordinates = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    coordinates[..., 0] = torch.where(reflection[..., None].bool(),
                                       255 / side - coordinates[..., 0], coordinates[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :2] @ basis, coordinates - .5)


out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint_path = run / f'joint_step_{step:05d}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        for record in synthetic_records:
            if not record['eligible']:
                continue
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                reflection = torch.tensor([int(arrays['reflection'])], device='cuda')
            prediction = model.predict(image)
            field = model.center_origin[None, :, None, None] + prediction['dense_coordinate'][:, :3] * model.center_scale[None, :, None, None]
            field = field.flatten(2).transpose(1, 2)[0]
            target = points(truth, reflection, chart16)[0]
            occupancy = F.avg_pool2d(valid[None, None].float(), 16).flatten()
            full_indices = valid.flatten().nonzero().flatten()
            full_chart = torch.stack((full_indices.remainder(side),
                                      full_indices.div(side, rounding_mode='floor')), -1).float() / side
            dense_state = model.dense_coordinate_plane(prediction)
            dense_plane = points(dense_state, torch.zeros(1, device='cuda'), full_chart)[0]
            row = {'set': 'synthetic', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id', 'sha256')},
                   'field_tissue_um': float(((field - target).norm(dim=-1) * occupancy).sum() / occupancy.sum()),
                   'field_all_um': float((field - target).norm(dim=-1).mean()),
                   'dense_plane_tissue_um': float((dense_plane - reference.reshape(-1, 3)[full_indices]).norm(dim=-1).mean()),
                   'valid_fraction': float(valid.float().mean())}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        for record in real_records:
            i = record['array_row_index']
            image = np.concatenate((real_images[i].astype(np.float32),
                                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
            image = F.interpolate(torch.from_numpy(image).cuda(), (side, side),
                                  mode='bilinear', align_corners=False)
            affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
            reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] + 192 * corners[:, 1:] * affine[:, 1]
            prediction = model.predict(image)
            dense_state = model.dense_coordinate_plane(prediction)
            row = {'set': 'real_weak_allen', 'step': step,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                                                   'experiment_id', 'section_id')},
                   'dense_five_um': float((points(dense_state, torch.zeros(1, device='cuda'), corners)[0]
                                           - reference).norm(dim=-1).mean())}
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'synthetic_eligible': sum(r['set'] == 'synthetic' and r['step'] == step for r in rows),
                          'real': len(real_records)}), flush=True)

summary = []
for step in steps:
    for split, names in (('synthetic', ('field_tissue_um', 'field_all_um', 'dense_plane_tissue_um')),
                         ('real_weak_allen', ('dense_five_um',))):
        group = [row for row in rows if row['set'] == split and row['step'] == step]
        identities = sorted({row['animal_id'] for row in group})
        summary.append({'step': step, 'set': split, 'rows': len(group),
                        'identities': len(identities), 'identity_equal_mean': {
                            name: float(np.mean([np.mean([row[name] for row in group
                                if row['animal_id'] == identity]) for identity in identities]))
                            for name in names}})
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'rows': len(rows),
    'checkpoint_sha256': {str(step): hashlib.sha256(
        (run / f'joint_step_{step:05d}.pt').read_bytes()).hexdigest() for step in steps},
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'real_records_sha256': hashlib.sha256((real / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows)}), flush=True)
