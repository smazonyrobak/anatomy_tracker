"""Post-exit native-256 tissue-pose and weak-real readout of every saved checkpoint."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

RUN = ROOT / 'runs/one_shot_joint_physical_pose_002'
SYN = ROOT / 'data/one_shot_native256_synthetic_dev_001'
REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
OUT = ROOT / 'runs/one_shot_joint_physical_pose_002_development_eval'
STEPS = (0, 2000, 4000, 6000, 8000)
SIDE = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert json.loads((RUN / 'completed.json').read_text())['updates'] == 8000


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / SIDE - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


panel = json.loads((SYN / 'completed.json').read_text())
assert panel['physical_sections'] == 128 and panel['animals'] == 4
assert sha(SYN / 'records.jsonl') == panel['records_sha256']
synthetic = [json.loads(line) for line in (SYN / 'records.jsonl').read_text().splitlines()]
assert len(synthetic) == 128 and len({row['panel_physical_section_id'] for row in synthetic}) == 128
for row in synthetic:
    assert sha(SYN / row['file']) == row['sha256']
all_real = [json.loads(line) for line in (REAL / 'records.jsonl').read_text().splitlines()]
real = [row for row in all_real if row['training_split'] == 'development']
assert len(real) == 64 and len({row['animal_id'] for row in real}) == 6
assert not {row['animal_id'] for row in real} & {row['animal_id'] for row in all_real if row['training_split'] == 'train'}
real_images = np.load(REAL / 'images.npy', mmap_mode='r')
with np.load(REAL / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
corner = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                       [127.5, 127.5]], device='cuda') / SIDE
flags = torch.tensor([0, 1], device='cuda')[None].expand(8, -1)
OUT.mkdir(parents=True, exist_ok=False)
rows, checkpoint_hashes = [], {}
with (OUT / 'rows.jsonl').open('w') as stream:
    for step in STEPS:
        path = RUN / f'joint_step_{step:05d}.pt'
        checkpoint_hashes[str(step)] = sha(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model = OneShotJointSliceModel(atlas_conditioning=True).cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        with torch.inference_mode():
            for record in synthetic:
                row = {'step': step, 'set': 'synthetic',
                       **{key: record[key] for key in
                          ('animal_id', 'specimen_id', 'experiment_id', 'section_id',
                           'panel_physical_section_id', 'appearance_mode')},
                       'eligible': record['eligible'], 'valid_pixels': record['valid_pixels'],
                       'input_sha256': record['sha256']}
                if record['eligible']:
                    with np.load(SYN / record['file'], allow_pickle=False) as arrays:
                        image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                        truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                        target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                        valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                        true_reflection = int(arrays['reflection'])
                    prediction = model.predict(image)
                    states = prediction['state'][0, :, None].expand(-1, 2, -1)
                    indices = valid.flatten().nonzero().flatten()
                    xy = torch.stack((indices.remainder(SIDE),
                        indices.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
                    reference = target.reshape(-1, 3)[indices]
                    tissue = (points(states, flags, xy) - reference).norm(dim=-1).mean(-1)
                    truth_five = points(truth, torch.tensor([true_reflection], device='cuda'), corner)[0]
                    five = (points(states, flags, corner) - truth_five).norm(dim=-1).mean(-1)
                    mass = prediction['log_mass'][0, :, None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit'][0]),
                        F.logsigmoid(prediction['reflection_logit'][0])), -1)
                    selected = int(mass.flatten().argmax())
                    truth_normal = full_frame_state_to_components(truth)[1][0, :, 2]
                    normals = full_frame_state_to_components(prediction['state'][0])[1][..., :, 2]
                    angle = torch.rad2deg(torch.acos(
                        (normals * truth_normal).sum(-1).abs().clamp(0, 1)))
                    row.update(selected_mode=selected // 2, selected_reflection=selected % 2,
                               true_reflection=true_reflection,
                               selected_tissue_um=float(tissue.flatten()[selected]),
                               oracle_tissue_um=float(tissue.min()),
                               selected_five_point_um=float(five.flatten()[selected]),
                               selected_normal_deg=float(angle[selected // 2]),
                               oracle_normal_deg=float(angle.min()),
                               branch_tissue_um=tissue.cpu().tolist(),
                               branch_log_mass=mass.cpu().tolist())
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
            for record in real:
                i = record['array_row_index']
                image = np.concatenate((real_images[i].astype(np.float32),
                                        np.zeros((4, 192, 192), dtype=np.float32)))[None]
                image = F.interpolate(torch.from_numpy(image).cuda(), (SIDE, SIDE),
                                      mode='bilinear', align_corners=False)
                affine = torch.as_tensor(affines[i], device='cuda', dtype=torch.float32)
                reference = affine[:, 2] + 192 * corner[:, :1] * affine[:, 0] + 192 * corner[:, 1:] * affine[:, 1]
                prediction = model.predict(image)
                states = prediction['state'][0, :, None].expand(-1, 2, -1)
                five = (points(states, flags, corner) - reference).norm(dim=-1).mean(-1)
                mass = prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)
                selected = int(mass.flatten().argmax())
                row = {'step': step, 'set': 'real_weak_allen',
                       **{key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                       'selected_five_point_um': float(five.flatten()[selected]),
                       'oracle_five_point_um': float(five.min()),
                       'selected_mode': selected // 2, 'selected_reflection': selected % 2,
                       'branch_five_point_um': five.cpu().tolist(),
                       'branch_log_mass': mass.cpu().tolist()}
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'evaluated_rows': len(rows)}), flush=True)
        del model
summary = []
for step in STEPS:
    for split in ('synthetic', 'real_weak_allen'):
        items = [row for row in rows if row['step'] == step and row['set'] == split]
        scored = [row for row in items if split != 'synthetic' or row['eligible']]
        keys = (('selected_tissue_um', 'oracle_tissue_um', 'selected_five_point_um',
                 'selected_normal_deg', 'oracle_normal_deg') if split == 'synthetic'
                else ('selected_five_point_um', 'oracle_five_point_um'))
        animals = sorted({row['animal_id'] for row in scored})
        summary.append({'step': step, 'set': split, 'raw_sections': len(items),
            'scored_sections': len(scored), 'identities': len(animals),
            'identity_equal_mean': {key: float(np.mean([
                np.mean([row[key] for row in scored if row['animal_id'] == animal])
                for animal in animals])) for key in keys},
            'pooled_mean': {key: float(np.mean([row[key] for row in scored])) for key in keys}})
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'completed.json').write_text(json.dumps({'steps': STEPS, 'rows': len(rows),
    'checkpoint_sha256': checkpoint_hashes, 'panel_records_sha256': panel['records_sha256'],
    'evaluator_sha256': sha(__file__), 'rows_sha256': sha(OUT / 'rows.jsonl'),
    'summary_sha256': sha(OUT / 'summary.json'), 'calibrated': False,
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'rows': len(rows), 'output': str(OUT)}), flush=True)
