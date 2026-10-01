"""Post-exit fixed256px readout: all11 checkpoints, no checkpoint selection."""
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'
sys.dont_write_bytecode = True
READY_AFTER_CONFIRMED_EXIT = True
assert READY_AFTER_CONFIRMED_EXIT, 'Do not read the mixed run until PID24036 has exited and root authorizes evaluation'

import hashlib
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

RUN = ROOT / 'runs/one_shot_joint_mixed_real_003'
DEV = ROOT / 'data/joint_v7_synthetic_dev192_001'
REAL = ROOT / 'data/joint_v7_allen_fullcanvas_192_001'
OUT = ROOT / 'runs/one_shot_joint_mixed_real_003_development_eval'
STEPS = tuple(range(0, 20001, 2000))
assert json.loads((RUN / 'completed.json').read_text())['updates'] == 20000
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
synthetic = [json.loads(line) for line in (DEV / 'records.jsonl').read_text().splitlines()]
all_real = [json.loads(line) for line in (REAL / 'records.jsonl').read_text().splitlines()]
real = [row for row in all_real if row['training_split'] == 'development']
assert len(synthetic) == len(real) == 64
assert len({r['animal_id'] for r in synthetic}) == 4 and len({r['animal_id'] for r in real}) == 6
assert not {r['animal_id'] for r in real} & {r['animal_id'] for r in all_real if r['training_split'] == 'train'}
real_images = np.load(REAL / 'images.npy', mmap_mode='r')
with np.load(REAL / 'geometry.npz', allow_pickle=False) as arrays:
    real_affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
uv = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.], [127.5, 127.5]], device='cuda') / 256
hashes = {}


def sha(path):
    with Path(path).open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    hashes[str(path)] = value
    return value


def five_points(state, reflection):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = uv.expand(*state.shape[:-1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(), 255 / 256 - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)


for path in (Path(__file__), RUN / 'completed.json', DEV / 'records.jsonl', REAL / 'records.jsonl', REAL / 'geometry.npz'):
    sha(path)
prepared = []
for record in synthetic:
    path = DEV / record['file']
    assert sha(path) == record['sha256']
    appearance = record['section_index'] % 3
    with np.load(path, allow_pickle=False) as arrays:
        prepared.append({'set': 'synthetic', 'record': record, 'appearance': appearance,
            'eligible': bool(arrays['eligible'][appearance]),
            'image': torch.from_numpy(arrays['inputs'][appearance:appearance+1].copy()),
            'truth': torch.from_numpy(arrays['target_state'][None].copy()),
            'reflection': int(arrays['reflection']),
            'target': torch.from_numpy(arrays['target_centre_um'][None].copy()),
            'valid': torch.from_numpy((arrays['visible_support'][appearance:appearance+1] > .25).copy()),
            'offsets': torch.from_numpy(arrays['offsets_um'][None].copy())})
for record in real:
    i = record['array_row_index']
    image = np.concatenate((real_images[i].astype(np.float32), np.zeros((4, 192, 192), dtype=np.float32)))
    prepared.append({'set': 'real_weak_allen', 'record': record, 'image': torch.from_numpy(image[None]),
                     'affine': torch.as_tensor(real_affines[i], dtype=torch.float32)})
for item in prepared:
    item['input_sha256'] = hashlib.sha256(item['image'].numpy().tobytes()).hexdigest()
rows = []
started = time.perf_counter()
with (OUT / 'rows.jsonl').open('w', encoding='utf-8') as output:
    for step in STEPS:
        path = RUN / f'joint_step_{step:05d}.pt'
        checkpoint_sha = sha(path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['step'] == step and not saved['calibrated']
        model = OneShotJointSliceModel().cuda().eval()
        model.load_state_dict(saved['model'], strict=True)
        assert model.modes == 8
        with torch.inference_mode():
            for item in prepared:
                record, split = item['record'], item['set']
                image = F.interpolate(item['image'].cuda(), (256, 256), mode='bilinear', align_corners=False)
                if split == 'synthetic':
                    truth = item['truth'].cuda()
                    reflected = torch.tensor([item['reflection']], device='cuda')
                    reference = five_points(truth, reflected)[0]
                    normal = full_frame_state_to_components(truth)[1][0, :, 2]
                else:
                    affine = item['affine'].cuda()
                    reference = affine[:, 2] + 192 * uv[:, :1] * affine[:, 0] + 192 * uv[:, 1:] * affine[:, 1]
                    normal = F.normalize(torch.linalg.cross(affine[:, 0], affine[:, 1]), dim=0)
                prediction = model.predict(image)
                states = prediction['state'][0, :, None].expand(-1, 2, -1)
                candidate = five_points(states, torch.tensor([[0, 1]], device='cuda').expand(8, -1))
                errors = (candidate-reference).norm(dim=-1).mean(-1)
                rlp = torch.stack((F.logsigmoid(-prediction['reflection_logit'][0]), F.logsigmoid(prediction['reflection_logit'][0])), -1)
                joint_lp = prediction['log_mass'][0, :, None] + rlp
                selected = int(joint_lp.flatten().argmax()); selected_mode, selected_r = divmod(selected, 2)
                predicted_r = rlp.argmax(-1)
                oracle_mode_pred_r = int(errors[torch.arange(8, device='cuda'), predicted_r].argmin())
                oracle = int(errors.flatten().argmin())
                policies = {'deployed': (selected_mode, selected_r),
                    'selected_mode_oracle_reflection': (selected_mode, int(errors[selected_mode].argmin())),
                    'oracle_mode_predicted_reflection': (oracle_mode_pred_r, int(predicted_r[oracle_mode_pred_r])),
                    'full_oracle': divmod(oracle, 2)}
                normals = full_frame_state_to_components(prediction['state'][0])[1][..., :, 2]
                angle = torch.rad2deg(torch.atan2(torch.linalg.cross(normals, normal[None]).norm(dim=-1), (normals*normal).sum(-1).abs()))
                assert bool(torch.isfinite(errors).all() and torch.isfinite(angle).all() and torch.isfinite(joint_lp).all())
                row = {'step': step, 'set': split, 'checkpoint_sha256': checkpoint_sha,
                    **{k: record[k] for k in ('animal_id', 'specimen_id', 'experiment_id', 'section_id')},
                    'input_sha256': item['input_sha256'], 'synthetic_eligible': item.get('eligible'),
                    'selected_mode': selected_mode, 'selected_reflection': selected_r,
                    'oracle_normal_angle_deg': float(angle.min()), 'oracle_normal_mode': int(angle.argmin()),
                    'branch_five_point_error_um': errors.cpu().tolist(), 'mode_normal_angle_deg': angle.cpu().tolist(),
                    'branch_log_mass': joint_lp.cpu().tolist(), 'candidate_state': prediction['state'][0].cpu().tolist(),
                    'reference_five_points_um': reference.cpu().tolist(), 'reference_normal': normal.cpu().tolist()}
                for label, (m, r) in policies.items():
                    row[label+'_error_um'] = float(errors[m, r])
                    row[label+'_normal_angle_deg'] = float(angle[m])
                    row[label+'_mode'], row[label+'_reflection'] = m, r
                row['selected_error_um'] = row['deployed_error_um']
                row['oracle_error_um'] = row['full_oracle_error_um']
                row['selected_normal_angle_deg'] = row['deployed_normal_angle_deg']
                if split == 'synthetic':
                    row.update(appearance_index=item['appearance'], appearance_mode=record['modes'][item['appearance']], target_reflection=item['reflection'])
                    if item['eligible']:
                        state = prediction['state'].clone(); state[:, 0] = truth
                        mapped = model.map({**prediction, 'state': state}, item['offsets'].cuda(), torch.zeros((1, 1), dtype=torch.long, device='cuda'), reflected[:, None], (256, 256))
                        target = F.interpolate(item['target'].cuda().permute(0, 3, 1, 2), (256, 256), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                        valid = F.interpolate(item['valid'].cuda()[:, None].float(), (256, 256), mode='nearest')[:, 0] > .5
                        surface, local = mapped['centre_surface_ccf_ap_dv_ml_um'][:, 0], mapped['local_displacement_um'][:, 0]
                        zero = surface - torch.einsum('bij,bjhw->bhwi', full_frame_state_to_components(truth)[1], local)
                        row['teacher_map_error_um'] = float((surface-target).norm(dim=-1)[valid].mean())
                        row['zero_warp_error_um'] = float((zero-target).norm(dim=-1)[valid].mean())
                rows.append(row); output.write(json.dumps(row, allow_nan=False)+'\n')
        output.flush()
        print(json.dumps({'step': step, 'completed_rows': len(rows), 'seconds': time.perf_counter()-started}), flush=True)
        del model, saved
metrics = ['selected_error_um', 'oracle_error_um', 'selected_normal_angle_deg', 'oracle_normal_angle_deg',
    *[f'{policy}_{metric}' for policy in ('deployed', 'selected_mode_oracle_reflection', 'oracle_mode_predicted_reflection', 'full_oracle') for metric in ('error_um', 'normal_angle_deg')],
    'teacher_map_error_um', 'zero_warp_error_um']
summary = []
for step in STEPS:
    for split, stratum in (('synthetic', 'all'), ('synthetic', 'eligible'), ('synthetic', 'censored'), ('real_weak_allen', 'all')):
        subset = [r for r in rows if r['step'] == step and r['set'] == split and (stratum == 'all' or r['synthetic_eligible'] == (stratum == 'eligible'))]
        animals = sorted({r['animal_id'] for r in subset})
        per_animal = {animal: {'rows': sum(r['animal_id'] == animal for r in subset),
            **{k: float(np.mean([r[k] for r in subset if r['animal_id'] == animal and k in r])) for k in metrics if any(r['animal_id'] == animal and k in r for r in subset)}} for animal in animals}
        summary.append({'step': step, 'set': split, 'stratum': stratum, 'rows': len(subset), 'animals': len(animals), 'per_animal': per_animal,
            'animal_macro': {k: float(np.mean([v[k] for v in per_animal.values() if k in v])) for k in metrics if any(k in v for v in per_animal.values())}})
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
receipt = {'checkpoints': list(STEPS), 'rows': len(rows), 'hashes': hashes, 'resolution': 256,
    'geometry': 'Same frozen curriculum-resized readout: five normalized-frame points at0,255/256,127.5/256; image192to256 bilinear align_cornersFalse. Reference retains legacy normalized-chart convention, not exact half-pixel image-resampling correspondence. Plane metrics use fitted frames and antipodal normals, not dense anatomical landmarks.',
    'selection': 'argmax joint KxR mass; policy oracles are evaluation-only. Full-oracle normal at five-point-best branch is separate from independently minimal normal angle.',
    'strata': 'Original saved appearance eligibility; all64synthetic retained, eligible/censored separately. Real64six-donor upstream-affine references remain weak. All summaries animal-equal; teacher mapping eligible synthetic only.',
    'scope': 'No optimization, checkpoint choice, calibration or public benchmark; no reliable uncertainty claim. Teacher mapping uses truth plane/reflection and is not inference accuracy.',
    'elapsed_seconds': time.perf_counter()-started, 'output_sha256': {p.name: sha(p) for p in (OUT/'rows.jsonl', OUT/'summary.json')}}
(OUT / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({'completed_rows': len(rows), 'output': str(OUT), 'seconds': receipt['elapsed_seconds']}), flush=True)
