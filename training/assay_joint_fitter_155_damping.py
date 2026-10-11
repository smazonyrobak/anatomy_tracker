"""Frozen v4 DEV comparison of 155 fit, atlas intensity, and correction gain."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_153 import JointAnatomyModel153


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


panel = root / 'data/fresh_v4_pose_dev_panel_132'
parent_run = root / 'runs/joint_pose_curriculum_154_train151_001'
train = root / 'runs/joint_fitter_155_train151_001'
out = root / 'runs/joint_fitter_155_damping_dev_assay_001'
assert all(path.drive.upper() == 'I:' for path in (panel, parent_run, train, out))
assert not out.exists()
panel_done = json.loads((panel / 'completed.json').read_text())
parent_done = json.loads((parent_run / 'completed.json').read_text())
train_done = json.loads((train / 'completed.json').read_text())
parent_checkpoint = parent_run / f"joint_pose_step_{parent_done['selected_step']:05d}.pt"
checkpoint = train / f"joint_fitter_step_{train_done['selected_step']:05d}.pt"
assert sha(parent_checkpoint) == parent_done['checkpoint_sha256'][str(parent_done['selected_step'])]
assert sha(checkpoint) == train_done['checkpoint_sha256'][str(train_done['selected_step'])]
assert train_done['parent_checkpoint_sha256'] == sha(parent_checkpoint)
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']

records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
records = [row for row in records if row['eligible']]
plans = sorted({row['synthetic_subject_plan_id'] for row in records})
assert len(plans) == 8 and len(records) == panel_done['eligible'] == 248
chosen = []
for plan in plans:
    group = sorted((row for row in records if row['synthetic_subject_plan_id'] == plan),
        key=lambda row: hashlib.sha256(row['section_id'].encode()).hexdigest())
    chosen.extend(group[:8])
assert len(chosen) == 64

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
atlas_zero = atlas.clone()
atlas_zero[0] = 0
model = JointAnatomyModel153().cuda().eval().requires_grad_(False)
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device='cuda')


def candidate(state, reflection, chart, lower, upper, target, generator, wrong=False):
    truth = rigid_points_090(state, reflection, chart)
    while True:
        delta = (2 * torch.rand(64, 9, device='cuda', generator=generator) - 1) * limits
        if wrong:
            delta[:, :6] *= 3
            delta[:, 6:] *= 2
        states = compose_full_frame_state(state.expand(64, -1), delta)
        errors = (rigid_points_090(states, reflection.expand(64), chart)
                  - truth).norm(dim=-1).mean(-1) / 1000
        accepted = (errors >= lower) & (errors <= upper)
        if bool(accepted.any()):
            index = int((errors - target).abs().masked_fill(~accepted, 10).argmin())
            return states[index:index + 1]


cases = []
with torch.inference_mode():
    for record in chosen:
        file = panel / record['file']
        assert sha(file) == record['sha256']
        with np.load(file, allow_pickle=False) as sample:
            valid = torch.from_numpy(sample['valid_mask']).cuda().bool()
            pixels = valid.flatten().nonzero()[:, 0]
            pixels = pixels[torch.linspace(0, len(pixels) - 1, min(1024, len(pixels)),
                device='cuda').long()]
            chart = torch.stack((pixels.remainder(256),
                pixels.div(256, rounding_mode='floor')), -1).float() / 256
            state = torch.from_numpy(sample['target_state'][None].copy()).cuda()
            reflection = torch.as_tensor(sample['reflection'], device='cuda').reshape(1)
            image = torch.from_numpy(sample['inputs'][None].copy()).cuda()
            offsets = torch.from_numpy(sample['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(sample['weights'][None].copy()).cuda()
        perturb_seed = int(hashlib.sha256(record['section_id'].encode()).hexdigest()[:15], 16)
        generator = torch.Generator(device='cuda').manual_seed(perturb_seed)
        near = candidate(state, reflection, chart, .35, 1.5, .9, generator)
        wrong = candidate(state, reflection, chart, 1.5, 3., 2.2, generator, wrong=True)
        states = torch.stack((state[0], near[0], wrong[0]))[None]
        flags = reflection[:, None].expand(1, 3)
        cases.append((record, image, state, reflection, offsets, weights, chart, states, flags))

out.mkdir(parents=True, exist_ok=False)
config = {'source_sha256': sha(__file__), 'model_source_sha256':
    sha(Path(__file__).with_name('joint_anatomy_model_153.py')),
    'panel_receipt_sha256': sha(panel / 'completed.json'),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'fitter_completion_sha256': sha(train / 'completed.json'),
    'checkpoint_sha256': {'154_parent': sha(parent_checkpoint),
                          '155_selected': sha(checkpoint)},
    'selection': 'first eight SHA256(section_id) among eligible sections per synthetic DEV plan',
    'arms': ['154_parent_full', '155_full', '155_atlas_intensity_zeroed'],
    'posthoc_geometric_correction_gains': [1.0, 0.8],
    'scope': 'reused synthetic DEV plans, no biological-animal or public benchmark truth',
    'calibrated': False, 'final_animals_used': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for arm, path, volume in (
        ('154_parent_full', parent_checkpoint, atlas),
        ('155_full', checkpoint, atlas),
        ('155_atlas_intensity_zeroed', checkpoint, atlas_zero),
    ):
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for record, image, state, reflection, offsets, weights, chart, states, flags in cases:
            output = model(image, volume, offsets, weights,
                torch.zeros(1, 3, device='cuda', dtype=torch.long), flags,
                state_override=states)
            truth = rigid_points_090(state, reflection, chart)
            initial = rigid_points_090(states, flags, chart[None])
            corrected = rigid_points_090(output['corrected_state'].float(), flags, chart[None])
            errors = {}
            for gain in (1.0, 0.8):
                blended = initial + gain * (corrected - initial)
                errors[str(gain)] = ((blended - truth[:, None]).norm(dim=-1)
                                     .mean(-1)[0] / 1000).tolist()
            initial_error = ((initial - truth[:, None]).norm(dim=-1)
                             .mean(-1)[0] / 1000).tolist()
            energy = output['fit_energy_uncalibrated'][0].tolist()
            row = {'arm': arm, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'synthetic_plan_id': record['synthetic_subject_plan_id'],
                'panel_file_sha256': record['sha256'],
                'appearance_mode': record['appearance_mode'],
                'valid_fraction': float(record['valid_fraction']) if 'valid_fraction' in record
                                  else float(record['valid_pixels'] / (256 * 256))
                                  if 'valid_pixels' in record else None,
                'initial_mm': initial_error, 'corrected_mm_by_gain': errors,
                'fit_energy_uncalibrated': energy,
                'exact_energy_below_wrong': float(energy[0] < energy[2])}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        print(json.dumps({'arm_complete': arm, 'cases': len(chosen)}), flush=True)

summary = {}
for arm in config['arms']:
    group = [row for row in rows if row['arm'] == arm]
    summary[arm] = {'n': len(group), 'synthetic_plans': len(plans)}
    for gain in ('1.0', '0.8'):
        exact = np.mean([row['corrected_mm_by_gain'][gain][0] for row in group])
        near = np.mean([row['corrected_mm_by_gain'][gain][1] for row in group])
        baseline = np.mean([row['initial_mm'][1] for row in group])
        summary[arm][f'exact_drift_mm_gain_{gain}'] = float(exact)
        summary[arm][f'near_corrected_mm_gain_{gain}'] = float(near)
        summary[arm][f'near_gain_mm_gain_{gain}'] = float(baseline - near)
    summary[arm]['exact_energy_below_wrong'] = float(np.mean(
        [row['exact_energy_below_wrong'] for row in group]))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'rows': len(rows), 'calibrated': False, 'final_animals_used': False,
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'assay_complete', 'summary': summary}), flush=True)
