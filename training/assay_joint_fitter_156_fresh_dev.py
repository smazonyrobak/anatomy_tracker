"""Frozen, plan-disjoint synthetic DEV gate for the actual damped 156 fitter."""

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
from training.joint_anatomy_model_156 import JointAnatomyModel156


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


panel = root / 'data/coarse_atlas_pose_150_fresh_dev_panel'
train = root / 'runs/joint_fitter_155_train151_001'
out = root / 'runs/joint_fitter_156_fresh_dev_assay_001'
assert all(path.drive.upper() == 'I:' for path in (panel, train, out))
assert not out.exists()
panel_done = json.loads((panel / 'completed.json').read_text())
train_done = json.loads((train / 'completed.json').read_text())
checkpoint = train / f"joint_fitter_step_{train_done['selected_step']:05d}.pt"
assert sha(checkpoint) == train_done['checkpoint_sha256'][str(train_done['selected_step'])]
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
records = [row for row in records if row['eligible']]
assert len(records) == panel_done['eligible'] == 243
assert len({row['synthetic_subject_plan_id'] for row in records}) == 8

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
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


out.mkdir(parents=True, exist_ok=False)
config = {'source_sha256': sha(__file__),
    'model_source_sha256': {'155': sha(Path(__file__).with_name('joint_anatomy_model_153.py')),
                            '156': sha(Path(__file__).with_name('joint_anatomy_model_156.py'))},
    'panel_receipt_sha256': sha(panel / 'completed.json'),
    'fitter_completion_sha256': sha(train / 'completed.json'),
    'checkpoint_sha256': sha(checkpoint), 'cases': 'all 243 eligible sections from eight synthetic DEV plans',
    'arms': ['155_full', '156_damped_full'],
    'gate': {'exact_drift_mm_at_most': .25, 'near_gain_mm_at_least': .10,
             'exact_energy_below_wrong_at_least': .70},
    'scope': 'synthetic DEV plans distinct from 155 training, one atlas; not biological animals',
    'calibrated': False, 'final_animals_used': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for arm, model_type in (('155_full', JointAnatomyModel153),
                            ('156_damped_full', JointAnatomyModel156)):
        model = model_type().cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        for record in records:
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
                centre = torch.from_numpy(sample['target_centre_um'].copy()).cuda()
            perturb_seed = int(hashlib.sha256(record['section_id'].encode()).hexdigest()[:15], 16)
            generator = torch.Generator(device='cuda').manual_seed(perturb_seed)
            near = candidate(state, reflection, chart, .35, 1.5, .9, generator)
            wrong = candidate(state, reflection, chart, 1.5, 3., 2.2, generator, wrong=True)
            states = torch.stack((state[0], near[0], wrong[0]))[None]
            flags = reflection[:, None].expand(1, 3)
            output = model(image, atlas, offsets, weights,
                torch.zeros(1, 3, device='cuda', dtype=torch.long), flags,
                state_override=states)
            truth = rigid_points_090(state, reflection, chart)
            initial = rigid_points_090(states, flags, chart[None])
            corrected = rigid_points_090(output['corrected_state'].float(), flags, chart[None])
            initial_mm = ((initial - truth[:, None]).norm(dim=-1).mean(-1)[0] / 1000).tolist()
            corrected_mm = ((corrected - truth[:, None]).norm(dim=-1).mean(-1)[0] / 1000).tolist()
            dense_mm = ((output['surface_ccf_um'][0, :, valid].float()
                         - centre[valid]).norm(dim=-1).mean(-1) / 1000).tolist()
            energy = output['fit_energy_uncalibrated'][0].tolist()
            row = {'arm': arm, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'synthetic_plan_id': record['synthetic_subject_plan_id'],
                'panel_file_sha256': record['sha256'], 'appearance_mode': record['appearance_mode'],
                'valid_fraction': record['valid_pixels'] / (256 * 256),
                'initial_mm': initial_mm, 'corrected_mm': corrected_mm,
                'dense_mm': dense_mm, 'fit_energy_uncalibrated': energy,
                'exact_energy_below_wrong': float(energy[0] < energy[2])}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        del model
        print(json.dumps({'arm_complete': arm, 'cases': len(records)}), flush=True)

summary = {}
for arm in config['arms']:
    group = [row for row in rows if row['arm'] == arm]
    result = {'n': len(group), 'synthetic_plans': len({row['synthetic_plan_id'] for row in group}),
        'exact_drift_mm': float(np.mean([row['corrected_mm'][0] for row in group])),
        'near_initial_mm': float(np.mean([row['initial_mm'][1] for row in group])),
        'near_corrected_mm': float(np.mean([row['corrected_mm'][1] for row in group])),
        'near_dense_mm': float(np.mean([row['dense_mm'][1] for row in group])),
        'exact_energy_below_wrong': float(np.mean([row['exact_energy_below_wrong'] for row in group]))}
    result['near_gain_mm'] = result['near_initial_mm'] - result['near_corrected_mm']
    result['feedback_gate_passed'] = (result['exact_drift_mm'] <= .25
        and result['near_gain_mm'] >= .10 and result['exact_energy_below_wrong'] >= .70)
    summary[arm] = result
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'rows': len(rows), 'calibrated': False, 'final_animals_used': False,
    'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'assay_complete', 'summary': summary}), flush=True)
