"""Fixed TRAIN-only weak-affine retention for the 221 pose-framing comparison."""

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
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_reserved_real_stream_v8 import (
    load_reserved_real_train, sample_reserved_real_train)
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve()
out = root / 'runs/pose_framing_221_old_train_guard_001'
parent_run = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A'
arm_runs = {arm: root / f'runs/pose_framing_221_{arm}_train_001'
            for arm in ('unframed', 'framed')}
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
protocol = source.parents[1] / 'docs/publication/POSE_FRAMING_221_PROTOCOL_20261011.md'
side = 256
assert source.drive.upper() == root.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

runs = {'parent187': parent_run, **arm_runs}
steps = {'parent187': (800,), 'unframed': (200, 400), 'framed': (200, 400)}
done = {name: json.loads((path / 'completed.json').read_text())
        for name, path in runs.items()}
configs = {name: json.loads((path / 'config.json').read_text())
           for name, path in runs.items()}
checkpoints = {(name, step): path / f'joint_step_{step:05d}.pt'
               for name, path in runs.items() for step in steps[name]}
assert all(sha(path) == done[name]['checkpoint_sha256'][str(step)]
           for (name, step), path in checkpoints.items())
assert all(sha(runs[name] / 'config.json') == done[name]['config_sha256']
           for name in runs)
assert all(done[arm]['parent_checkpoint_sha256'] == sha(checkpoints['parent187', 800])
           for arm in arm_runs)
assert all(done[name]['calibrated'] is False and
           done[name]['public_benchmark_used'] is False for name in runs)
assert all(done[arm]['protocol_sha256'] == sha(protocol) for arm in arm_runs)

coronal = load_reserved_real_train()
assert coronal['bindings'] == configs['parent187']['old_coronal_bindings']
assert all(coronal['bindings'] == configs[arm]['old_coronal_bindings'] for arm in arm_runs)
sagittal_summary = json.loads((sagittal_dir / 'summary.json').read_text())
assert all(sha(sagittal_dir / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
assert all(sha(sagittal_dir / 'summary.json') == configs[name]['sagittal_summary_sha256']
           for name in runs)
sagittal_records = [json.loads(line) for line in (sagittal_dir / 'geometry.jsonl').open()]
sagittal_images = np.load(sagittal_dir / 'model_input.npy', mmap_mode='r')
assert sagittal_summary['roles'] == {'train': {'donors': 40, 'sections': 653}}
assert len(sagittal_records) == len(sagittal_images) == 653
assert all(record['split'] == 'train' for record in sagittal_records)

affine = torch.tensor(np.asarray([row['model_pixel_to_ccf_ref9_ap_dv_ml_um']
    for row in sagittal_records]), dtype=torch.float64)
ouv = torch.stack((affine[:, :, 2], side * affine[:, :, 0],
                   side * affine[:, :, 1]), 1)
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
normal = full_frame_state_to_components(sagittal_states)[1][..., :, 2]
sagittal_reflection = normal.gather(1, normal.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sagittal_reflection, 0] += (side - 1) / side * ouv[sagittal_reflection, 1]
ouv[sagittal_reflection, 1] *= -1
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()

five = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                     [127.5, 127.5]], device='cuda') / side
coronal_donors = np.linspace(0, len(coronal['donors']) - 1, 48, dtype=int)
sagittal_donors = {}
for index, record in enumerate(sagittal_records):
    sagittal_donors.setdefault(record['donor_id'], []).append(index)
assert len(set(coronal_donors)) == 48 and len(sagittal_donors) == 40
cases, images, targets = [], [], []
with torch.inference_mode():
    for donor_index in coronal_donors:
        donor = coronal['donors'][int(donor_index)]
        section_index = len(donor['identities']) // 2
        old = sample_reserved_real_train(coronal, int(donor_index), [section_index])
        images.append(F.interpolate(old['inputs'], (side, side),
                                    mode='bilinear', align_corners=False))
        targets.append(rigid_points_090(old['state'], old['reflection'], five))
        cases.append({'cohort': 'old_coronal', 'donor_id': donor['animal_id'],
            'section_index': section_index, **old['identities'][0],
            'source_array_sha256': coronal['bindings'][str(donor['images'])]})
    for donor_id, indices in sorted(sagittal_donors.items()):
        index = indices[len(indices) // 2]
        image = torch.zeros(1, 5, side, side, device='cuda')
        image[:, :1] = torch.from_numpy(np.asarray(
            sagittal_images[index:index + 1]).copy()).to('cuda')
        images.append(image)
        targets.append(rigid_points_090(sagittal_states[index:index + 1].to('cuda'),
            sagittal_reflection[index:index + 1].long().to('cuda'), five))
        record = sagittal_records[index]
        cases.append({'cohort': 'old_sagittal', 'donor_id': donor_id,
            'section_index': index, **{key: record[key] for key in
            ('specimen_id', 'experiment_id', 'section_id', 'image_sha256')}})
    images = torch.cat(images)
    targets = torch.cat(targets)

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
flags = torch.arange(2, device='cuda')[None, None]
rows = []
with torch.inference_mode():
    for (name, step), path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['step'] == step and saved['config'] == configs[name]
        assert saved['calibrated'] is False
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for start in range(0, len(cases), 8):
            prediction = model.predict(images[start:start + 8])
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(states), model.modes, 2)
            proposed = rigid_points_090(states, reflections, five)
            error = (proposed - targets[start:start + 8, None, None]).norm(
                dim=-1).mean(-1).flatten(1) / 1000
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            chosen = prior.argmax(-1)
            for offset in range(len(states)):
                branch = int(chosen[offset])
                rows.append({'model': name, 'step': step, **cases[start + offset],
                    'top1_branch_id': branch,
                    'top1_weak_five_point_mm': float(error[offset, branch]),
                    'best160_weak_five_point_mm': float(error[offset].min())})
        print(json.dumps({'model': name, 'step': step, 'cases': len(cases)}), flush=True)

summary = {}
for cohort in ('old_coronal', 'old_sagittal'):
    summary[cohort] = {}
    for name in runs:
        for step in steps[name]:
            subset = [row for row in rows if row['cohort'] == cohort and
                      row['model'] == name and row['step'] == step]
            values = np.asarray([row['top1_weak_five_point_mm'] for row in subset])
            donors = sorted({row['donor_id'] for row in subset})
            donor_means = {donor: float(np.mean([row['top1_weak_five_point_mm']
                for row in subset if row['donor_id'] == donor])) for donor in donors}
            summary[cohort][f'{name}_{step}'] = {
                'sections': len(subset), 'donors': len(donors),
                'top1_mean_mm': float(values.mean()),
                'top1_median_mm': float(np.median(values)),
                'top1_p90_mm': float(np.percentile(values, 90)),
                'top1_donor_equal_mean_mm': float(np.mean(list(donor_means.values()))),
                'best160_mean_mm': float(np.mean([
                    row['best160_weak_five_point_mm'] for row in subset])),
                'top1_by_donor_mean_mm': donor_means}
    parent = summary[cohort]['parent187_800']['top1_donor_equal_mean_mm']
    for name in arm_runs:
        for step in steps[name]:
            summary[cohort][f'{name}_{step}']['top1_donor_equal_minus_parent_mm'] = (
                summary[cohort][f'{name}_{step}']['top1_donor_equal_mean_mm'] - parent)

out.mkdir(parents=True, exist_ok=False)
config = {'evaluator_source_sha256': sha(source), 'protocol_sha256': sha(protocol),
    'checkpoint_sha256': {f'{name}_{step}': sha(path)
        for (name, step), path in checkpoints.items()},
    'training_completion_sha256': {name: sha(path / 'completed.json')
        for name, path in runs.items()},
    'coronal_train_manifest_sha256': coronal['manifest_sha256'],
    'coronal_train_bindings_sha256': hashlib.sha256(json.dumps(
        coronal['bindings'], sort_keys=True).encode()).hexdigest(),
    'sagittal_train_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'sagittal_train_output_sha256': sagittal_summary['output_sha256'],
    'fixed_selection': '48 evenly spaced sorted coronal TRAIN donors and all 40 sagittal TRAIN donors; midpoint section per donor',
    'metric': 'top-prior and best-of-160 mean five-point CCF disagreement against automated weak Allen affine, mm',
    'scope': 'TRAIN-only retention diagnostic, not independent anatomical ground truth',
    'calibrated': False, 'public_benchmark_used': False,
    'development_calibration_final_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({'cases': len(cases), 'rows': len(rows),
    'output_sha256': {name: sha(out / name)
        for name in ('config.json', 'summary.json', 'rows.jsonl')},
    'calibrated': False, 'public_benchmark_used': False,
    'development_calibration_final_used': False}, indent=2, allow_nan=False) + '\n')
print(json.dumps({'event': 'old_train_guard_complete', 'cases': len(cases),
                  'rows': len(rows)}), flush=True)
