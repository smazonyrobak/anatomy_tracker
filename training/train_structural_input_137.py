"""Matched 132 replay: fixed structural channels are the sole treatment."""

import hashlib
import json
import math
import os
import sys
import time
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
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_reserved_real_stream_v8 import (
    load_reserved_real_train, sample_reserved_real_train)
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.structural_input_137 import fill_structure


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/STRUCTURAL_INPUT_137_PROTOCOL_20261010.md'
parent_dir = root / 'runs/joint_in_path_correspondence_128_treatment'
parent_path = parent_dir / 'joint_step_06000.pt'
control_dir = root / 'runs/v4_pose_adaptation_132'
control_path = control_dir / 'joint_step_00000.pt'
run = root / 'runs/structural_input_137'
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
side, sites, updates = 256, 128, 2000
checkpoints = (0, 2000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()

parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
control_done = json.loads((control_dir / 'completed.json').read_text())
control_config = json.loads((control_dir / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['6000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert parent_config['arm'] == 'treatment'
assert sha(control_dir / 'config.json') == control_done['config_sha256']
assert sha(control_dir / 'draws.jsonl') == control_done['draws_sha256']
assert sha(control_dir / 'training.jsonl') == control_done['training_sha256']
assert sha(control_path) == control_done['checkpoint_sha256']['0']
assert control_done['parent_checkpoint_sha256'] == control_config['parent_checkpoint_sha256'] == sha(parent_path)
assert control_config['updates'] == control_done['updates'] == updates
assert control_config['synthetic_per_update'] == {'v4': 2, 'v3': 1}
assert control_config['real_weak_per_update'] == {'coronal': 1, 'sagittal': 1}
assert control_config['seed'] == 20261010132 and control_config['valid_sites'] == sites
assert control_done['unique_synthetic_physical_sections'] == 6000
assert all(sha(source / name) == digest for name, digest in
           control_config['source_sha256'].items())
assert all(sha(source / name) == digest for name, digest in
           parent_config['source_sha256'].items())
assert not any(control_done.get(key, False) or parent_done.get(key, False) for key in
    ('calibrated', 'expert_real_truth_used', 'public_benchmark_used',
     'external_pretrained_weights_used'))

context = load_streaming_synthetic_v7_64(device='cuda')
coronal = load_reserved_real_train()
assert json.loads(json.dumps(context['provenance'])) == control_config['synthetic_provenance']
assert coronal['bindings'] == control_config['coronal_bindings']
assert len(context['bases']) == 64 and len(context['subjects']) == 4096
sagittal_summary = json.loads((sagittal_dir / 'summary.json').read_text())
assert sha(sagittal_dir / 'summary.json') == control_config['sagittal_summary_sha256']
assert all(sha(sagittal_dir / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal_dir / 'geometry.jsonl').open()]
assert all(row['split'] == 'train' for row in sagittal_records)
sagittal_images = np.load(sagittal_dir / 'model_input.npy', mmap_mode='r')
sagittal_donors = {}
for index, record in enumerate(sagittal_records):
    sagittal_donors.setdefault(record['donor_id'], []).append(index)
assert len(sagittal_records) == 653 and len(sagittal_donors) == 40
sagittal_donor_ids = sorted(sagittal_donors)
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

torch.manual_seed(control_config['seed'])
torch.cuda.manual_seed_all(control_config['seed'])
rng = np.random.default_rng(control_config['seed'])
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
attach_global_plane_matcher(model, enabled=True)
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
control_zero = torch.load(control_path, map_location='cpu', weights_only=True)
assert parent['step'] == 6000 and parent['config'] == parent_config
assert control_zero['step'] == 0 and control_zero['config'] == control_config
assert parent['calibrated'] is False and control_zero['calibrated'] is False
assert parent['model'].keys() == control_zero['model'].keys()
assert all(torch.equal(parent['model'][key], control_zero['model'][key])
           for key in parent['model'])
model.load_state_dict(parent['model'], strict=True)
del parent
with torch.no_grad():
    model.encoder[0][0].weight[:, 3:5].zero_()
assert torch.count_nonzero(model.encoder[0][0].weight[:, 3:5]) == 0
model.requires_grad_(False)
pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
shared_modules = (model.encoder, model.lateral)
for module in (*pose_modules, *shared_modules):
    module.requires_grad_(True)
groups = [
    {'params': [p for module in pose_modules for p in module.parameters()], 'base_lr': 2e-5},
    {'params': [p for module in shared_modules for p in module.parameters()], 'base_lr': 5e-6},
]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
optimizer.load_state_dict(control_zero['optimizer'])
parameters = [p for group in groups for p in group['params']]
rng.bit_generator.state = control_zero['numpy_rng']
torch.set_rng_state(control_zero['torch_rng'])
torch.cuda.set_rng_state_all(control_zero['cuda_rng'])
del control_zero

source_files = tuple(control_config['source_sha256']) + (
    'structural_input_137.py', 'train_structural_input_137.py')
config = {'experiment': '137 matched fixed structural input treatment',
    'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'control_completion_sha256': sha(control_dir / 'completed.json'),
    'control_config_sha256': sha(control_dir / 'config.json'),
    'control_draws_sha256': sha(control_dir / 'draws.jsonl'),
    'control_step0_sha256': sha(control_path),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'seed': control_config['seed'], 'updates': updates, 'side': side,
    'valid_sites': sites, 'checkpoints': checkpoints,
    'synthetic_per_update': control_config['synthetic_per_update'],
    'real_weak_per_update': control_config['real_weak_per_update'],
    'synthetic_provenance': context['provenance'],
    'coronal_bindings': coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'sagittal_output_sha256': sagittal_summary['output_sha256'],
    'sole_change': 'fixed image-only channels 3/4; first-conv channels 3/4 zero at step 0',
    'optimised': control_config['optimised'], 'frozen': control_config['frozen'],
    'loss': control_config['loss'],
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))
path = run / 'joint_step_00000.pt'
temp = path.with_suffix('.tmp')
torch.save({'step': 0, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
    'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
    'cuda_rng': torch.cuda.get_rng_state_all(), 'config': config,
    'calibrated': False}, temp)
os.replace(temp, path)

flags = torch.arange(2, device='cuda')[None, None]
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
started = time.perf_counter()
draw_count = 0
accepted_ids = set()
with (control_dir / 'draws.jsonl').open() as frozen_draws, \
     (control_dir / 'training.jsonl').open() as frozen_training, \
     (run / 'draws.jsonl').open('w') as draws, \
     (run / 'training.jsonl').open('w') as log:
    for step in range(1, updates + 1):
        samples = []
        for slot, appearance in enumerate(('v4', 'v4', 'v3')):
            sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                       else sample_one_shot_slide_artifacts_v3)
            while True:
                virtual = int(rng.integers(len(context['subjects'])))
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    sample = sampler(context, [virtual], draw_seed, side=side)
                eligible = bool(sample['eligible'][0])
                record = sample['provenance'][0]
                draw_count += 1
                row = {'kind': 'synthetic_train', 'update': step, 'slot': slot,
                    'appearance_version': appearance, 'draw_attempt': draw_count,
                    'used': eligible, **record}
                assert row == json.loads(next(frozen_draws))
                draws.write(json.dumps(row, allow_nan=False) + '\n')
                if eligible:
                    assert record['physical_section_id'] not in accepted_ids
                    accepted_ids.add(record['physical_section_id'])
                    samples.append(sample)
                    break

        donor_index = int(rng.integers(len(coronal['donors'])))
        section_index = int(rng.integers(len(coronal['donors'][donor_index]['identities'])))
        real_coronal = sample_reserved_real_train(coronal, donor_index, [section_index])
        coronal_inputs = F.interpolate(real_coronal['inputs'], (side, side),
            mode='bilinear', align_corners=False)
        donor_id = sagittal_donor_ids[int(rng.integers(len(sagittal_donor_ids)))]
        sag_rows = sagittal_donors[donor_id]
        sag_index = sag_rows[int(rng.integers(len(sag_rows)))]
        sagittal_inputs = torch.zeros(1, 5, side, side, device='cuda')
        sagittal_inputs[:, :1] = torch.from_numpy(np.asarray(
            sagittal_images[sag_index:sag_index + 1]).copy()).to('cuda')
        coronal_identity = real_coronal['identities'][0]
        sagittal_identity = {key: sagittal_records[sag_index][key] for key in
            ('donor_id', 'specimen_id', 'experiment_id', 'section_id', 'image_sha256')}
        for row in ({'kind': 'coronal_weak_train', 'update': step, 'slot': 3,
                     'used': True, **coronal_identity},
                    {'kind': 'sagittal_weak_train', 'update': step, 'slot': 4,
                     'used': True, **sagittal_identity}):
            assert row == json.loads(next(frozen_draws))
            draws.write(json.dumps(row, allow_nan=False) + '\n')

        inputs = fill_structure(torch.cat([sample['inputs'] for sample in samples]
                                          + [coronal_inputs, sagittal_inputs]))
        target_state = torch.cat([sample['state'] for sample in samples])
        target_reflection = torch.cat([sample['reflection'] for sample in samples])
        valid = torch.cat([sample['valid_mask'] for sample in samples])
        real_state = torch.cat((real_coronal['state'],
            sagittal_states[sag_index:sag_index + 1].to('cuda')))
        real_reflection = torch.cat((real_coronal['reflection'],
            sagittal_reflection[sag_index:sag_index + 1].long().to('cuda')))
        prediction = model.predict(inputs)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflections = flags.expand(len(states), model.modes, 2)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)

        pixel = torch.multinomial(valid.flatten(1).float(), sites, replacement=True)
        chart = torch.stack((pixel.remainder(side),
            pixel.div(side, rounding_mode='floor')), -1).float() / side
        reference = rigid_points_090(target_state, target_reflection, chart)
        proposed = rigid_points_090(states[:3], reflections[:3], chart[:, None, None])
        reference_five = rigid_points_090(target_state, target_reflection, corners)
        proposed_five = rigid_points_090(states[:3], reflections[:3], corners)
        tissue_mm = (proposed - reference[:, None, None]).norm(dim=-1).mean(-1)
        five_mm = (proposed_five - reference_five[:, None, None]).norm(dim=-1).mean(-1)
        predicted_normal = full_frame_state_to_components(prediction['state'][:3])[1][..., :, 2]
        true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
        normal_penalty = 4000 * (1 - (predicted_normal * true_normal[:, None]
            ).sum(-1).abs().clamp_max(1))
        physical = (.75 * tissue_mm + .25 * five_mm
            + normal_penalty[..., None]).flatten(1) / 1000
        target_mass = F.softmax(-physical.detach() / .7, -1)
        nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
        near_mode = model.base_modes + nearest.indices
        near_cost = physical.reshape(3, model.modes, 2).min(-1).values.gather(1, near_mode)
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        synthetic_loss = (-1.5 * torch.logsumexp(prior[:3] - physical / 1.5, -1)
            + .5 * F.kl_div(prior[:3], target_mass, reduction='none').sum(-1)
            + .5 * physical[:, :2 * model.base_modes].min(-1).values
            + (neighbourhood * near_cost).sum(-1)).mean()

        real_five = rigid_points_090(real_state, real_reflection, corners)
        real_proposed = rigid_points_090(states[3:], reflections[3:], corners)
        real_cost = (real_proposed - real_five[:, None, None]).norm(dim=-1
            ).mean(-1).flatten(1) / 1000
        real_mass = F.softmax(-real_cost.detach() / .7, -1)
        real_loss = (-1.5 * torch.logsumexp(prior[3:] - real_cost / 1.5, -1)
            + .5 * F.kl_div(prior[3:], real_mass, reduction='none').sum(-1)
            + .5 * real_cost.min(-1).values).mean()
        loss = synthetic_loss + .5 * real_loss
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'synthetic_presentations': 3 * step,
            'v4_presentations': 2 * step, 'v3_presentations': step,
            'synthetic_section_ids': [sample['provenance'][0]['physical_section_id']
                for sample in samples], 'coronal_identity': coronal_identity,
            'sagittal_identity': sagittal_identity,
            'v4_best160_train_pose_cost': float(physical[:2].detach().min(-1).values.mean()),
            'v4_direct_top1_train_pose_cost': float(physical[:2].detach().gather(
                1, prior[:2].detach().argmax(-1)[:, None]).mean()),
            'v3_best160_train_pose_cost': float(physical[2].detach().min()),
            'v3_direct_top1_train_pose_cost': float(physical[2].detach()[prior[2].detach().argmax()]),
            'coronal_weak_top1_mm': float(real_cost[0].detach()[prior[3].detach().argmax()]),
            'sagittal_weak_top1_mm': float(real_cost[1].detach()[prior[4].detach().argmax()]),
            'synthetic_loss': float(synthetic_loss.detach()),
            'real_weak_loss': float(real_loss.detach()), 'loss': float(loss.detach()),
            'gradient_norm': float(gradient),
            'seconds_since_process_start': time.perf_counter() - started}
        control_row = json.loads(next(frozen_training))
        assert all(row[key] == control_row[key] for key in
            ('update', 'synthetic_section_ids', 'coronal_identity', 'sagittal_identity'))
        if step == 1:
            assert all(abs(row[key] - control_row[key]) < 1e-5 for key in
                ('v4_best160_train_pose_cost', 'v4_direct_top1_train_pose_cost',
                 'v3_best160_train_pose_cost', 'v3_direct_top1_train_pose_cost',
                 'coronal_weak_top1_mm', 'sagittal_weak_top1_mm',
                 'synthetic_loss', 'real_weak_loss', 'loss'))
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({'event': 'train_milestone',
                **{key: row[key] for key in ('update', 'v4_direct_top1_train_pose_cost',
                    'v3_direct_top1_train_pose_cost', 'coronal_weak_top1_mm',
                    'sagittal_weak_top1_mm', 'loss')}}), flush=True)
        if step in checkpoints:
            path = run / f'joint_step_{step:05d}.pt'
            temp = path.with_suffix('.tmp')
            torch.save({'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'config': config, 'calibrated': False}, temp)
            os.replace(temp, path)
    assert frozen_draws.readline() == ''
    assert frozen_training.readline() == ''

assert draw_count == control_done['synthetic_draw_attempts']
assert len(accepted_ids) == control_done['unique_synthetic_physical_sections'] == 6000
assert sha(run / 'draws.jsonl') == control_done['draws_sha256']
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'synthetic_presentations': 3 * updates, 'v4_presentations': 2 * updates,
    'v3_presentations': updates, 'synthetic_draw_attempts': draw_count,
    'unique_synthetic_physical_sections': len(accepted_ids),
    'real_weak_coronal_presentations': updates,
    'real_weak_sagittal_presentations': updates,
    'parent_checkpoint_sha256': sha(parent_path),
    'control_completion_sha256': sha(control_dir / 'completed.json'),
    'control_draws_sha256': control_done['draws_sha256'],
    'protocol_sha256': sha(protocol), 'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
        for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
