"""Bounded mixed-real pose continuation of the scratch-trained 132 joint model."""

import hashlib
import json
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
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_reserved_real_stream_v8 import (
    load_reserved_real_train, sample_reserved_real_train)
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/MIXED_POSE_RECOVERY_160_PROTOCOL_20261011.md'
parent_dir = root / 'runs/v4_pose_adaptation_132'
parent_path = parent_dir / 'joint_step_02000.pt'
run = root / 'runs/mixed_pose_recovery_160_train_001'
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
seed, side, updates, sites = 20261011160, 256, 653, 128
checkpoints = (0, 326, 653)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
parent_done = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
assert sha(parent_path) == parent_done['checkpoint_sha256']['2000']
assert sha(parent_dir / 'config.json') == parent_done['config_sha256']
assert parent_done['updates'] == 2000 and not any(parent_done[key] for key in (
    'calibrated', 'public_benchmark_used', 'expert_real_truth_used',
    'external_pretrained_weights_used'))
assert sha(source / 'arbitrary_plane_one_shot_model.py') == parent_config[
    'source_sha256']['arbitrary_plane_one_shot_model.py']
assert sha(source / 'global_plane_matcher_120.py') == parent_config[
    'source_sha256']['global_plane_matcher_120.py']

context = load_streaming_synthetic_v7_64(device='cuda')
coronal = load_reserved_real_train()
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
assert coronal['bindings'] == parent_config['coronal_bindings']
assert len(context['bases']) == 64 and len(context['subjects']) == 4096
assert len(coronal['donors']) >= updates

sagittal_summary = json.loads((sagittal_dir / 'summary.json').read_text())
assert sha(sagittal_dir / 'summary.json') == parent_config['sagittal_summary_sha256']
assert all(sha(sagittal_dir / name) == digest
           for name, digest in sagittal_summary['output_sha256'].items())
sagittal_records = [json.loads(line) for line in (sagittal_dir / 'geometry.jsonl').open()]
assert len(sagittal_records) == updates and all(row['split'] == 'train'
                                                 for row in sagittal_records)
sagittal_images = np.load(sagittal_dir / 'model_input.npy', mmap_mode='r')
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

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
coronal_order = rng.permutation(len(coronal['donors']))[:updates]
sagittal_order = rng.permutation(updates)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(parent_path, map_location='cpu', weights_only=True)
assert saved['step'] == 2000 and saved['config'] == parent_config
assert saved['calibrated'] is False
model.load_state_dict(saved['model'], strict=True)
model.requires_grad_(False)
pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
shared_modules = (model.encoder, model.lateral)
for module in (*pose_modules, *shared_modules):
    module.requires_grad_(True)
groups = [
    {'params': [p for module in pose_modules for p in module.parameters()], 'lr': 4e-6},
    {'params': [p for module in shared_modules for p in module.parameters()], 'lr': 1e-6},
]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
optimizer.load_state_dict(saved['optimizer'])
assert len(optimizer.param_groups) == 2
assert [len(group['params']) for group in optimizer.param_groups] == [
    len(group['params']) for group in groups]
assert all(float(group['lr']) <= limit * (1 + 1e-12) for group, limit in zip(
    optimizer.param_groups, (4e-6, 1e-6)))
for group, initial_lr in zip(optimizer.param_groups, (4e-6, 1e-6)):
    group['lr'] = initial_lr
parameters = [p for group in optimizer.param_groups for p in group['params']]
del saved

source_files = ('train_mixed_pose_recovery_160.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py', 'global_plane_matcher_120.py')
config = {'seed': seed, 'updates': updates, 'side': side,
    'synthetic_per_update': {'v7_v3': 1, 'v7_v4': 1, 'full_box_v4': 1},
    'real_weak_per_update': {'distinct_coronal_donors': 1, 'distinct_sagittal_sections': 1},
    'checkpoints': checkpoints, 'parent_checkpoint_sha256': sha(parent_path),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'parent_config_sha256': sha(parent_dir / 'config.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'coronal_bindings': coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'sagittal_output_sha256': sagittal_summary['output_sha256'],
    'optimized': '132 direct pose heads and shared encoder/lateral; retained AdamW moments',
    'frozen': 'atlas matcher, local mapper, fitted scores and all other weights',
    'initial_lrs': [4e-6, 1e-6], 'final_lr_fraction': .5,
    'loss': '132 synthetic physical-pose mixture plus 0.5 weak-real five-point mixture',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))


def save(step):
    path = run / f'joint_step_{step:05d}.pt'
    temp = path.with_suffix('.tmp')
    torch.save({'step': step, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'config': config, 'calibrated': False}, temp)
    os.replace(temp, path)


save(0)
flags = torch.arange(2, device='cuda')[None, None]
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
started = time.perf_counter()
draw_count = 0
accepted_ids = set()
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as log:
    for step in range(1, updates + 1):
        samples = []
        for slot, (geometry, appearance) in enumerate((
                ('v7', 'v3'), ('v7', 'v4'), ('full_box', 'v4'))):
            sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
                       else sample_one_shot_slide_artifacts_v4)
            while True:
                index = int(rng.integers(4096 if geometry == 'v7' else 64))
                draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    source_sample = (None if geometry == 'v7' else
                        sample_panel_geometry_train_151(context, [index], draw_seed, side))
                    sample = sampler(context, [index], draw_seed, side,
                                     source_sample=source_sample)
                eligible = bool(sample['eligible'][0])
                draw_count += 1
                record = sample['provenance'][0]
                draws.write(json.dumps({'kind': 'synthetic_train', 'update': step,
                    'slot': slot, 'geometry': geometry, 'appearance_version': appearance,
                    'draw_attempt': draw_count, 'used': eligible, **record},
                    allow_nan=False) + '\n')
                if eligible:
                    accepted_ids.add(record['physical_section_id'])
                    samples.append(sample)
                    break

        donor_index = int(coronal_order[step - 1])
        section_index = int(rng.integers(len(coronal['donors'][donor_index]['identities'])))
        real_coronal = sample_reserved_real_train(coronal, donor_index, [section_index])
        coronal_inputs = F.interpolate(real_coronal['inputs'], (side, side),
            mode='bilinear', align_corners=False)
        sag_index = int(sagittal_order[step - 1])
        sagittal_inputs = torch.zeros(1, 5, side, side, device='cuda')
        sagittal_inputs[:, :1] = torch.from_numpy(np.asarray(
            sagittal_images[sag_index:sag_index + 1]).copy()).to('cuda')
        coronal_identity = real_coronal['identities'][0]
        sagittal_identity = {key: sagittal_records[sag_index][key] for key in
            ('donor_id', 'specimen_id', 'experiment_id', 'section_id', 'image_sha256')}
        draws.write(json.dumps({'kind': 'coronal_weak_train', 'update': step,
            'slot': 3, 'used': True, **coronal_identity}, allow_nan=False) + '\n')
        draws.write(json.dumps({'kind': 'sagittal_weak_train', 'update': step,
            'slot': 4, 'used': True, **sagittal_identity}, allow_nan=False) + '\n')

        inputs = torch.cat([sample['inputs'] for sample in samples]
                           + [coronal_inputs, sagittal_inputs])
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
        for group, start_lr in zip(optimizer.param_groups, (4e-6, 1e-6)):
            group['lr'] = start_lr * (1 - .5 * step / updates)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'synthetic_presentations': 3 * step,
            'synthetic_section_ids': [sample['provenance'][0]['physical_section_id']
                for sample in samples], 'coronal_identity': coronal_identity,
            'sagittal_identity': sagittal_identity,
            'v4_direct_top1_train_pose_cost': float(physical[1:].detach().gather(
                1, prior[1:3].detach().argmax(-1)[:, None]).mean()),
            'v3_direct_top1_train_pose_cost': float(physical[0].detach()[
                prior[0].detach().argmax()]),
            'coronal_weak_top1_mm': float(real_cost[0].detach()[prior[3].detach().argmax()]),
            'sagittal_weak_top1_mm': float(real_cost[1].detach()[prior[4].detach().argmax()]),
            'synthetic_loss': float(synthetic_loss.detach()),
            'real_weak_loss': float(real_loss.detach()), 'loss': float(loss.detach()),
            'gradient_norm': float(gradient),
            'seconds_since_process_start': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step in checkpoints:
            log.flush()
            draws.flush()
            print(json.dumps({'event': 'train_milestone', **{key: row[key] for key in
                ('update', 'v4_direct_top1_train_pose_cost',
                 'v3_direct_top1_train_pose_cost', 'coronal_weak_top1_mm',
                 'sagittal_weak_top1_mm', 'loss')}}), flush=True)
        if step in checkpoints:
            save(step)

(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'synthetic_presentations': 3 * updates,
    'synthetic_draw_attempts': draw_count,
    'unique_synthetic_physical_sections': len(accepted_ids),
    'distinct_coronal_train_donors': updates,
    'distinct_sagittal_train_sections': updates,
    'parent_checkpoint_sha256': sha(parent_path), 'protocol_sha256': sha(protocol),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
        for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'completed', 'updates': updates,
                  'seconds': time.perf_counter() - started}), flush=True)
