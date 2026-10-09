"""Continue v3 one-pass full-plane pose with physical sagittal and coronal TRAIN images."""
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
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090

source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/V3_MIXED_REAL_POSE_108_PROTOCOL_20261009.md'
parent_run = root / 'runs/v3_one_pass_pose_capture_pilot_001'
parent = parent_run / 'joint_step_02000.pt'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
run = root / 'runs/v3_mixed_real_pose_continuation_108'
seed, updates, synthetic_count, side, sites = 20261009108, 1959, 2, 256, 128
checkpoints = (0, 653, 1959)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
parent_receipt = json.loads((parent_run / 'completed.json').read_text())
assert sha(parent) == parent_receipt['checkpoint_sha256']['2000']
assert sha(parent_run / 'config.json') == parent_receipt['config_sha256']
assert parent_receipt['updates'] == 8000 and parent_receipt['accepted_synthetic'] == 16000
assert not any(parent_receipt[key] for key in ('calibrated', 'public_benchmark_used',
    'expert_real_truth_used', 'external_pretrained_weights_used'))
sagittal_summary = json.loads((sagittal / 'summary.json').read_text())
assert sha(sagittal / 'model_input.npy') == sagittal_summary['output_sha256']['model_input.npy']
assert sha(sagittal / 'geometry.jsonl') == sagittal_summary['output_sha256']['geometry.jsonl']
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
assert len(sagittal_records) == 653 and len({row['section_id'] for row in sagittal_records}) == 653
assert len({row['donor_id'] for row in sagittal_records}) == 40
assert all(row['split'] == 'train' and row['array_row_index'] == index and
    row['fixed_gain'] == 4.259364821544654 for index, row in enumerate(sagittal_records))
assert not {row['donor_id'] for row in sagittal_records} & {
    10422, 10405, 10355, 10347, 10430, 10248, 10443, 10354,
    10410, 10223, 10302, 10376, 10392, 10305, 10173, 10351}
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')
assert sagittal_images.shape == (653, 1, side, side)
affine = torch.tensor(np.asarray([row['model_pixel_to_ccf_ref9_ap_dv_ml_um']
    for row in sagittal_records]), dtype=torch.float64)
ouv = torch.stack((affine[:, :, 2], side * affine[:, :, 0], side * affine[:, :, 1]), 1)
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
normal = full_frame_state_to_components(sagittal_states)[1][..., :, 2]
sagittal_reflection = normal.gather(1, normal.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sagittal_reflection, 0] += (side - 1) / side * ouv[sagittal_reflection, 1]
ouv[sagittal_reflection, 1] *= -1
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()

context = load_streaming_synthetic_v7_64(device='cuda')
coronal = load_reserved_real_train()
rng = np.random.default_rng(seed)
sagittal_schedule = np.concatenate([rng.permutation(653) for _ in range(3)]).astype(np.int32)
previous_coronal = {tuple(map(int, pair)) for pair in np.load(parent_run / 'real_schedule.npy')}
available = [[index for index in range(len(donor['identities']))
    if (donor_index, index) not in previous_coronal]
    for donor_index, donor in enumerate(coronal['donors'])]
coronal_schedule = []
while len(coronal_schedule) < updates:
    for donor_index in rng.permutation(len(available)):
        donor_index = int(donor_index)
        if available[donor_index]:
            pick = int(rng.integers(len(available[donor_index])))
            coronal_schedule.append((donor_index, available[donor_index].pop(pick)))
        if len(coronal_schedule) == updates:
            break
assert len(set(coronal_schedule)) == updates and not set(coronal_schedule) & previous_coronal
run.mkdir(parents=True, exist_ok=False)
np.save(run / 'sagittal_schedule.npy', sagittal_schedule)
np.save(run / 'coronal_schedule.npy', np.asarray(coronal_schedule, dtype=np.int32))
source_files = ('train_v3_mixed_real_pose_108.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py')
config = {'seed': seed, 'updates': updates, 'synthetic_per_update': synthetic_count,
    'coronal_per_update': 1, 'sagittal_per_update': 1, 'side': side,
    'sampled_valid_pixels': sites, 'checkpoints': checkpoints,
    'parent_checkpoint_sha256': sha(parent),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'coronal_bindings': coronal['bindings'],
    'coronal_real_label_role': coronal['label_role'],
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'sagittal_output_sha256': sagittal_summary['output_sha256'],
    'sagittal_real_label_role': 'weak inherited Allen affine',
    'schedule_sha256': {name: sha(run / name) for name in
        ('sagittal_schedule.npy', 'coronal_schedule.npy')},
    'trainable_heads': 'pose, anchor_pose, anchor_global: updates 1-1959',
    'trainable_shared': 'encoder and lateral: updates 654-1959',
    'frozen': 'atlas matcher, warp, fitted scorer, uncertainty',
    'loss': 'synthetic v3 direct pose + 0.5*(mean coronal/sagittal five-point weak real)',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
(run / 'config.json').write_text(json.dumps(config, indent=2))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 2000 and checkpoint['calibrated'] is False
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
model.requires_grad_(False)
heads = (model.pose, model.anchor_pose, model.anchor_global)
shared = (model.encoder, model.lateral)
for module in heads:
    module.requires_grad_(True)
groups = [{'params': list(module.parameters()), 'lr': 1e-5, 'base_lr': 1e-5}
    for module in heads] + [{'params': list(module.parameters()), 'lr': 0.,
    'base_lr': 2e-6} for module in shared]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
parameters = [parameter for group in groups for parameter in group['params']]
subject_rng = torch.Generator().manual_seed(seed)
draw_seed = seed * 1000000
flags = torch.tensor([0, 1], device='cuda')
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
    [127.5, 127.5]], device='cuda') / side


def save(step):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'subject_rng': subject_rng.get_state(), 'draw_seed': draw_seed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'config': config, 'calibrated': False},
        run / f'joint_step_{step:05d}.pt')


save(0)
attempts = 0
started = time.perf_counter()
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        if step == 654:
            for module in shared:
                module.requires_grad_(True)
        accepted, pending = {}, list(range(synthetic_count))
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),),
                generator=subject_rng).tolist()
            with torch.no_grad():
                sampled = sample_one_shot_slide_artifacts_v3(context, virtual,
                    draw_seed, side=side)
            for row, record in enumerate(sampled['provenance']):
                slot, used = pending[row], bool(sampled['eligible'][row])
                attempts += 1
                draws.write(json.dumps({'kind': 'synthetic_train', **record,
                    'update': step, 'slot': slot, 'draw_attempt': attempts,
                    'used': used}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                        ('inputs', 'state', 'reflection', 'centre', 'valid_mask')}
                    accepted[slot]['record'] = record
            pending = [slot for slot in pending if slot not in accepted]
            draw_seed += 1
        synthetic = {key: torch.cat([accepted[slot][key]
            for slot in range(synthetic_count)]) for key in
            ('inputs', 'state', 'reflection', 'centre', 'valid_mask')}
        donor_index, coronal_index = coronal_schedule[step - 1]
        acquired_coronal = sample_reserved_real_train(coronal, donor_index,
            [coronal_index], device='cuda')
        coronal_image = F.interpolate(acquired_coronal['inputs'], (side, side),
            mode='bilinear', align_corners=False)
        sagittal_index = int(sagittal_schedule[step - 1])
        sagittal_image = torch.zeros(1, 5, side, side, device='cuda')
        sagittal_image[:, :1] = torch.from_numpy(np.asarray(
            sagittal_images[sagittal_index]).copy()).to('cuda')
        images = torch.cat((synthetic['inputs'], coronal_image, sagittal_image))
        real_states = torch.cat((acquired_coronal['state'],
            sagittal_states[sagittal_index:sagittal_index + 1].to('cuda')))
        real_reflection = torch.cat((acquired_coronal['reflection'],
            sagittal_reflection[sagittal_index:sagittal_index + 1].long().to('cuda')))
        draws.write(json.dumps({'kind': 'coronal_weak_train',
            **acquired_coronal['identities'][0], 'update': step,
            'slot': synthetic_count, 'used': True}) + '\n')
        draws.write(json.dumps({'kind': 'sagittal_weak_train',
            **{key: sagittal_records[sagittal_index][key] for key in
                ('donor_id', 'specimen_id', 'experiment_id', 'section_id', 'image_sha256')},
            'update': step, 'slot': synthetic_count + 1, 'used': True}) + '\n')
        prediction = model.predict(images)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflections = flags[None, None].expand(len(states), model.modes, 2)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)

        pixel = torch.multinomial(synthetic['valid_mask'].flatten(1).float(),
            sites, replacement=True)
        target = synthetic['centre'].reshape(synthetic_count, -1, 3).gather(
            1, pixel[..., None].expand(-1, -1, 3))
        chart = torch.stack((pixel.remainder(side),
            pixel.div(side, rounding_mode='floor')), -1).float() / side
        source_five = rigid_points_090(synthetic['state'],
            synthetic['reflection'], corners)
        synthetic_five = (rigid_points_090(states[:synthetic_count],
            reflections[:synthetic_count], corners) - source_five[:, None, None]
            ).norm(dim=-1).mean(-1)
        synthetic_dense = (rigid_points_090(states[:synthetic_count],
            reflections[:synthetic_count], chart[:, None, None]) - target[:, None, None]
            ).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(synthetic['state'])[1][..., :, 2]
        normal_penalty = 4000 * (1 - (normal[:synthetic_count] * true_normal[:, None])
            .sum(-1).abs().clamp_max(1))
        synthetic_cost = (.75 * synthetic_dense + .25 * synthetic_five
            + normal_penalty[..., None]).flatten(1) / 1000
        synthetic_target = F.softmax(-synthetic_cost.detach() / .7, -1)
        synthetic_kl = F.kl_div(prior[:synthetic_count], synthetic_target,
            reduction='none').sum(-1)
        nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
        near_mode = model.base_modes + nearest.indices
        near_distance = synthetic_cost.reshape(synthetic_count, model.modes, 2).min(
            -1).values.gather(1, near_mode)
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        synthetic_loss = (-1.5 * torch.logsumexp(prior[:synthetic_count]
            - synthetic_cost / 1.5, -1) + .5 * synthetic_kl
            + .5 * synthetic_cost[:, :2 * model.base_modes].min(-1).values
            + (neighbourhood * near_distance).sum(-1)).mean()

        real_five = rigid_points_090(real_states, real_reflection, corners)
        real_cost = (rigid_points_090(states[synthetic_count:],
            reflections[synthetic_count:], corners) - real_five[:, None, None]
            ).norm(dim=-1).mean(-1).flatten(1) / 1000
        real_target = F.softmax(-real_cost.detach() / .7, -1)
        real_kl = F.kl_div(prior[synthetic_count:], real_target,
            reduction='none').sum(-1)
        real_loss = (-1.5 * torch.logsumexp(prior[synthetic_count:]
            - real_cost / 1.5, -1) + .5 * real_kl
            + .5 * real_cost.min(-1).values).mean()
        loss = synthetic_loss + .5 * real_loss
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        for index, group in enumerate(optimizer.param_groups):
            group['lr'] = group['base_lr'] * decay if index < len(heads) or step > 653 else 0.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'update': step, 'stage': 'heads' if step <= 653 else 'shared',
            'synthetic_section_ids': [accepted[slot]['record']['physical_section_id']
                for slot in range(synthetic_count)],
            'coronal_identity': acquired_coronal['identities'][0],
            'sagittal_identity': {key: sagittal_records[sagittal_index][key]
                for key in ('donor_id', 'specimen_id', 'experiment_id', 'section_id')},
            'synthetic_best160_cost_mm': float(synthetic_cost.detach().min(-1).values.mean()),
            'synthetic_prior_cost_mm': float(synthetic_cost.detach().gather(
                1, prior[:synthetic_count].detach().argmax(-1)[:, None]).mean()),
            'coronal_best160_weak_mm': float(real_cost[0].detach().min()),
            'sagittal_best160_weak_mm': float(real_cost[1].detach().min()),
            'coronal_prior_weak_mm': float(real_cost[0].detach()[prior[2].detach().argmax()]),
            'sagittal_prior_weak_mm': float(real_cost[1].detach()[prior[3].detach().argmax()]),
            'synthetic_loss': float(synthetic_loss.detach()),
            'weak_real_loss': float(real_loss.detach()),
            'total_loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 326 == 0 or step in checkpoints:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in ('update', 'stage',
                'synthetic_best160_cost_mm', 'synthetic_prior_cost_mm',
                'coronal_prior_weak_mm', 'sagittal_prior_weak_mm',
                'total_loss', 'seconds')}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

(run / 'completed.json').write_text(json.dumps({
    'updates': updates, 'accepted_synthetic': updates * synthetic_count,
    'synthetic_draw_attempts': attempts,
    'real_sagittal_presentations': updates,
    'unique_sagittal_train_sections': 653,
    'real_coronal_presentations': updates,
    'unique_coronal_train_sections': len(set(coronal_schedule)),
    'parent_checkpoint_sha256': sha(parent), 'protocol_sha256': sha(protocol),
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'schedule_sha256': config['schedule_sha256'],
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
        for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'updates': updates,
    'seconds': time.perf_counter() - started}), flush=True)
