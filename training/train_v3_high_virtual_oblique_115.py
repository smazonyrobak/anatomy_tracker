"""Matched 113-prefix continuation with higher virtual-oblique pose loss."""
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
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_reserved_real_oblique_v3 import sample_reserved_real_oblique_v3
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train, sample_reserved_real_train
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


source = Path(__file__).resolve().parent
protocol = source.parent / 'docs/publication/V3_HIGH_VIRTUAL_OBLIQUE_115_PROTOCOL_20261009.md'
parent_run = root / 'runs/v3_mixed_real_pose_coronal_risk_111'
parent = parent_run / 'joint_step_01959.pt'
reference_run = root / 'runs/v3_mixed_virtual_oblique_113'
teacher_path = root / 'runs/v3_mixed_real_pose_continuation_108/joint_step_00653.pt'
geometry_audit = root / 'runs/reserved_train_volume_geometry_105_audit/rows.jsonl'
sagittal = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
run = root / 'runs/v3_high_virtual_oblique_115'
seed, updates, schedule_updates, side, sites = 20261009113, 1306, 4000, 256, 128
checkpoints = (0, 500, 1000, 1306)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert root.drive.upper() == source.drive.upper() == 'I:' and not run.exists()
reference_done = json.loads((reference_run / 'completed.json').read_text())
reference_config = json.loads((reference_run / 'config.json').read_text())
assert reference_done['batches'] == schedule_updates and reference_config['seed'] == seed
assert sha(reference_run / 'config.json') == reference_done['config_sha256']
assert sha(reference_run / 'draws.jsonl') == reference_done['draws_sha256']
reference_prefix = hashlib.sha256()
reference_synthetic, reference_sagittal, reference_virtual = set(), set(), set()
with (reference_run / 'draws.jsonl').open('rb') as stream:
    for line in stream:
        row = json.loads(line)
        if row['batch'] > updates:
            break
        reference_prefix.update(line)
        if row['kind'] == 'synthetic_train' and row['used']:
            key, seen = row['physical_section_id'], reference_synthetic
        elif row['kind'] == 'sagittal_weak_train':
            key, seen = row['section_id'], reference_sagittal
        elif row['kind'] == 'virtual_oblique_weak_train' and row['used']:
            key, seen = row['section_id'], reference_virtual
        else:
            continue
        assert key not in seen
        seen.add(key)
assert (len(reference_synthetic), len(reference_sagittal), len(reference_virtual)) == (
    2 * updates, updates // 2, updates // 2)
expected_draws_sha256 = reference_prefix.hexdigest()
parent_done = json.loads((parent_run / 'completed.json').read_text())
assert sha(parent) == parent_done['checkpoint_sha256']['1959']
assert parent_done['calibrated'] is False and parent_done['public_benchmark_used'] is False
teacher_done = json.loads((teacher_path.parent / 'completed.json').read_text())
assert sha(teacher_path) == teacher_done['checkpoint_sha256']['653']

context = load_streaming_synthetic_v7_64(device='cuda')
coronal = load_reserved_real_train()
by_id = {str(donor['animal_id']): i for i, donor in enumerate(coronal['donors'])}
audited = [json.loads(line) for line in geometry_audit.open()]
regular = [by_id[row['donor'].removeprefix('donor_')] for row in audited
    if row['sections'] == 140 and row['section_number_gaps'] == 0
    and row['inplane_basis_max_variation_um'] < 1e-5
    and row['section_step_max_residual_um'] < 5]
assert len(audited) == 64 and len(regular) == 63 and len(set(regular)) == 63
rng = np.random.default_rng(seed)
permuted = rng.permutation(regular).tolist()
virtual_holdout, virtual_train = permuted[:8], permuted[8:]
coronal_train = [i for i in range(len(coronal['donors'])) if i not in virtual_holdout]

sagittal_done = json.loads((sagittal / 'summary.json').read_text())
assert sha(sagittal / 'model_input.npy') == sagittal_done['output_sha256']['model_input.npy']
assert sha(sagittal / 'geometry.jsonl') == sagittal_done['output_sha256']['geometry.jsonl']
sagittal_records = [json.loads(line) for line in (sagittal / 'geometry.jsonl').open()]
assert len(sagittal_records) == 653 and all(row['split'] == 'train' for row in sagittal_records)
sagittal_images = np.load(sagittal / 'model_input.npy', mmap_mode='r')
affine = torch.tensor(np.asarray([row['model_pixel_to_ccf_ref9_ap_dv_ml_um']
    for row in sagittal_records]), dtype=torch.float64)
ouv = torch.stack((affine[:, :, 2], side * affine[:, :, 0], side * affine[:, :, 1]), 1)
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
normal = full_frame_state_to_components(sagittal_states)[1][..., :, 2]
sagittal_reflection = normal.gather(1, normal.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sagittal_reflection, 0] += (side - 1) / side * ouv[sagittal_reflection, 1]
ouv[sagittal_reflection, 1] *= -1
sagittal_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()
sagittal_schedule = rng.permutation(653).astype(np.int32)
for _ in range(3):
    rng.permutation(653)  # Match 113's downstream RNG state without replaying sections.

source_files = ('train_v3_high_virtual_oblique_115.py',
    'arbitrary_plane_reserved_real_oblique_v3.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'global_atlas_contrast_090.py')
config = {'seed': seed, 'batches': updates, 'side': side,
    'learning_rate_schedule_batches': schedule_updates,
    'synthetic_per_batch': 2, 'coronal_per_batch': 1,
    'sagittal_per_odd_batch': 1, 'virtual_oblique_per_even_batch': 1,
    'virtual_source_role': 'weak Allen affine TRAIN serial-stack reslice; not acquired oblique',
    'virtual_source_audit_sha256': sha(geometry_audit),
    'virtual_train_animal_ids': [coronal['donors'][i]['animal_id'] for i in virtual_train],
    'virtual_holdout_animal_ids': [coronal['donors'][i]['animal_id'] for i in virtual_holdout],
    'virtual_holdout_seeds': [seed * 100000 + i for i in range(4)],
    'virtual_holdout_role': 'TRAIN-subset novel planes; parent model previously saw some cardinal donor images, not animal-independent',
    'checkpoints': checkpoints, 'sampled_valid_pixels': sites,
    'comparison_113_completed_sha256': sha(reference_run / 'completed.json'),
    'expected_113_draw_prefix_sha256': expected_draws_sha256,
    'parent_checkpoint_sha256': sha(parent), 'teacher_checkpoint_sha256': sha(teacher_path),
    'parent_completion_sha256': sha(parent_run / 'completed.json'),
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'coronal_bindings': coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal / 'summary.json'),
    'sagittal_output_sha256': sagittal_done['output_sha256'],
    'trainable': 'shared encoder/lateral and direct pose heads',
    'frozen': 'atlas fitter, deformation, scorer, uncertainty',
    'loss': '111 synthetic pose + 0.25 coronal weak pose + 0.25 sagittal weak pose on odd batches or 2.0 virtual weak pose on even batches + 2 coronal teacher retention + 2 coronal posterior risk',
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
assert config['virtual_train_animal_ids'] == reference_config['virtual_train_animal_ids']
assert config['virtual_holdout_animal_ids'] == reference_config['virtual_holdout_animal_ids']
assert all(config['source_sha256'][name] == digest for name, digest in
    reference_config['source_sha256'].items() if name != 'train_v3_mixed_virtual_oblique_113.py')
run.mkdir(parents=True, exist_ok=False)
(run / 'config.json').write_text(json.dumps(config, indent=2))

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
checkpoint = torch.load(parent, map_location='cpu', weights_only=True)
teacher_checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
teacher = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(checkpoint['model'], strict=True)
teacher.load_state_dict(teacher_checkpoint['model'], strict=True)
del teacher_checkpoint
model.requires_grad_(False)
heads = (model.pose, model.anchor_pose, model.anchor_global)
shared = (model.encoder, model.lateral)
for module in (*heads, *shared):
    module.requires_grad_(True)
groups = [{'params': list(module.parameters()), 'lr': 1e-5,
    'base_lr': 1e-5} for module in heads] + [
    {'params': list(module.parameters()), 'lr': 2e-6, 'base_lr': 2e-6}
    for module in shared]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
optimizer.load_state_dict(checkpoint['optimizer'])
del checkpoint
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
started = time.perf_counter()
attempts = 0
with (run / 'training.jsonl').open('w') as log, (run / 'draws.jsonl').open('w') as draws:
    for step in range(1, updates + 1):
        accepted, pending = {}, [0, 1]
        while pending:
            virtual = torch.randint(len(context['subjects']), (len(pending),),
                generator=subject_rng).tolist()
            with torch.no_grad():
                sampled = sample_one_shot_slide_artifacts_v3(context, virtual,
                    draw_seed, side=side)
            for row, record in enumerate(sampled['provenance']):
                slot, used = pending[row], bool(sampled['eligible'][row])
                attempts += 1
                draws.write(json.dumps({'kind': 'synthetic_train', 'batch': step,
                    'slot': slot, 'used': used, **record}) + '\n')
                if used:
                    accepted[slot] = {key: sampled[key][row:row + 1] for key in
                        ('inputs', 'state', 'reflection', 'centre', 'valid_mask')}
            pending = [slot for slot in pending if slot not in accepted]
            draw_seed += 1
        synthetic = {key: torch.cat([accepted[slot][key] for slot in (0, 1)])
            for key in ('inputs', 'state', 'reflection', 'centre', 'valid_mask')}
        donor_index = int(rng.choice(coronal_train))
        section_index = int(rng.integers(len(coronal['donors'][donor_index]['identities'])))
        acquired = sample_reserved_real_train(coronal, donor_index,
            [section_index], device='cuda')
        coronal_image = F.interpolate(acquired['inputs'], (side, side),
            mode='bilinear', align_corners=False)
        draws.write(json.dumps({'kind': 'coronal_weak_train', 'batch': step,
            **acquired['identities'][0]}) + '\n')
        if step % 2:
            sagittal_index = int(sagittal_schedule[(step - 1) // 2])
            other_image = torch.zeros(1, 5, side, side, device='cuda')
            other_image[:, :1] = torch.from_numpy(np.asarray(
                sagittal_images[sagittal_index]).copy()).to('cuda')
            other_state = sagittal_states[sagittal_index:sagittal_index + 1].cuda()
            other_reflection = sagittal_reflection[sagittal_index:sagittal_index + 1].long().cuda()
            other_record = {'kind': 'sagittal_weak_train', 'batch': step,
                **{key: sagittal_records[sagittal_index][key] for key in
                ('donor_id', 'specimen_id', 'experiment_id', 'section_id', 'image_sha256')}}
        else:
            donor_index = int(rng.choice(virtual_train))
            while True:
                with torch.no_grad():
                    other = sample_reserved_real_oblique_v3(coronal,
                        donor_index, draw_seed, side=side, device='cuda')
                used = bool(other['eligible'][0])
                draws.write(json.dumps({'kind': 'virtual_oblique_weak_train',
                    'batch': step, 'used': used, **other['provenance'][0]}) + '\n')
                draw_seed += 1
                if used:
                    break
                donor_index = int(rng.choice(virtual_train))
            other_image = other['inputs']
            other_state = other['state']
            other_reflection = other['reflection']
            other_record = {'kind': 'virtual_oblique_weak_train',
                'batch': step, 'animal_id': other['provenance'][0]['animal_id'],
                'section_id': other['provenance'][0]['section_id']}
        if step % 2:
            draws.write(json.dumps(other_record) + '\n')
        images = torch.cat((synthetic['inputs'], coronal_image, other_image))
        real_states = torch.cat((acquired['state'], other_state))
        real_reflection = torch.cat((acquired['reflection'], other_reflection))
        prediction = model.predict(images)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflections = flags[None, None].expand(4, model.modes, 2)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)

        pixel = torch.multinomial(synthetic['valid_mask'].flatten(1).float(),
            sites, replacement=True)
        target = synthetic['centre'].reshape(2, -1, 3).gather(
            1, pixel[..., None].expand(-1, -1, 3))
        chart = torch.stack((pixel.remainder(side),
            pixel.div(side, rounding_mode='floor')), -1).float() / side
        source_five = rigid_points_090(synthetic['state'],
            synthetic['reflection'], corners)
        synthetic_five = (rigid_points_090(states[:2], reflections[:2], corners)
            - source_five[:, None, None]).norm(dim=-1).mean(-1)
        synthetic_dense = (rigid_points_090(states[:2],
            reflections[:2], chart[:, None, None]) - target[:, None, None]
            ).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(synthetic['state'])[1][..., :, 2]
        normal_penalty = 4000 * (1 - (normal[:2] * true_normal[:, None])
            .sum(-1).abs().clamp_max(1))
        synthetic_cost = (.75 * synthetic_dense + .25 * synthetic_five
            + normal_penalty[..., None]).flatten(1) / 1000
        synthetic_target = F.softmax(-synthetic_cost.detach() / .7, -1)
        synthetic_kl = F.kl_div(prior[:2], synthetic_target,
            reduction='none').sum(-1)
        nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
        near_mode = model.base_modes + nearest.indices
        near_distance = synthetic_cost.reshape(2, model.modes, 2).min(
            -1).values.gather(1, near_mode)
        neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
        synthetic_loss = (-1.5 * torch.logsumexp(prior[:2]
            - synthetic_cost / 1.5, -1) + .5 * synthetic_kl
            + .5 * synthetic_cost[:, :2 * model.base_modes].min(-1).values
            + (neighbourhood * near_distance).sum(-1)).mean()

        real_five = rigid_points_090(real_states, real_reflection, corners)
        real_cost = (rigid_points_090(states[2:], reflections[2:], corners)
            - real_five[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
        real_target = F.softmax(-real_cost.detach() / .7, -1)
        real_kl = F.kl_div(prior[2:], real_target,
            reduction='none').sum(-1)
        real_pose_terms = (-1.5 * torch.logsumexp(prior[2:]
            - real_cost / 1.5, -1) + .5 * real_kl
            + .5 * real_cost.min(-1).values)
        real_loss = real_pose_terms.mean()
        virtual_extra = 1.75 * real_pose_terms[1] if step % 2 == 0 else 0.
        with torch.no_grad():
            teacher_prediction = teacher.predict(coronal_image)
            teacher_prior = (teacher_prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-teacher_prediction['reflection_logit']),
                F.logsigmoid(teacher_prediction['reflection_logit'])), -1)).flatten(1)
            teacher_mass = teacher_prior.exp()
            teacher_five = rigid_points_090(
                teacher_prediction['state'][:, :, None].expand(-1, -1, 2, -1),
                reflections[2:3], corners)
        student_five = rigid_points_090(states[2:3], reflections[2:3], corners)
        teacher_kl = F.kl_div(prior[2:3], teacher_mass, reduction='batchmean')
        teacher_geometry_mm = (teacher_mass * (student_five - teacher_five).norm(dim=-1)
            .mean(-1).flatten(1) / 1000).sum(-1).mean()
        coronal_risk_mm = (prior[2].exp() * F.relu(real_cost[0] - 1.0)).sum()
        loss = (synthetic_loss + .5 * real_loss + virtual_extra
            + 2 * (teacher_kl + teacher_geometry_mm) + 2 * coronal_risk_mm)
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / schedule_updates))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        row = {'batch': step, 'other_kind': other_record['kind'],
            'synthetic_best160_mm': float(synthetic_cost.detach().min(-1).values.mean()),
            'synthetic_selected_mm': float(synthetic_cost.detach().gather(
                1, prior[:2].detach().argmax(-1)[:, None]).mean()),
            'coronal_best160_weak_mm': float(real_cost[0].detach().min()),
            'coronal_selected_weak_mm': float(real_cost[0].detach()[prior[2].detach().argmax()]),
            'other_best160_weak_mm': float(real_cost[1].detach().min()),
            'other_selected_weak_mm': float(real_cost[1].detach()[prior[3].detach().argmax()]),
            'synthetic_loss': float(synthetic_loss.detach()),
            'real_loss': float(real_loss.detach()),
            'coronal_pose_loss': float(real_pose_terms[0].detach()),
            'other_pose_loss': float(real_pose_terms[1].detach()),
            'coronal_pose_weighted': float((.25 * real_pose_terms[0]).detach()),
            'other_pose_weighted': float((
                (2. if step % 2 == 0 else .25) * real_pose_terms[1]).detach()),
            'teacher_kl': float(teacher_kl.detach()),
            'teacher_geometry_mm': float(teacher_geometry_mm.detach()),
            'coronal_risk_mm': float(coronal_risk_mm.detach()),
            'loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'seconds': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 500 == 0:
            log.flush()
            draws.flush()
            print(json.dumps({key: row[key] for key in ('batch', 'other_kind',
                'synthetic_selected_mm', 'coronal_selected_weak_mm',
                'other_selected_weak_mm', 'loss', 'seconds')}), flush=True)
        if step in checkpoints:
            save(step)
    log.flush()
    draws.flush()

draws_sha256 = sha(run / 'draws.jsonl')
assert draws_sha256 == expected_draws_sha256
(run / 'completed.json').write_text(json.dumps({
    'batches': updates, 'synthetic_accepted': 2 * updates,
    'synthetic_attempts': attempts,
    'virtual_oblique_presentations': updates // 2,
    'virtual_train_donors': len(virtual_train),
    'virtual_holdout_donors': len(virtual_holdout),
    'parent_checkpoint_sha256': sha(parent),
    'teacher_checkpoint_sha256': sha(teacher_path),
    'comparison_113_completed_sha256': sha(reference_run / 'completed.json'),
    'matched_113_draw_prefix_sha256': expected_draws_sha256,
    'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': draws_sha256,
    'training_sha256': sha(run / 'training.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
        for step in checkpoints},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False,
    'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'batches': updates,
    'seconds': time.perf_counter() - started}), flush=True)
