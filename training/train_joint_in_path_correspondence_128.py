"""Same-draw frozen-122 continuation; JOINT128_ARM=control or treatment."""

import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
import shutil
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
    compose_full_frame_state, full_frame_state_from_components,
    full_frame_state_to_components)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_reserved_real_stream_v8 import (
    load_reserved_real_train, sample_reserved_real_train)
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.in_path_global_matcher_128 import (
    global_plane_match_128, in_path_correspondence_ce_128)
from training.joint_pose_map_121 import (
    joint_forward_121, joint_spatial_ce_121, joint_target_loss_121)
from training.whole_slice_atlas_feedback_083 import WholeSliceAtlasFeedback083

source = Path(__file__).resolve().parent
arm = os.environ['JOINT128_ARM']
assert arm in ('control', 'treatment')
preflight = os.environ.get('JOINT128_PREFLIGHT') == '1'
run = root / f'runs/joint_in_path_correspondence_128_{arm}'
manifest_dir = root / 'runs/joint_in_path_correspondence_128_manifest'
parent_dir = root / 'runs/joint_pose_map_122'
parent = parent_dir / 'joint_step_02000.pt'
teacher_dir = root / 'runs/v3_mixed_real_pose_coronal_risk_111'
teacher_checkpoint = teacher_dir / 'joint_step_01959.pt'
sagittal_dir = root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008'
protocol = source.parent / 'docs/publication/JOINT_IN_PATH_CORRESPONDENCE_128_PROTOCOL_20261010.md'
seed, updates, side, sites, blind_count = 20261010128, 6000, 256, 128, 16
checkpoints = (0, 1000, 3000, 6000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side


def rigid_cost(states, reflection, target_state, target_reflection, chart):
    reference = rigid_points_090(target_state, target_reflection, chart)
    predicted = rigid_points_090(states, reflection, chart[:, None])
    fixed = rigid_points_090(target_state, target_reflection,
        corners[None].expand(len(states), -1, -1))
    corners_pred = rigid_points_090(states, reflection,
        corners[None, None].expand(len(states), 1, -1, -1))
    normal = full_frame_state_to_components(states)[1][..., :, 2]
    true_normal = full_frame_state_to_components(target_state)[1][..., :, 2]
    penalty = 4 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
    return (.75 * (predicted - reference[:, None]).norm(dim=-1).mean(-1)
            + .25 * (corners_pred - fixed[:, None]).norm(dim=-1).mean(-1)) / 1000 + penalty


assert root.drive.upper() == source.drive.upper() == 'I:'
if preflight:
    manifest_receipt = {'updates_sha256': 'TRAIN_ONLY_PREFLIGHT',
                        'draws_sha256': 'TRAIN_ONLY_PREFLIGHT'}
else:
    manifest_receipt = json.loads((manifest_dir / 'completed.json').read_text())
    assert manifest_receipt['updates'] == updates and manifest_receipt['accepted_synthetic'] == 12000
    assert sha(manifest_dir / 'config.json') == manifest_receipt['config_sha256']
    assert sha(manifest_dir / 'draws.jsonl') == manifest_receipt['draws_sha256']
    assert sha(manifest_dir / 'updates.jsonl') == manifest_receipt['updates_sha256']
    manifest_config = json.loads((manifest_dir / 'config.json').read_text())
    assert manifest_config['protocol_sha256'] == sha(protocol)
    assert manifest_config['source_sha256'] == manifest_receipt['source_sha256']
    assert all(sha(source / name) == digest for name, digest in
        manifest_config['source_sha256'].items())
    schedule = [json.loads(line) for line in (manifest_dir / 'updates.jsonl').open()]
    assert len(schedule) == updates and all(row['update'] == i + 1 for i, row in enumerate(schedule))
parent_receipt = json.loads((parent_dir / 'completed.json').read_text())
parent_config = json.loads((parent_dir / 'config.json').read_text())
assert sha(parent) == parent_receipt['checkpoint_sha256']['2000']
assert sha(teacher_checkpoint) == parent_config['teacher_checkpoint_sha256']
assert not any(parent_receipt[key] for key in (
    'calibrated', 'public_benchmark_used', 'expert_real_truth_used',
    'external_pretrained_weights_used'))
context = load_streaming_synthetic_v7_64(device='cuda')
if preflight:
    rng = torch.Generator().manual_seed(seed + 1)
    draw_seed = seed * 1000000 + 997
    while True:
        virtual = int(torch.randint(len(context['subjects']), (1,), generator=rng))
        with torch.no_grad():
            sampled = sample_one_shot_slide_artifacts_v3(context, [virtual],
                draw_seed, side=side)
        if bool(sampled['eligible'][0]):
            break
        draw_seed += 1
    item = {'subject_index': virtual, 'draw_seed': draw_seed,
        'physical_section_id': sampled['provenance'][0]['physical_section_id'],
        'site_seed': seed + 2, 'jitter_seed': seed + 3,
        'ce_jitter_seed': seed + 4}
    schedule = [{'update': 1, 'synthetic': [item]}]
    print(json.dumps({'event': 'train_only_preflight_draw',
        'physical_section_id': item['physical_section_id'],
        'split': sampled['provenance'][0]['split']}), flush=True)
else:
    assert context['provenance'] == manifest_config['synthetic_provenance']
coronal = load_reserved_real_train()
if not preflight:
    assert coronal['bindings'] == manifest_config['coronal_bindings']
sag_summary = json.loads((sagittal_dir / 'summary.json').read_text())
assert sha(sagittal_dir / 'model_input.npy') == sag_summary['output_sha256']['model_input.npy']
assert sha(sagittal_dir / 'geometry.jsonl') == sag_summary['output_sha256']['geometry.jsonl']
sag_records = [json.loads(line) for line in (sagittal_dir / 'geometry.jsonl').open()]
sag_images = np.load(sagittal_dir / 'model_input.npy', mmap_mode='r')
affine = torch.tensor(np.asarray([row['model_pixel_to_ccf_ref9_ap_dv_ml_um']
    for row in sag_records]), dtype=torch.float64)
ouv = torch.stack((affine[:, :, 2], side * affine[:, :, 0],
                   side * affine[:, :, 1]), 1)
sag_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
sag_normals = full_frame_state_to_components(sag_states)[1][..., :, 2]
sag_reflection = sag_normals.gather(1, sag_normals.abs().argmax(-1)[:, None])[:, 0] < 0
ouv[sag_reflection, 0] += (side - 1) / side * ouv[sag_reflection, 1]
ouv[sag_reflection, 1] *= -1
sag_states = full_frame_state_from_components(*physical_ouv_to_frame(ouv)).float()

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().train()
attach_global_plane_matcher(model, enabled=True)
spatial = WholeSliceAtlasFeedback083().cuda().train()
parent_state = torch.load(parent, map_location='cpu', weights_only=True)
assert parent_state['step'] == 2000 and parent_state['calibrated'] is False
model.load_state_dict(parent_state['model'], strict=True)
spatial.load_state_dict(parent_state['spatial'], strict=True)
teacher = copy.deepcopy(model).eval().requires_grad_(False)
teacher.load_state_dict(torch.load(teacher_checkpoint, map_location='cpu',
    weights_only=True)['model'], strict=False)
model.requires_grad_(False)
spatial.requires_grad_(False)
pose_modules = (model.pose, model.anchor_pose, model.anchor_global)
shared_modules = (model.encoder, model.lateral)
map_modules = (model.warp_shared, model.warp_condition,
               model.atlas_encoder, model.pair, model.warp)
for module in (*pose_modules, *shared_modules, *map_modules,
               model.global_plane_matcher_120, spatial.image, spatial.atlas):
    module.requires_grad_(True)
spatial.log_temperature.requires_grad_(True)
groups = [
    {'params': [p for module in pose_modules for p in module.parameters()], 'base_lr': 1e-5},
    {'params': [p for module in shared_modules for p in module.parameters()], 'base_lr': 2e-6},
    {'params': list(model.global_plane_matcher_120.parameters()), 'base_lr': 5e-5},
    {'params': [p for module in map_modules for p in module.parameters()], 'base_lr': 1e-5},
    {'params': [*spatial.image.parameters(), *spatial.atlas.parameters(),
                spatial.log_temperature], 'base_lr': 3e-5},
]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
optimizer.load_state_dict(parent_state['optimizer'])
parameters = [p for group in groups for p in group['params']]
source_files = ('train_joint_in_path_correspondence_128.py',
    'in_path_global_matcher_128.py', 'joint_pose_map_121.py',
    'arbitrary_plane_one_shot_model.py', 'global_plane_matcher_120.py',
    'whole_slice_atlas_feedback_083.py', 'whole_slice_atlas_feedback_081.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_reserved_real_stream_v8.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py')
config = {'arm': arm, 'seed': seed, 'updates': updates,
    'synthetic_per_update': 2, 'synthetic_presentations': 12000,
    'side': side, 'valid_sites': sites, 'blind_count': blind_count,
    'checkpoints': checkpoints, 'resume_every': 100,
    'parent_checkpoint': str(parent), 'parent_checkpoint_sha256': sha(parent),
    'teacher_checkpoint_sha256': sha(teacher_checkpoint),
    'parent_completion_sha256': sha(parent_dir / 'completed.json'),
    'manifest_completion_sha256': ('TRAIN_ONLY_PREFLIGHT' if preflight
        else sha(manifest_dir / 'completed.json')),
    'manifest_updates_sha256': manifest_receipt['updates_sha256'],
    'draws_sha256': manifest_receipt['draws_sha256'],
    'protocol_sha256': sha(protocol),
    'source_sha256': {name: sha(source / name) for name in source_files},
    'synthetic_provenance': context['provenance'],
    'coronal_bindings': coronal['bindings'],
    'sagittal_summary_sha256': sha(sagittal_dir / 'summary.json'),
    'sagittal_output_sha256': sag_summary['output_sha256'],
    'loss_weights': {'direct': 1., 'action_kl': .5, 'action_expected_cost': .25,
        'map': 1., 'selected_rigid': .5, 'fine_ce': .05, 'coarse_ce': .05,
        'warp': .03, 'exact_no_shift': .1, 'near_shift': .1,
        'in_path_ce': .003 if arm == 'treatment' else 0., 'fit': 0.,
        'real_weak_pose': .5, 'coronal_teacher_retention': 2.,
        'coronal_posterior_risk': 2.},
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False, 'external_pretrained_weights_used': False}
config = json.loads(json.dumps(config))
resume_path = run / 'resume.pt'
if preflight:
    start, audit = 0, None
elif resume_path.exists():
    assert json.loads((run / 'config.json').read_text()) == config
    assert sha(run / 'draws.jsonl') == manifest_receipt['draws_sha256']
    state = torch.load(resume_path, map_location='cpu', weights_only=True)
    start = state['step']
    model.load_state_dict(state['model'], strict=True)
    spatial.load_state_dict(state['spatial'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    audit = state['gradient_audit']
    previous = [line for line in (run / 'training.jsonl').read_text().splitlines()
        if json.loads(line)['update'] <= start]
    (run / 'training.jsonl').write_text('\n'.join(previous) + ('\n' if previous else ''))
    print(json.dumps({'event': 'resumed', 'update': start}), flush=True)
else:
    assert not run.exists()
    run.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    shutil.copyfile(manifest_dir / 'draws.jsonl', run / 'draws.jsonl')
    start, audit = 0, None


def save(path, step):
    temp = path.with_suffix('.tmp')
    torch.save({'step': step, 'model': model.state_dict(), 'spatial': spatial.state_dict(),
        'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'gradient_audit': audit,
        'config': config, 'calibrated': False}, temp)
    os.replace(temp, path)


if start == 0 and not preflight:
    save(run / 'joint_step_00000.pt', 0)
    save(resume_path, 0)
started = time.perf_counter()
with (nullcontext() if preflight else (run / 'training.jsonl').open('a')) as log:
    for step in range(start + 1, (1 if preflight else updates) + 1):
        record = schedule[step - 1]
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * step / updates))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * decay
        optimizer.zero_grad(set_to_none=True)
        metrics = []
        for slot, item in enumerate(record['synthetic']):
            with torch.no_grad():
                sample = sample_one_shot_slide_artifacts_v3(context,
                    [item['subject_index']], item['draw_seed'], side=side)
            assert bool(sample['eligible'][0])
            if preflight:
                assert item['physical_section_id'] == sample['provenance'][0]['physical_section_id']
            else:
                assert all(item[key] == value for key, value in
                    json.loads(json.dumps(sample['provenance'][0])).items())
            assert sample['provenance'][0]['split'] == 'train'
            inputs = sample['inputs']
            prediction = model.predict(inputs)
            flags = torch.arange(2, device='cuda')[None, None].expand(1, model.modes, 2)
            states160 = prediction['state'][:, :, None].expand(-1, -1, 2, -1).reshape(1, -1, 12)
            reflections160 = flags.reshape(1, -1)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            site_rng = torch.Generator(device='cuda').manual_seed(item['site_seed'])
            pixel = torch.multinomial(sample['valid_mask'].flatten(1).float(),
                sites, replacement=True, generator=site_rng)
            chart = torch.stack((pixel.remainder(side),
                pixel.div(side, rounding_mode='floor')), -1).float() / side
            physical = rigid_cost(states160, reflections160,
                sample['state'], sample['reflection'], chart)
            target_mass = F.softmax(-physical.detach() / .7, -1)
            true_normal = full_frame_state_to_components(sample['state'])[1][..., :, 2]
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(1, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            direct = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1).mean()
                + .5 * F.kl_div(prior, target_mass, reduction='batchmean')
                + .5 * physical[:, :2 * model.base_modes].min(-1).values.mean()
                + (neighbourhood * near_cost).sum(-1).mean())
            with torch.no_grad():
                beam = torch.cat((prior[:, :32].topk(8, -1).indices,
                    prior[:, 32:].topk(6, -1).indices + 32), -1)
                normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
                row = torch.arange(1, device='cuda')
                for _ in range(blind_count - 14):
                    chosen = normals[row[:, None], beam // 2]
                    similarity = (normals[:, 16:, None] * chosen[:, None]).sum(-1).abs().amax(-1)
                    diversity = -similarity
                    diversity.scatter_(1, beam[:, 8:] // 2 - 16, -2.)
                    anchor = diversity.argmax(-1)
                    reflected = prior[:, 32:].reshape(1, 64, 2)[row, anchor].argmax(-1)
                    beam = torch.cat((beam, (2 * (anchor + 16) + reflected)[:, None]), -1)
                blind_mode, blind_reflection = beam // 2, beam % 2
            match16 = global_plane_match_128(model, prediction, blind_mode, blind_reflection,
                sample['offsets'], sample['weights'], context['atlas'],
                inputs.shape[-2:], side=24)
            original_cost = rigid_cost(match16['input_state'], blind_reflection,
                sample['state'], sample['reflection'], chart)
            corrected_cost = rigid_cost(match16['state'], blind_reflection,
                sample['state'], sample['reflection'], chart)
            action_cost = torch.stack((original_cost, corrected_cost), -1).flatten(1)
            action_scores = torch.stack((match16['input_score'], match16['score']), -1).flatten(1)
            action_target = F.softmax(-action_cost.detach() / .7, -1)
            action_kl = F.kl_div(F.log_softmax(action_scores, -1), action_target,
                reduction='batchmean')
            action_expected = (F.softmax(action_scores, -1) * action_cost.detach()).sum(-1).mean()
            with torch.no_grad():
                best = original_cost.argmin(-1)
                best_cost = original_cost.gather(1, best[:, None])
                wrong_eligible = original_cost >= best_cost + 1.
                wrong = original_cost.masked_fill(~wrong_eligible, float('inf')).argmin(-1)
                wrong_available = wrong_eligible.gather(1, wrong[:, None])[:, 0]
                chosen = beam.gather(1, torch.stack((best, wrong), -1))
                jitter_rng = torch.Generator().manual_seed(item['jitter_seed'])
                jitter_limits = sample['state'].new_tensor(
                    (.2, .2, .2, 1500., 1500., 1500., .08, .08, .05))
                jitter = (2 * torch.rand(1, 9, generator=jitter_rng).cuda() - 1) * jitter_limits
                near_state = compose_full_frame_state(sample['state'], jitter)
            training_mode = torch.cat((chosen // 2, blind_mode[:, :1].expand(-1, 2)), 1)
            training_reflection = torch.cat((chosen % 2,
                sample['reflection'][:, None].expand(-1, 2)), 1)
            row = torch.arange(1, device='cuda')[:, None]
            candidate_state = torch.cat((prediction['state'][row, chosen // 2],
                sample['state'][:, None], near_state[:, None]), 1)
            joint = joint_forward_121(model, prediction, inputs, training_mode,
                training_reflection, context['atlas'], sample['offsets'],
                sample['weights'], candidate_state=candidate_state)
            mapped = joint['mapped']
            targets = joint_target_loss_121(mapped, sample['valid_mask'],
                sample['centre'], sample['state'], sample['reflection'], pixel)
            best_gate = torch.sigmoid((2.5 - best_cost.detach()) / .5)[:, 0]
            map_loss = (targets['map_loss'][:, 2:4].mean()
                + .5 * (best_gate * targets['map_loss'][:, 0]).mean())
            selected_rigid = (targets['rigid_loss'][:, 2:4].mean()
                + .5 * (best_gate * targets['rigid_loss'][:, 0]).mean())
            spatial_ce = joint_spatial_ce_121(spatial, prediction,
                joint['match']['state'][:, 2:4], training_reflection[:, 2:4],
                context['atlas'], sample['offsets'], sample['weights'],
                sample['centre'], sample['valid_mask'])
            local = mapped['local_displacement_um'][:, (0, 2, 3)] / 1000
            warp = (local.square().mean()
                + .3 * (local[..., 1:] - local[..., :-1]).square().mean()
                + .3 * (local[..., 1:, :] - local[..., :-1, :]).square().mean())
            exact_gate = F.softplus(joint['match']['match_logit'][:, 2]).mean()
            near_gate = F.softplus(-joint['match']['match_logit'][:, 3]).mean()
            base_loss = (direct + .5 * action_kl + .25 * action_expected
                + map_loss + .5 * selected_rigid
                + .05 * (spatial_ce['fine_ce'] + spatial_ce['coarse_ce'])
                + .03 * warp + .1 * exact_gate + .1 * near_gate)
            ce_jitter_rng = torch.Generator().manual_seed(item['ce_jitter_seed'])
            ce_random = 2 * torch.rand(3, generator=ce_jitter_rng).cuda() - 1
            ce_jitter = torch.zeros(1, 9, device='cuda')
            ce_jitter[0, 2] = .05 * ce_random[0]
            ce_jitter[0, 3:5] = 750 * ce_random[1:]
            ce_state = compose_full_frame_state(sample['state'], ce_jitter)
            ce_modes = (chosen[:, :1] // 2).expand(-1, 2)
            ce_flags = torch.cat((sample['reflection'][:, None], chosen[:, :1] % 2), 1)
            ce_candidates = torch.cat((ce_state[:, None],
                prediction['state'][row, ce_modes[:, 1:2]]), 1)
            ce_match = global_plane_match_128(model, prediction, ce_modes, ce_flags,
                sample['offsets'], sample['weights'], context['atlas'],
                inputs.shape[-2:], side=24, candidate_state=ce_candidates,
                return_attention=True)
            positive = torch.cat((torch.ones_like(best_cost, dtype=torch.bool),
                best_cost <= 1.5), 1)
            ce = in_path_correspondence_ce_128(ce_match, ce_candidates, ce_flags,
                sample['centre'], sample['valid_mask'], sample['offsets'], positive)
            if audit is None and step == 1 and slot == 0:
                probes = (model.global_plane_matcher_120['source'][0].weight,
                    model.global_plane_matcher_120['atlas'][0].weight,
                    model.encoder[0][0].weight, model.pose[-1].weight)
                base_grad = torch.autograd.grad(base_loss, probes,
                    retain_graph=True, allow_unused=True)
                ce_grad = torch.autograd.grad(ce['loss'], probes,
                    retain_graph=True, allow_unused=True)
                base_norm = [0. if value is None else float(value.norm()) for value in base_grad]
                ce_norm = [0. if value is None else float(value.norm()) for value in ce_grad]
                assert all(math.isfinite(value) for value in base_norm + ce_norm)
                assert all(value > 0 for value in ce_norm[:3])
                audit = {'base_gradient_norm': base_norm, 'ce_gradient_norm': ce_norm,
                    'weighted_ce_to_base_ratio': [.003 * ce_norm[i] / max(base_norm[i], 1e-12)
                        for i in range(3)], 'probe_order':
                    ['matcher_source', 'matcher_atlas', 'shared_encoder', 'direct_pose_head'],
                    'base_scope': 'complete synthetic base_loss before weak-real objective'}
                print(json.dumps({'event': 'train_gradient_audit', **audit}), flush=True)
                if preflight:
                    print(json.dumps({'event': 'train_only_preflight_complete',
                        'weighted_ce_coefficient': .003,
                        'source_ratio': audit['weighted_ce_to_base_ratio'][0],
                        'atlas_ratio': audit['weighted_ce_to_base_ratio'][1],
                        'encoder_ratio': audit['weighted_ce_to_base_ratio'][2],
                        'pose_ratio': .003 * ce_norm[3] / max(base_norm[3], 1e-12),
                        'jittered_label_cells': int(ce['pixels_per_candidate'][0, 0]),
                        'near_label_cells': int(ce['pixels_per_candidate'][0, 1])}), flush=True)
                    sys.exit(0)
            loss = base_loss + (.003 * ce['loss'] if arm == 'treatment' else 0.)
            (loss / 2).backward()
            with torch.no_grad():
                selected_action = action_scores.argmax(-1)
                metrics.append({'physical_section_id': item['physical_section_id'],
                    'direct': float(direct), 'action_kl': float(action_kl),
                    'action_expected_cost': float(action_expected),
                    'map_mm': float(map_loss), 'rigid_selected_mm': float(selected_rigid),
                    'spatial_fine_ce': float(spatial_ce['fine_ce']),
                    'spatial_coarse_ce': float(spatial_ce['coarse_ce']),
                    'in_path_ce': float(ce['loss']),
                    'in_path_pixels_jittered': int(ce['pixels_per_candidate'][0, 0]),
                    'in_path_pixels_predicted_near': int(ce['pixels_per_candidate'][0, 1]),
                    'in_path_predicted_eligible': bool(positive[0, 1]),
                    'warp': float(warp),
                    'blind16_input_selected_cost': float(original_cost.gather(
                        1, match16['input_score'].argmax(-1)[:, None]).mean()),
                    'blind16_joint_selected_cost': float(action_cost.gather(
                        1, selected_action[:, None]).mean()),
                    'blind16_best_original_cost': float(original_cost.min(-1).values.mean()),
                    'blind16_best_corrected_cost': float(corrected_cost.min(-1).values.mean()),
                    'best_blind_index': int(best[0]), 'wrong_blind_index': int(wrong[0]),
                    'loss': float(loss)})

        donor = record['coronal_donor']
        section = record['coronal_section']
        real_coronal = sample_reserved_real_train(coronal, donor, [section])
        assert real_coronal['identities'][0] == record['coronal_identity']
        coronal_inputs = F.interpolate(real_coronal['inputs'], (side, side),
            mode='bilinear', align_corners=False)
        sag_index = record['sagittal_index']
        assert {key: sag_records[sag_index][key] for key in
            ('donor_id', 'specimen_id', 'experiment_id', 'section_id')} == record['sagittal_identity']
        sag_inputs = torch.zeros(1, 5, side, side, device='cuda')
        sag_inputs[:, :1] = torch.from_numpy(np.asarray(
            sag_images[sag_index:sag_index + 1]).copy()).to('cuda')
        real_inputs = torch.cat((coronal_inputs, sag_inputs))
        real_state = torch.cat((real_coronal['state'],
            sag_states[sag_index:sag_index + 1].to('cuda')))
        real_reflection = torch.cat((real_coronal['reflection'],
            sag_reflection[sag_index:sag_index + 1].long().to('cuda')))
        real_prediction = model.predict(real_inputs)
        real_prior = (real_prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-real_prediction['reflection_logit']),
            F.logsigmoid(real_prediction['reflection_logit'])), -1)).flatten(1)
        real_flags = torch.arange(2, device='cuda')[None, None].expand(2, model.modes, 2)
        real_states = real_prediction['state'][:, :, None].expand(-1, -1, 2, -1).reshape(2, -1, 12)
        real_five = rigid_points_090(real_states, real_flags.reshape(2, -1), corners)
        weak_five = rigid_points_090(real_state, real_reflection, corners)
        real_cost = (real_five - weak_five[:, None]).norm(dim=-1).mean(-1) / 1000
        real_mass = F.softmax(-real_cost.detach() / .7, -1)
        real_loss = (-1.5 * torch.logsumexp(real_prior - real_cost / 1.5, -1)
            + .5 * F.kl_div(real_prior, real_mass, reduction='none').sum(-1)
            + .5 * real_cost.min(-1).values).mean()
        with torch.no_grad():
            teacher_prediction = teacher.predict(coronal_inputs)
            teacher_prior = (teacher_prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-teacher_prediction['reflection_logit']),
                F.logsigmoid(teacher_prediction['reflection_logit'])), -1)).flatten(1)
            teacher_mass = teacher_prior.exp()
            teacher_states = teacher_prediction['state'][:, :, None].expand(
                -1, -1, 2, -1).reshape(1, -1, 12)
            teacher_five = rigid_points_090(teacher_states,
                real_flags[:1].reshape(1, -1), corners)
        teacher_kl = F.kl_div(real_prior[:1], teacher_mass, reduction='batchmean')
        teacher_geometry_mm = (teacher_mass * (real_five[:1] - teacher_five
            ).norm(dim=-1).mean(-1) / 1000).sum(-1).mean()
        coronal_posterior_risk_mm = (real_prior[0].exp() *
            F.relu(real_cost[0] - 1.)).sum()
        real_objective = (.5 * real_loss + 2. * (teacher_kl + teacher_geometry_mm)
            + 2. * coronal_posterior_risk_mm)
        real_objective.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        real_row = {'coronal_identity': real_coronal['identities'][0],
            'sagittal_identity': record['sagittal_identity'],
            'weak_pose_loss': float(real_loss.detach()),
            'coronal_teacher_kl': float(teacher_kl.detach()),
            'coronal_teacher_geometry_mm': float(teacher_geometry_mm.detach()),
            'coronal_posterior_risk_mm': float(coronal_posterior_risk_mm.detach()),
            'real_objective': float(real_objective.detach()),
            'coronal_selected_weak_mm': float(real_cost[0,
                real_prior[0].argmax()].detach()),
            'sagittal_selected_weak_mm': float(real_cost[1,
                real_prior[1].argmax()].detach())}
        row = {'update': step, 'synthetic_presentations': 2 * step,
            'physical_section_ids': [item['physical_section_id']
                for item in record['synthetic']],
            'synthetic': metrics, 'real': real_row,
            'gradient_norm': float(gradient),
            'seconds_since_process_start': time.perf_counter() - started}
        log.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 1 or step % 1000 == 0:
            print(json.dumps({'event': 'train_milestone', 'arm': arm,
                'update': step, 'synthetic_presentations': 2 * step,
                'direct': sum(item['direct'] for item in metrics) / 2,
                'selected_rigid_action_cost': sum(item['blind16_joint_selected_cost']
                    for item in metrics) / 2,
                'in_path_ce': sum(item['in_path_ce'] for item in metrics) / 2,
                'jittered_label_cells': sum(item['in_path_pixels_jittered'] for item in metrics),
                'near_label_cells': sum(item['in_path_pixels_predicted_near'] for item in metrics),
                'coronal_weak_mm': real_row['coronal_selected_weak_mm'],
                'sagittal_weak_mm': real_row['sagittal_selected_weak_mm'],
                'gradient_norm': float(gradient)}), flush=True)
        if step % 100 == 0 or step in checkpoints:
            log.flush()
            if step in checkpoints:
                save(run / f'joint_step_{step:05d}.pt', step)
            save(resume_path, step)

log_sha = sha(run / 'training.jsonl')
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'accepted_synthetic': 2 * updates,
    'weak_real_coronal_presentations': updates,
    'weak_real_sagittal_presentations': updates,
    'parent_checkpoint_sha256': sha(parent),
    'teacher_checkpoint_sha256': sha(teacher_checkpoint),
    'manifest_completion_sha256': sha(manifest_dir / 'completed.json'),
    'manifest_updates_sha256': manifest_receipt['updates_sha256'],
    'protocol_sha256': sha(protocol), 'source_sha256': config['source_sha256'],
    'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': log_sha,
    'checkpoint_sha256': {str(step): sha(run / f'joint_step_{step:05d}.pt')
        for step in checkpoints},
    'gradient_audit': audit,
    'calibrated': False, 'public_benchmark_used': False,
    'expert_real_truth_used': False,
    'external_pretrained_weights_used': False}, indent=2))
print(json.dumps({'event': 'train_complete', 'arm': arm, 'updates': updates,
    'seconds_since_process_start': time.perf_counter() - started}), flush=True)
