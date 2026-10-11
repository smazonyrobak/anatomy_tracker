"""Matched feedback with retained pose optimizer and four-section updates."""

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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_156 import JointAnatomyModel156


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


seed, side, updates, presentations_per_update = 20261011158, 256, 1000, 4
checkpoints = (0, 500, 1000)
train = root / 'runs/joint_fitter_155_train151_001'
train_done = json.loads((train / 'completed.json').read_text())
parent_step = train_done['selected_step']
parent_checkpoint = train / f'joint_fitter_step_{parent_step:05d}.pt'
pose_train = root / 'runs/joint_pose_curriculum_154_train151_001'
pose_done = json.loads((pose_train / 'completed.json').read_text())
pose_step = pose_done['selected_step']
pose_checkpoint = pose_train / f'joint_pose_step_{pose_step:05d}.pt'
run = root / 'runs/joint_feedback_158_train151_001'
source = Path(__file__).resolve().parent
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()
assert sha(parent_checkpoint) == train_done['checkpoint_sha256'][str(parent_step)]
assert sha(pose_checkpoint) == pose_done['checkpoint_sha256'][str(pose_step)]
assert train_done['parent_checkpoint_sha256'] == sha(pose_checkpoint)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64
assert all(base['lineage']['split'] == 'train' for base in context['bases'])
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
    pose_parent = torch.load(pose_checkpoint, map_location='cpu', weights_only=True)
pose_keys = [key for key in parent['model'] if key.startswith('pose.')]
assert all(torch.equal(parent['model'][key], pose_parent['model'][key]) for key in pose_keys)
old_optimizer = pose_parent['optimizer']
pose_group = old_optimizer['param_groups'][0]
pose_moments = {'state': {key: value for key, value in old_optimizer['state'].items()
                          if key in pose_group['params']}, 'param_groups': [pose_group]}
del pose_parent

protocol = {'seed': seed, 'side': side, 'updates_per_arm': updates,
    'presentations_per_update': presentations_per_update,
    'arms': ['feedback', 'detached_control'], 'checkpoints': checkpoints,
    'gradient_bases': list(range(60)), 'inner_bases': list(range(60, 64)),
    'inner_sections_per_base': 12, 'appearance_schedule': 'v4,v4,v3',
    'candidate_selection': 'best physical geometry among truth-blind direct top16; fit if <=1.5 mm',
    'fitter': '155 matcher and field frozen in both arms; 156 damped rigid correction',
    'feedback': '0.5 times supervised corrected rigid tissue/canvas/normal loss; live predicted state',
    'control': 'same input, checkpoint, frozen fitter, direct loss and candidate rule; fitting state detached',
    'pose_lr': 2e-5, 'pose_gradient_clip': 2., 'minimum_match_confidence_mass': 10.,
    'pose_optimizer': 'retained 154 AdamW pose moments; four fresh independent sections per update',
    'chart_seed': 'seed plus one-based presentation index, identical between arms',
    'checkpoint_selection': 'mean direct top1 mm + 0.5 mean oracle best mm + 1*(1-top16 near fraction)',
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'pose_optimizer_checkpoint_sha256': sha(pose_checkpoint),
    'pretrained_weights_used': False, 'real_images_used': False,
    'dev_or_public_benchmark_used': False, 'final_animals_used': False,
    'calibrated': False}
source_names = ('train_joint_feedback_158.py', 'joint_anatomy_model_156.py',
    'coarse_atlas_pose_150.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_ribbon_v6.py',
    'arbitrary_plane_geometry.py', 'global_atlas_contrast_090.py')
config = {'protocol': protocol, 'source_sha256': {name: sha(source / name)
    for name in source_names}, 'synthetic_provenance': context['provenance'],
    'torch': torch.__version__, 'numpy': np.__version__}
run.mkdir(parents=True, exist_ok=False)
(run / 'protocol.json').write_text(json.dumps(protocol, indent=2))
config['protocol_sha256'] = sha(run / 'protocol.json')
(run / 'config.json').write_text(json.dumps(config, indent=2))

corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
reflection_options = torch.arange(2, device='cuda')[None, None]
axis24 = (torch.arange(24, device='cuda') + .5) / 24
y24, x24 = torch.meshgrid(axis24, axis24, indexing='ij')
grid24 = (2 * torch.stack((x24, y24), -1) - 1)[None]


def draw(base, draw_seed, appearance):
    sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
               else sample_one_shot_slide_artifacts_v4)
    with torch.no_grad():
        original = sample_panel_geometry_train_151(context, [base], draw_seed, side)
        sample = sampler(context, [base], draw_seed, side, source_sample=original)
        valid_queries = int((F.grid_sample(sample['valid_mask'][:, None].float(), grid24,
            mode='bilinear', align_corners=False) > .99).sum())
    return sample, valid_queries


def direct_loss(sample):
    pixels = torch.multinomial(sample['valid_mask'].flatten(1).float(), 128, replacement=True)
    chart = torch.stack((pixels.remainder(side), pixels.div(side, rounding_mode='floor')),
                        -1).float() / side
    prediction = model.pose.predict(sample['inputs'])
    proposed_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1).float()
    proposed_reflections = reflection_options.expand(1, model.pose.modes, 2)
    prior = (prediction['log_mass'].float()[..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'].float()),
        F.logsigmoid(prediction['reflection_logit'].float())), -1)).flatten(1)
    true = rigid_points_090(sample['state'], sample['reflection'], chart)
    proposed = rigid_points_090(proposed_states, proposed_reflections, chart[:, None, None])
    true_five = rigid_points_090(sample['state'], sample['reflection'], corners)
    proposed_five = rigid_points_090(proposed_states, proposed_reflections, corners)
    tissue_um = (proposed - true[:, None, None]).norm(dim=-1).mean(-1)
    five_um = (proposed_five - true_five[:, None, None]).norm(dim=-1).mean(-1)
    normal = full_frame_state_to_components(prediction['state'].float())[1][..., :, 2]
    true_normal = full_frame_state_to_components(sample['state'])[1][..., :, 2]
    normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
    physical = (.75 * tissue_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
    target_mass = F.softmax(-physical.detach() / .7, -1)
    nearest = (true_normal @ model.pose.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
    near_mode = model.pose.base_modes + nearest.indices
    near_cost = physical.reshape(1, model.pose.modes, 2).min(-1).values.gather(1, near_mode)
    neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
    loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
            + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
            + .5 * physical[:, :2 * model.pose.base_modes].min(-1).values
            + (neighbourhood * near_cost).sum(-1)).mean()
    return loss, chart


def on_policy(sample, chart):
    with torch.no_grad():
        prediction = model.pose.predict(sample['inputs'])
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = prior.topk(16, -1).indices[0]
        states = prediction['state'][0, branch // 2]
        reflections = branch % 2
        truth = rigid_points_090(sample['state'], sample['reflection'], chart)
        proposed = rigid_points_090(states, reflections, chart)
        errors = (proposed - truth).norm(dim=-1).mean(-1) / 1000
        best = int(errors.argmin())
        choice = branch[best:best + 1][None]
        state = states[best:best + 1][None]
    return choice, state, float(errors[best])


inner = []
inner_rng = np.random.default_rng(seed + 1)
with (run / 'inner_draws.jsonl').open('w') as stream:
    for base in range(60, 64):
        for slot in range(12):
            appearance = 'v3' if slot % 3 == 0 else 'v4'
            while True:
                draw_seed = int(inner_rng.integers(0, 2**63 - 1, dtype=np.int64))
                sample, valid_queries = draw(base, draw_seed, appearance)
                used = bool(sample['eligible'][0]) and valid_queries >= 4
                stream.write(json.dumps({'base_index': base, 'slot': slot,
                    'draw_seed': draw_seed, 'appearance_version': appearance,
                    'used': used, 'valid_queries': valid_queries,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    break
            pixels = sample['valid_mask'][0].flatten().nonzero()[:, 0]
            pixels = pixels[torch.linspace(0, len(pixels) - 1, min(512, len(pixels)),
                device='cuda').long()]
            chart = torch.stack((pixels.remainder(side),
                pixels.div(side, rounding_mode='floor')), -1).float() / side
            inner.append({'sample': {key: sample[key].cpu() for key in
                ('inputs', 'state', 'reflection', 'offsets', 'weights')},
                'chart': chart.cpu(), 'base_index': base,
                'physical_section_id': sample['provenance'][0]['physical_section_id']})


def inner_score():
    model.eval()
    rows = []
    with torch.inference_mode():
        for item in inner:
            sample = {key: value.cuda() for key, value in item['sample'].items()}
            chart = item['chart'].cuda()
            prediction = model.pose.predict(sample['inputs'])
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = reflection_options.expand(1, model.pose.modes, 2)
            truth = rigid_points_090(sample['state'], sample['reflection'], chart)
            proposed = rigid_points_090(states, reflections, chart[None, None])
            errors = ((proposed - truth[:, None, None]).norm(dim=-1).mean(-1) / 1000).flatten()
            order = prior.argsort(descending=True)
            best = order[:16][errors[order[:16]].argmin()]
            row = {'base_index': item['base_index'],
                'physical_section_id': item['physical_section_id'],
                'oracle_best_mm': float(errors.min()),
                'direct_top1_mm': float(errors[order[0]]),
                'top16_near': bool((errors[order[:16]] <= 1.5).any()),
                'near_prefit_mm': float(errors[best])}
            if row['top16_near']:
                choice = best.reshape(1, 1)
                fitted = model(sample['inputs'], context['atlas'], sample['offsets'],
                    sample['weights'], choice.div(2, rounding_mode='floor'),
                    choice.remainder(2))
                corrected = rigid_points_090(fitted['corrected_state'].float(),
                                             choice.remainder(2), chart[None])
                row['near_postfit_mm'] = float((corrected - truth[:, None]).norm(dim=-1).mean() / 1000)
            rows.append(row)
    model.train()
    model.matcher.eval()
    model.field_body.eval()
    model.field_out.eval()
    metrics = {'n': len(rows),
        'direct_top1_mm': float(np.mean([row['direct_top1_mm'] for row in rows])),
        'oracle_best_mm': float(np.mean([row['oracle_best_mm'] for row in rows])),
        'top16_near_fraction': float(np.mean([row['top16_near'] for row in rows])),
        'near_prefit_mm': float(np.mean([row['near_prefit_mm'] for row in rows if row['top16_near']])),
        'near_postfit_mm': float(np.mean([row['near_postfit_mm'] for row in rows
                                         if row['top16_near']]))}
    metrics['selection_score'] = (metrics['direct_top1_mm'] + .5 * metrics['oracle_best_mm']
                                  + 1 - metrics['top16_near_fraction'])
    return metrics, rows


def save(arm, step, optimizer):
    path = run / arm / f'joint_feedback_step_{step:05d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save({'step': step, 'arm': arm, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'config': config,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'calibrated': False}, temporary)
    os.replace(temporary, path)


draw_rng = np.random.default_rng(seed)
accepted_draws = []
arm_results = {}
started = time.perf_counter()
with (run / 'draws.jsonl').open('w') as draws:
    for arm in protocol['arms']:
        torch.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 3)
        model = JointAnatomyModel156().cuda().train()
        model.load_state_dict(parent['model'], strict=True)
        model.matcher.requires_grad_(False).eval()
        model.field_body.requires_grad_(False).eval()
        model.field_out.requires_grad_(False).eval()
        optimizer = torch.optim.AdamW(model.pose.parameters(), lr=2e-5, weight_decay=1e-4)
        assert len(optimizer.param_groups[0]['params']) == len(pose_group['params'])
        optimizer.load_state_dict(pose_moments)
        optimizer.param_groups[0]['lr'] = 2e-5
        directory = run / arm
        directory.mkdir(parents=True, exist_ok=False)
        scores = {}
        save(arm, 0, optimizer)
        metrics, rows = inner_score()
        scores[0] = metrics
        with (directory / 'training.jsonl').open('w') as training, \
                (directory / 'inner_scores.jsonl').open('w') as selections:
            selections.write(json.dumps({'step': 0, 'metrics': metrics, 'rows': rows}) + '\n')
            print(json.dumps({'event': 'inner_checkpoint', 'arm': arm,
                              'step': 0, 'metrics': metrics}), flush=True)
            for step in range(1, updates + 1):
                optimizer.zero_grad(set_to_none=True)
                batch_rows = []
                for slot in range(presentations_per_update):
                    presentation = (step - 1) * presentations_per_update + slot + 1
                    appearance = 'v3' if presentation % 3 == 0 else 'v4'
                    if arm == 'feedback':
                        while True:
                            base = int(draw_rng.integers(60))
                            draw_seed = int(draw_rng.integers(0, 2**63 - 1, dtype=np.int64))
                            sample, valid_queries = draw(base, draw_seed, appearance)
                            used = bool(sample['eligible'][0]) and valid_queries >= 4
                            draws.write(json.dumps({'update': step, 'slot': slot,
                                'presentation': presentation, 'base_index': base,
                                'draw_seed': draw_seed, 'appearance_version': appearance,
                                'used': used, 'valid_queries': valid_queries,
                                'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                            if used:
                                accepted_draws.append((base, draw_seed, appearance))
                                break
                    else:
                        base, draw_seed, appearance = accepted_draws[presentation - 1]
                        sample, valid_queries = draw(base, draw_seed, appearance)
                        assert bool(sample['eligible'][0]) and valid_queries >= 4

                    torch.manual_seed(seed + presentation)
                    torch.cuda.manual_seed_all(seed + presentation)
                    direct, chart = direct_loss(sample)
                    (direct / presentations_per_update).backward()
                    choice, detached_state, prefit_mm = on_policy(sample, chart)
                    active = prefit_mm <= 1.5
                    feedback_loss = confidence_mass = 0.
                    if active:
                        mode, reflection = choice.div(2, rounding_mode='floor'), choice.remainder(2)
                        output = model(sample['inputs'], context['atlas'], sample['offsets'],
                            sample['weights'], mode, reflection,
                            state_override=detached_state if arm == 'detached_control' else None)
                        confidence_mass = float(output['matcher']['confidence'].sum().detach())
                        active = confidence_mass >= 10.
                        if active:
                            truth = rigid_points_090(sample['state'], sample['reflection'], chart)
                            corrected = rigid_points_090(output['corrected_state'].float(),
                                                         reflection, chart[None])
                            truth_five = rigid_points_090(sample['state'], sample['reflection'], corners)
                            corrected_five = rigid_points_090(output['corrected_state'].float(),
                                                              reflection, corners)
                            corrected_normal = full_frame_state_to_components(
                                output['corrected_state'].float())[1][..., :, 2]
                            true_normal = full_frame_state_to_components(sample['state'])[1][..., :, 2]
                            tissue_mm = (corrected - truth[:, None]).norm(dim=-1).mean() / 1000
                            canvas_mm = (corrected_five - truth_five[:, None]).norm(dim=-1).mean() / 1000
                            normal_mm = 4 * (1 - (corrected_normal * true_normal[:, None]).sum(-1)
                                             .abs().clamp_max(1)).mean()
                            combined = .75 * tissue_mm + .25 * canvas_mm + normal_mm
                            loss = F.smooth_l1_loss(combined,
                                torch.zeros_like(combined), beta=.25)
                            feedback_loss = float(loss.detach())
                            if arm == 'feedback':
                                (.5 * loss / presentations_per_update).backward()
                        del output
                    batch_rows.append({'update': step, 'slot': slot,
                        'presentation': presentation, 'arm': arm, 'base_index': base,
                        'physical_section_id': sample['provenance'][0]['physical_section_id'],
                        'draw_seed': draw_seed, 'appearance_version': appearance,
                        'direct_loss': float(direct.detach()), 'on_policy_prefit_mm': prefit_mm,
                        'fit_gradient_active': bool(active), 'match_confidence_mass': confidence_mass,
                        'postfit_loss': feedback_loss})
                    del sample, direct
                gradient = torch.nn.utils.clip_grad_norm_(model.pose.parameters(), 2.,
                                                            error_if_nonfinite=True)
                optimizer.step()
                for row in batch_rows:
                    row['pose_gradient_norm'] = float(gradient)
                    row['elapsed_seconds'] = time.perf_counter() - started
                    training.write(json.dumps(row, allow_nan=False) + '\n')
                if step == 1 or step % 100 == 0:
                    training.flush()
                    draws.flush()
                    print(json.dumps({'event': 'train_milestone', 'arm': arm,
                        'update': step, 'presentations': presentation,
                        'mean_direct_loss': float(np.mean([row['direct_loss'] for row in batch_rows])),
                        'fit_active_count': sum(row['fit_gradient_active'] for row in batch_rows),
                        'pose_gradient_norm': float(gradient),
                        'elapsed_seconds': time.perf_counter() - started}), flush=True)
                if step in checkpoints:
                    save(arm, step, optimizer)
                    metrics, rows = inner_score()
                    scores[step] = metrics
                    selections.write(json.dumps({'step': step, 'metrics': metrics,
                                                   'rows': rows}, allow_nan=False) + '\n')
                    selections.flush()
                    print(json.dumps({'event': 'inner_checkpoint', 'arm': arm,
                                      'step': step, 'metrics': metrics}), flush=True)
        selected = min(scores, key=lambda step: scores[step]['selection_score'])
        arm_results[arm] = {'selected_step': selected, 'selected_inner_metrics': scores[selected],
            'checkpoint_sha256': {str(step): sha(directory / f'joint_feedback_step_{step:05d}.pt')
                                  for step in checkpoints},
            'training_sha256': sha(directory / 'training.jsonl'),
            'inner_scores_sha256': sha(directory / 'inner_scores.jsonl')}
        print(json.dumps({'event': 'arm_complete', 'arm': arm,
                          'selected_step': selected, 'metrics': scores[selected]}), flush=True)
        del model, optimizer

assert len(accepted_draws) == updates * presentations_per_update
(run / 'completed.json').write_text(json.dumps({'updates_per_arm': updates,
    'presentations_per_update': presentations_per_update,
    'accepted_physical_sections_per_arm': updates * presentations_per_update,
    'arms': arm_results,
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'pose_optimizer_checkpoint_sha256': sha(pose_checkpoint),
    'protocol_sha256': sha(run / 'protocol.json'), 'config_sha256': sha(run / 'config.json'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'inner_draws_sha256': sha(run / 'inner_draws.jsonl'),
    'source_sha256': config['source_sha256'], 'calibrated': False,
    'pretrained_weights_used': False, 'real_images_used': False,
    'dev_or_public_benchmark_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'train_complete', 'arms': arm_results}), flush=True)
