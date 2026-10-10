"""Continue the scratch 153 lineage with fresh arbitrary-plane pose presentations.

TRAIN-only proposal-coverage curriculum; matcher and tissue map are retained but
frozen. No acquired, DEV, final-animal, or public benchmark data are read.
"""

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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_153 import JointAnatomyModel153


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


seed, side, updates, batch_size = 20261011154, 256, 8000, 4
checkpoints = (0, 2000, 4000, 6000, 8000)
source = Path(__file__).resolve().parent
parent_run = root / 'runs/joint_anatomy_model_153_train151_002'
parent_checkpoint = parent_run / 'joint_step_01300.pt'
run = root / 'runs/joint_pose_curriculum_154_train151_001'
assert source.drive.upper() == run.drive.upper() == 'I:' and not run.exists()

parent_completed = json.loads((parent_run / 'completed.json').read_text())
assert parent_completed['selected_step'] == 1300
assert sha(parent_checkpoint) == parent_completed['checkpoint_sha256']['1300']
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    parent = torch.load(parent_checkpoint, map_location='cpu', weights_only=True)
assert parent['step'] == 1300 and parent['calibrated'] is False
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64
assert all(base['lineage']['split'] == 'train' for base in context['bases'])
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
model = JointAnatomyModel153().cuda()
model.load_state_dict(parent['model'], strict=True)
model.matcher.requires_grad_(False)
model.field_body.requires_grad_(False)
model.field_out.requires_grad_(False)
optimizer = torch.optim.AdamW([
    {'params': list(model.pose.parameters()), 'lr': 1e-4},
    {'params': list(model.matcher.parameters()), 'lr': 0.},
    {'params': list(model.field_body.parameters()) + list(model.field_out.parameters()), 'lr': 0.},
], weight_decay=1e-4)
optimizer.load_state_dict(parent['optimizer'])
del parent

protocol = {'seed': seed, 'side': side, 'updates': updates, 'batch_size': batch_size,
    'parent_run': str(parent_run), 'parent_step': 1300,
    'parent_checkpoint_sha256': sha(parent_checkpoint),
    'gradient_bases': list(range(60)), 'inner_bases': list(range(60, 64)),
    'inner_sections_per_base': 16, 'inner_geometry_sites': 1024,
    'stage_1': 'updates 1..2000: v3, valid fraction >=0.12',
    'stage_2': 'updates 2001..4000: alternating v3/v4, valid fraction >=0.08',
    'stage_3': 'updates 4001..8000: v3 every third update, otherwise v4; all eligible',
    'appearance': 'one independently drawn physical plane and one appearance per accepted section',
    'optimizer': 'parent AdamW moments retained, matcher/map frozen, pose lr 1e-4 then 7e-5 then 5e-5',
    'inner_selection_score': 'mean oracle best-of-160 rigid mm + mean blind top-one rigid mm',
    'calibrated': False, 'pretrained_external_weights_used': False,
    'real_images_used': False, 'dev_or_public_benchmark_used': False,
    'final_animals_used': False}
source_names = ('train_joint_pose_curriculum_154.py', 'joint_anatomy_model_153.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_panel_geometry_train_151.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_ribbon_v6.py',
    'arbitrary_plane_geometry.py', 'arbitrary_plane_joint_model_v7.py',
    'arbitrary_plane_recurrent_model.py', 'coarse_atlas_pose_150.py',
    'global_atlas_contrast_090.py')
config = {'protocol': protocol,
          'source_sha256': {name: sha(source / name) for name in source_names},
          'synthetic_provenance': context['provenance'],
          'torch': torch.__version__, 'numpy': np.__version__}
run.mkdir(parents=True, exist_ok=False)
(run / 'protocol.json').write_text(json.dumps(protocol, indent=2))
config['protocol_sha256'] = sha(run / 'protocol.json')
(run / 'config.json').write_text(json.dumps(config, indent=2))

corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / side
reflection_options = torch.arange(2, device='cuda')[None, None]


def direct_loss(inputs, states, reflections, valid):
    batch = len(inputs)
    pixels = torch.multinomial(valid.flatten(1).float(), 128, replacement=True)
    chart = torch.stack((pixels.remainder(side), pixels.div(side, rounding_mode='floor')),
                        -1).float() / side
    prediction = model.pose.predict(inputs)
    proposed_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1).float()
    proposed_reflections = reflection_options.expand(batch, model.pose.modes, 2)
    prior = (prediction['log_mass'].float()[..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'].float()),
        F.logsigmoid(prediction['reflection_logit'].float())), -1)).flatten(1)
    true = rigid_points_090(states, reflections, chart)
    proposed = rigid_points_090(proposed_states, proposed_reflections, chart[:, None, None])
    true_five = rigid_points_090(states, reflections, corners)
    proposed_five = rigid_points_090(proposed_states, proposed_reflections, corners)
    tissue_um = (proposed - true[:, None, None]).norm(dim=-1).mean(-1)
    five_um = (proposed_five - true_five[:, None, None]).norm(dim=-1).mean(-1)
    normal = full_frame_state_to_components(prediction['state'].float())[1][..., :, 2]
    true_normal = full_frame_state_to_components(states)[1][..., :, 2]
    normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
    physical = (.75 * tissue_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
    target_mass = F.softmax(-physical.detach() / .7, -1)
    nearest = (true_normal @ model.pose.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
    near_mode = model.pose.base_modes + nearest.indices
    near_cost = physical.reshape(batch, model.pose.modes, 2).min(-1).values.gather(1, near_mode)
    neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
    loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
            + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
            + .5 * physical[:, :2 * model.pose.base_modes].min(-1).values
            + (neighbourhood * near_cost).sum(-1)).mean()
    return loss


inner = []
inner_rng = np.random.default_rng(seed + 1)
with (run / 'inner_draws.jsonl').open('w') as log:
    for base in range(60, 64):
        for slot in range(16):
            appearance = 'v3' if slot % 3 == 0 else 'v4'
            sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
                       else sample_one_shot_slide_artifacts_v4)
            while True:
                draw_seed = int(inner_rng.integers(0, 2**63 - 1, dtype=np.int64))
                with torch.no_grad():
                    original = sample_panel_geometry_train_151(context, [base], draw_seed, side)
                    sample = sampler(context, [base], draw_seed, side, source_sample=original)
                used = bool(sample['eligible'][0])
                log.write(json.dumps({'base_index': base, 'slot': slot, 'draw_seed': draw_seed,
                    'appearance_version': appearance, 'used': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    break
                del sample, original
            valid = sample['valid_mask'][0]
            pixels = valid.flatten().nonzero()[:, 0]
            selected = pixels[torch.linspace(0, len(pixels) - 1,
                min(1024, len(pixels)), device='cuda').long()]
            chart = torch.stack((selected.remainder(side),
                selected.div(side, rounding_mode='floor')), -1).float() / side
            normal = sample['provenance'][0]['physical_unit_normal_ap_dv_ml']
            inner.append({'inputs': sample['inputs'][0].cpu(),
                'state': sample['state'][0].cpu(),
                'reflection': sample['reflection'][0].cpu(),
                'chart': chart.cpu(), 'base_index': base, 'slot': slot,
                'appearance_version': appearance, 'valid_fraction': float(valid.float().mean()),
                'nearest_axis': ('AP', 'DV', 'ML')[int(np.abs(normal).argmax())],
                'obliquity_deg': float(np.degrees(np.arccos(max(abs(x) for x in normal))))})
            del sample, original


def inner_score():
    model.pose.eval()
    rows = []
    with torch.inference_mode():
        for item in inner:
            inputs = item['inputs'][None].cuda()
            chart = item['chart'].cuda()
            state = item['state'][None].cuda()
            reflection = item['reflection'][None].cuda()
            prediction = model.pose.predict(inputs)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            truth = rigid_points_090(state, reflection, chart)
            errors = []
            for start in range(0, model.pose.modes, 8):
                candidate = prediction['state'][:, start:start + 8, None].expand(-1, -1, 2, -1)
                reflections = reflection_options.expand(1, 8, 2)
                points = rigid_points_090(candidate, reflections, chart[None, None])
                errors.append(((points - truth[:, None, None]).norm(dim=-1).mean(-1) / 1000).flatten())
            error = torch.cat(errors)
            order = prior.argsort(descending=True)
            row = {'base_index': item['base_index'], 'slot': item['slot'],
                'appearance_version': item['appearance_version'],
                'valid_fraction': item['valid_fraction'],
                'nearest_axis': item['nearest_axis'], 'obliquity_deg': item['obliquity_deg'],
                'oracle_best_mm': float(error.min()), 'direct_top1_mm': float(error[order[0]]),
                'top2_near': bool((error[order[:2]] <= 1.5).any()),
                'top16_near': bool((error[order[:16]] <= 1.5).any()),
                'oracle_near': bool((error <= 1.5).any())}
            rows.append(row)
    model.pose.train()
    metrics = {'n': len(rows), 'oracle_near_fraction': float(np.mean([r['oracle_near'] for r in rows])),
        'top16_near_fraction': float(np.mean([r['top16_near'] for r in rows])),
        'top2_near_fraction': float(np.mean([r['top2_near'] for r in rows])),
        'mean_oracle_best_mm': float(np.mean([r['oracle_best_mm'] for r in rows])),
        'mean_direct_top1_mm': float(np.mean([r['direct_top1_mm'] for r in rows]))}
    metrics['selection_score_mm'] = metrics['mean_oracle_best_mm'] + metrics['mean_direct_top1_mm']
    return metrics, rows


def save(step):
    path = run / f'joint_pose_step_{step:05d}.pt'
    temporary = path.with_suffix('.tmp')
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(), 'config': config,
        'parent_checkpoint_sha256': protocol['parent_checkpoint_sha256'],
        'calibrated': False}, temporary)
    os.replace(temporary, path)


started = time.perf_counter()
attempted = accepted = 0
scores = {}
with (run / 'draws.jsonl').open('w') as draws, (run / 'training.jsonl').open('w') as training, \
        (run / 'inner_scores.jsonl').open('w') as selections:
    save(0)
    metrics, rows = inner_score()
    scores[0] = metrics['selection_score_mm']
    selections.write(json.dumps({'step': 0, 'metrics': metrics, 'rows': rows}) + '\n')
    selections.flush()
    print(json.dumps({'event': 'inner_checkpoint', 'step': 0, 'metrics': metrics}), flush=True)
    for step in range(1, updates + 1):
        if step <= 2000:
            stage, appearance, minimum_valid, lr = 1, 'v3', .12, 1e-4
        elif step <= 4000:
            stage, appearance, minimum_valid, lr = 2, ('v3' if step % 2 else 'v4'), .08, 7e-5
        else:
            stage, appearance, minimum_valid, lr = 3, ('v3' if step % 3 == 0 else 'v4'), 0., 5e-5
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = optimizer.param_groups[2]['lr'] = 0.
        sampler = (sample_one_shot_slide_artifacts_v3 if appearance == 'v3'
                   else sample_one_shot_slide_artifacts_v4)
        images, states, reflections, masks, fractions = [], [], [], [], []
        while len(images) < batch_size:
            need = batch_size - len(images)
            bases = [int(x) for x in rng.integers(0, 60, size=need)]
            draw_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
            with torch.no_grad():
                original = sample_panel_geometry_train_151(context, bases, draw_seed, side)
                sample = sampler(context, bases, draw_seed, side, source_sample=original)
            for row, base in enumerate(bases):
                attempted += 1
                fraction = float(sample['valid_mask'][row].float().mean())
                used = bool(sample['eligible'][row]) and fraction >= minimum_valid
                draws.write(json.dumps({'update': step, 'stage': stage,
                    'appearance_version': appearance, 'attempt': attempted,
                    'base_index': base, 'draw_seed': draw_seed, 'row': row,
                    'valid_fraction': fraction, 'used': used,
                    'provenance': sample['provenance'][row]}, allow_nan=False) + '\n')
                if used:
                    accepted += 1
                    images.append(sample['inputs'][row].clone())
                    states.append(sample['state'][row].clone())
                    reflections.append(sample['reflection'][row].clone())
                    masks.append(sample['valid_mask'][row].clone())
                    fractions.append(fraction)
            del original, sample
        optimizer.zero_grad(set_to_none=True)
        loss = direct_loss(torch.stack(images), torch.stack(states),
                           torch.stack(reflections), torch.stack(masks))
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.pose.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        record = {'update': step, 'stage': stage, 'appearance_version': appearance,
            'attempted': attempted, 'accepted': accepted, 'batch_size': batch_size,
            'mean_valid_fraction': float(np.mean(fractions)),
            'direct_loss': float(loss.detach()), 'gradient_norm': float(gradient),
            'elapsed_seconds': time.perf_counter() - started}
        training.write(json.dumps(record, allow_nan=False) + '\n')
        if step == 1 or step % 200 == 0:
            draws.flush()
            training.flush()
            print(json.dumps({'event': 'train_milestone', **record}), flush=True)
        if step in checkpoints:
            save(step)
            metrics, rows = inner_score()
            scores[step] = metrics['selection_score_mm']
            selections.write(json.dumps({'step': step, 'metrics': metrics,
                'rows': rows}, allow_nan=False) + '\n')
            selections.flush()
            print(json.dumps({'event': 'inner_checkpoint', 'step': step,
                              'metrics': metrics}), flush=True)
        del loss, images, states, reflections, masks

best = min(scores, key=scores.get)
(run / 'completed.json').write_text(json.dumps({'updates': updates,
    'attempted': attempted, 'accepted_physical_sections': accepted,
    'selected_step': best, 'selected_inner_score_mm': scores[best],
    'parent_checkpoint_sha256': protocol['parent_checkpoint_sha256'],
    'protocol_sha256': sha(run / 'protocol.json'), 'config_sha256': sha(run / 'config.json'),
    'inner_draws_sha256': sha(run / 'inner_draws.jsonl'),
    'draws_sha256': sha(run / 'draws.jsonl'),
    'training_sha256': sha(run / 'training.jsonl'),
    'inner_scores_sha256': sha(run / 'inner_scores.jsonl'),
    'checkpoint_sha256': {str(step): sha(run / f'joint_pose_step_{step:05d}.pt')
        for step in checkpoints}, 'source_sha256': config['source_sha256'],
    'calibrated': False, 'real_images_used': False,
    'dev_or_public_benchmark_used': False, 'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'train_complete', 'selected_step': best,
                  'accepted_physical_sections': accepted}), flush=True)
