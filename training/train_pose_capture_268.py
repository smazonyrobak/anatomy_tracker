"""Scratch v8 synthetic TRAIN pose capture; no inherited weights or real labels."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v8 import sample_one_shot_slide_artifacts_v8
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


SEED, SIDE, BATCH, UPDATES = 20261012268, 192, 4, 8000
STEPS = (0, 2000, 4000, 8000)
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/POSE_CAPTURE_268_PROTOCOL_20261012.md'
RUN = ROOT / 'runs/pose_capture_268_v8_train_inner_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
context = load_streaming_synthetic_v7_64(device='cuda')
assert len(context['bases']) == 64
assert all(base['lineage']['split'] == 'train' for base in context['bases'])
source_names = ('train_pose_capture_268.py', 'arbitrary_plane_one_shot_model.py',
    'arbitrary_plane_one_shot_slide_artifacts_v8.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_one_shot_slide_artifacts_v4.py',
    'arbitrary_plane_one_shot_slide_artifacts_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_streaming_synthetic_v7.py',
    'arbitrary_plane_subject_torch_v6.py',
    'arbitrary_plane_subject_sampling_v6.py',
    'arbitrary_plane_allen_atlas_binding_v6.py',
    'subject_deformed_slab_multiresolution_bundle_v2.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_joint_model_v7.py', 'arbitrary_plane_recurrent_model.py',
    'arbitrary_plane_ribbon_v6.py', 'global_atlas_contrast_090.py')
protocol = {'candidate': 268, 'seed': SEED, 'side': SIDE, 'batch': BATCH,
    'max_optimizer_batches': UPDATES, 'readouts': STEPS,
    'gradient_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'inner_eligible_per_base': 8, 'inner_site_limit': 1024,
    'model': 'fresh OneShotJointSliceModel(modes=16, normal_anchor_count=64)',
    'trained': 'encoder,lateral,pose,anchor_pose,anchor_global only',
    'input': 'one independent v8 plane/warp/artifact/exterior per example; no paired appearance',
    'chart': '(x/192,y/192) at integer pixel centres; reflected x=(191-x)/192',
    'loss': '154 physical-site+fivepoint+normal mixture, adapted to source_shape=(192,192)',
    'stages': [[1, 2000, .12, 1e-4], [2001, 4000, .08, 7e-5],
               [4001, 8000, 0., 5e-5]],
    'optimizer': 'AdamW weight_decay=1e-4, gradient_clip=2',
    'selection': 'trained checkpoint: max blind top16 hits then min equal-base mean top1 rigid site mm',
    'gate_at_8000': 'top16<=1.5mm >=45/64; >step0 hits in >=6/8 bases; median minor FOV ratio>=.5; <=4 selected minor FOV <=1.2mm',
    'calibrated': False, 'prior_model_weights_features_pseudolabels_used': False,
    'acquired_images_used': False, 'dev_cal_final_or_public_benchmark_used': False}
config = {'protocol': protocol, 'protocol_document_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE / name) for name in source_names},
    'synthetic_provenance': context['provenance'],
    'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'protocol.json').write_text(json.dumps(protocol, indent=2))
config['protocol_sha256'] = sha(RUN / 'protocol.json')
(RUN / 'config.json').write_text(json.dumps(config, indent=2))

# Freeze the TRAIN-inner panel before initializing weights or taking an optimizer step.
inner_rng = np.random.default_rng(SEED + 1)
inner, panel_attempt = [], 0
with (RUN / 'inner_draws.jsonl').open('w') as log:
    for base in range(56, 64):
        for slot in range(8):
            while True:
                panel_attempt += 1
                variant = int(inner_rng.integers(VARIANTS))
                subject_index = base * VARIANTS + variant
                draw_seed = SEED + 1_000_000_000 + panel_attempt
                with torch.no_grad():
                    sample = sample_one_shot_slide_artifacts_v8(context, [subject_index], draw_seed, SIDE)
                used = bool(sample['eligible'][0])
                record = sample['provenance'][0]
                log.write(json.dumps({'base_index': base, 'slot': slot, 'variant': variant,
                    'subject_index': subject_index, 'draw_seed': draw_seed, 'used': used,
                    'provenance': record}, allow_nan=False) + '\n')
                if used:
                    break
                del sample
            valid = sample['valid_mask'][0]
            pixels = valid.flatten().nonzero()[:, 0]
            selected = pixels[torch.linspace(0, len(pixels) - 1,
                min(1024, len(pixels)), device='cuda').long()]
            chart = torch.stack((selected.remainder(SIDE),
                selected.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            normal = record['physical_unit_normal_ap_dv_ml']
            angle = float(np.degrees(np.arccos(np.clip(max(abs(x) for x in normal), -1, 1))))
            inner.append({'inputs': sample['inputs'][0].cpu(), 'state': sample['state'][0].cpu(),
                'reflection': sample['reflection'][0].cpu(), 'chart': chart.cpu(),
                'base_index': base, 'slot': slot, 'physical_section_id': record['physical_section_id'],
                'valid_fraction': float(valid.float().mean()), 'obliquity_deg': angle,
                'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                'nearest_axis': ('AP', 'DV', 'ML')[int(np.abs(normal).argmax())],
                'exterior_style': record['one_shot_slide_artifacts_v8']['exterior_style'],
                'source_mode': record['mode'], 'provenance': record})
            del sample
assert len(inner) == 64 and len({item['physical_section_id'] for item in inner}) == 64
torch.save(inner, RUN / 'inner_panel.pt')

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
subject_rng = np.random.default_rng(SEED + 2)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda()
model.requires_grad_(False)
for module in (model.encoder, model.lateral, model.pose, model.anchor_pose,
               model.anchor_global):
    module.requires_grad_(True)
pose_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(pose_parameters, lr=1e-4, weight_decay=1e-4)
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
reflection_options = torch.arange(2, device='cuda')[None, None]
readouts, attempted, accepted = {}, 0, 0
started = time.perf_counter()
with (RUN / 'draws.jsonl').open('w') as draws, (RUN / 'training.jsonl').open('w') as training, \
        (RUN / 'inner_scores.jsonl').open('w') as selections:
    for step in range(UPDATES + 1):
        if step:
            if step <= 2000:
                minimum_valid, lr, stage = .12, 1e-4, 1
            elif step <= 4000:
                minimum_valid, lr, stage = .08, 7e-5, 2
            else:
                minimum_valid, lr, stage = 0., 5e-5, 3
            optimizer.param_groups[0]['lr'] = lr
            images, states, reflections, masks, fractions = [], [], [], [], []
            while len(images) < BATCH:
                attempted += 1
                base = int(subject_rng.integers(56))
                variant = int(subject_rng.integers(VARIANTS))
                subject_index = base * VARIANTS + variant
                draw_seed = SEED + 2_000_000_000 + attempted
                with torch.no_grad():
                    sample = sample_one_shot_slide_artifacts_v8(context, [subject_index], draw_seed, SIDE)
                fraction = float(sample['valid_mask'][0].float().mean())
                used = bool(sample['eligible'][0]) and fraction >= minimum_valid
                draws.write(json.dumps({'step': step, 'stage': stage, 'attempt': attempted,
                    'base_index': base, 'variant': variant, 'subject_index': subject_index,
                    'draw_seed': draw_seed, 'physical_section_id': sample['provenance'][0]['physical_section_id'],
                    'valid_fraction': fraction, 'used': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    accepted += 1
                    images.append(sample['inputs'][0].clone())
                    states.append(sample['state'][0].clone())
                    reflections.append(sample['reflection'][0].clone())
                    masks.append(sample['valid_mask'][0].clone())
                    fractions.append(fraction)
                del sample
            inputs, states, reflections, valid = map(torch.stack,
                (images, states, reflections, masks))
            pixels = torch.multinomial(valid.flatten(1).float(), 128, replacement=True)
            chart = torch.stack((pixels.remainder(SIDE),
                pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            optimizer.zero_grad(set_to_none=True)
            prediction = model.predict(inputs)
            proposed_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1).float()
            proposed_reflections = reflection_options.expand(BATCH, model.modes, 2)
            prior = (prediction['log_mass'].float()[..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'].float()),
                F.logsigmoid(prediction['reflection_logit'].float())), -1)).flatten(1)
            true = rigid_points_090(states, reflections, chart, (SIDE, SIDE))
            proposed = rigid_points_090(proposed_states, proposed_reflections,
                chart[:, None, None], (SIDE, SIDE))
            true_five = rigid_points_090(states, reflections, corners, (SIDE, SIDE))
            proposed_five = rigid_points_090(proposed_states, proposed_reflections,
                corners, (SIDE, SIDE))
            site_um = (proposed - true[:, None, None]).norm(dim=-1).mean(-1)
            five_um = (proposed_five - true_five[:, None, None]).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(prediction['state'].float())[1][..., :, 2]
            true_normal = full_frame_state_to_components(states)[1][..., :, 2]
            normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
            physical = (.75 * site_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-physical.detach() / .7, -1)
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(BATCH, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
                    + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
                    + .5 * physical[:, :2 * model.base_modes].min(-1).values
                    + (neighbourhood * near_cost).sum(-1)).mean()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(pose_parameters, 2., error_if_nonfinite=True)
            optimizer.step()
            record = {'step': step, 'stage': stage, 'attempted': attempted, 'accepted': accepted,
                'batch_size': BATCH, 'mean_valid_fraction': float(np.mean(fractions)),
                'direct_loss': float(loss.detach()), 'gradient_norm': float(gradient),
                'elapsed_seconds': time.perf_counter() - started}
            training.write(json.dumps(record, allow_nan=False) + '\n')
            if step == 1 or step % 200 == 0:
                draws.flush()
                training.flush()
                print(json.dumps({'event': 'train_milestone', **record}), flush=True)
            del inputs, states, reflections, valid, prediction, loss

        if step in STEPS:
            draws.flush()
            training.flush()
            model.eval()
            rows = []
            with torch.inference_mode():
                for item in inner:
                    inputs = item['inputs'][None].cuda()
                    chart = item['chart'].cuda()
                    state = item['state'][None].cuda()
                    reflection = item['reflection'][None].cuda()
                    prediction = model.predict(inputs)
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                    truth = rigid_points_090(state, reflection, chart, (SIDE, SIDE))
                    errors = []
                    for start in range(0, model.modes, 8):
                        candidate = prediction['state'][:, start:start + 8, None].expand(-1, -1, 2, -1)
                        points = rigid_points_090(candidate, reflection_options.expand(1, 8, 2),
                            chart[None, None], (SIDE, SIDE))
                        errors.append(((points - truth[:, None, None]).norm(dim=-1).mean(-1) / 1000).flatten())
                    error = torch.cat(errors)
                    order = prior.argsort(descending=True)
                    top = int(order[0])
                    selected = prediction['state'][0, top // 2]
                    true_normal = full_frame_state_to_components(state)[1][0, :, 2]
                    selected_normal = full_frame_state_to_components(selected)[1][:, 2]
                    selected_basis = full_frame_state_to_components(selected)[2]
                    true_basis = full_frame_state_to_components(state)[2][0]
                    minor = float(torch.linalg.svdvals(selected_basis).min() / 1000)
                    true_minor = float(torch.linalg.svdvals(true_basis).min() / 1000)
                    normal_angle = float(torch.rad2deg(torch.acos(
                        (selected_normal * true_normal).sum().abs().clamp(0, 1))))
                    rows.append({'base_index': item['base_index'], 'slot': item['slot'],
                        'physical_section_id': item['physical_section_id'],
                        'valid_fraction': item['valid_fraction'],
                        'visible_fraction': item['visible_fraction'],
                        'obliquity_deg': item['obliquity_deg'],
                        'nearest_axis': item['nearest_axis'], 'exterior_style': item['exterior_style'],
                        'source_mode': item['source_mode'],
                        'oracle_mm': float(error.min()), 'top1_mm': float(error[top]),
                        'top16_hit': bool((error[order[:16]] <= 1.5).any()),
                        'top2_hit': bool((error[order[:2]] <= 1.5).any()),
                        'oracle_hit': bool((error <= 1.5).any()),
                        'normal_axis_error_deg': normal_angle,
                        'selected_minor_fov_mm': minor, 'true_minor_fov_mm': true_minor,
                        'minor_fov_ratio': minor / true_minor})
            model.train()
            base = {str(index): {'top16_hits': sum(row['top16_hit'] for row in rows
                if row['base_index'] == index), 'mean_top1_mm': float(np.mean([row['top1_mm']
                for row in rows if row['base_index'] == index]))} for index in range(56, 64)}
            strata = {}
            for label, predicate in (
                ('visible_lt_0.15', lambda row: row['visible_fraction'] < .15),
                ('visible_0.15_to_0.40', lambda row: .15 <= row['visible_fraction'] < .40),
                ('visible_ge_0.40', lambda row: row['visible_fraction'] >= .40),
                ('axis_le_20deg', lambda row: row['obliquity_deg'] <= 20),
                ('axis_20_to_45deg', lambda row: 20 < row['obliquity_deg'] <= 45),
                ('axis_gt_45deg', lambda row: row['obliquity_deg'] > 45)):
                subset = [row for row in rows if predicate(row)]
                strata[label] = {'n': len(subset), 'top16_hits': sum(row['top16_hit'] for row in subset),
                    'mean_top1_mm': float(np.mean([row['top1_mm'] for row in subset]))
                    if subset else None}
            metrics = {'n': 64, 'top16_hits': sum(row['top16_hit'] for row in rows),
                'top2_hits': sum(row['top2_hit'] for row in rows),
                'oracle_hits': sum(row['oracle_hit'] for row in rows),
                'mean_top1_mm': float(np.mean([row['top1_mm'] for row in rows])),
                'equal_base_top1_mm': float(np.mean([base[str(index)]['mean_top1_mm']
                    for index in range(56, 64)])),
                'median_minor_fov_ratio': float(np.median([row['minor_fov_ratio'] for row in rows])),
                'selected_minor_fov_le_1.2mm': sum(row['selected_minor_fov_mm'] <= 1.2 for row in rows),
                'base': base, 'strata': strata}
            readouts[step] = metrics
            selections.write(json.dumps({'step': step, 'metrics': metrics, 'rows': rows},
                allow_nan=False) + '\n')
            selections.flush()
            checkpoint = RUN / f'pose_step_{step:05d}.pt'
            temporary = checkpoint.with_suffix('.tmp')
            torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'subject_rng': subject_rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'attempted': attempted, 'accepted': accepted, 'config': config,
                'calibrated': False}, temporary)
            os.replace(temporary, checkpoint)
            print(json.dumps({'event': 'inner_checkpoint', 'step': step,
                'top16_hits': metrics['top16_hits'], 'mean_top1_mm': metrics['mean_top1_mm']}), flush=True)

best = max((2000, 4000, 8000), key=lambda step: (readouts[step]['top16_hits'],
    -readouts[step]['equal_base_top1_mm']))
gate = {'top16_at_least_45': readouts[8000]['top16_hits'] >= 45,
    'improved_in_six_bases': sum(readouts[8000]['base'][str(index)]['top16_hits'] >
        readouts[0]['base'][str(index)]['top16_hits'] for index in range(56, 64)) >= 6,
    'median_minor_fov_ratio_at_least_0.5': readouts[8000]['median_minor_fov_ratio'] >= .5,
    'at_most_four_minor_fov_le_1.2mm': readouts[8000]['selected_minor_fov_le_1.2mm'] <= 4}
receipt = {'candidate': 268, 'optimizer_batches': UPDATES, 'accepted_presentations': accepted,
    'attempted_draws': attempted, 'panel_attempted_draws': panel_attempt,
    'selected_step': best, 'gate': gate, 'gate_pass': all(gate.values()),
    'next_action': 'separately predeclare continuation if gate passes and coverage improves; otherwise diagnose; never extend this recipe silently',
    'protocol_sha256': sha(RUN / 'protocol.json'), 'config_sha256': sha(RUN / 'config.json'),
    'inner_panel_sha256': sha(RUN / 'inner_panel.pt'),
    'inner_draws_sha256': sha(RUN / 'inner_draws.jsonl'),
    'draws_sha256': sha(RUN / 'draws.jsonl'), 'training_sha256': sha(RUN / 'training.jsonl'),
    'inner_scores_sha256': sha(RUN / 'inner_scores.jsonl'),
    'checkpoint_sha256': {str(step): sha(RUN / f'pose_step_{step:05d}.pt') for step in STEPS},
    'source_sha256': config['source_sha256'],
    'calibrated': False, 'acquired_images_used': False,
    'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({'event': 'train_complete', 'selected_step': best,
    'accepted_presentations': accepted, 'gate_pass': receipt['gate_pass']}), flush=True)
