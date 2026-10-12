"""Scratch pose, source reliability, and exact-pose mapper stage for joint model 305."""
import hashlib
import json
import math
import os
import shutil
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
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


SEED, SIDE, BATCH, UPDATES = 20261012305, 192, 4, 26000
READOUTS = (0, 2000, 8000, 16000, 26000)
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/JOINT_SPATIAL_FEEDBACK_305_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/joint_spatial_feedback_305_pose_001'
RUN = ROOT / 'runs/joint_spatial_feedback_305_pose_002'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
context = load_streaming_synthetic_v7_64('cuda')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


parent_hashes = {
    'config.json': '20bf2217aee22d07a8aa4e9a4b64df96cac815592be0df71b7ade1418de113be',
    'inner_panel.pt': '53ca593e97fdcd02c5c8db66172f45662fbed2c98017e16ac7dbebc89194611f',
    'panel_draws.jsonl': '43ff2df34b2f3804f48cba32171c9277dbc9b36720bf13a23871e89394f3dcb5',
    'pose_step_00000.pt': '01c09ed4e0d12c9a133df181fdcf1c0760ea9f0661c60b2049bbe7c65e59f547',
    'pose_step_02000.pt': '39e2468eeed4b0bdd8ebf41c3ed852a1afd30c548454e8456cf8686471b43c61',
    'training.jsonl': '00894c1fefe4bb4d040a584f6bba8da6ba7c0e158702bc00136e7a407ebb5b1b',
    'training_draws.jsonl': '5b3abc4b983a07b7e354f19f14284bf5d9dcf0a528330e78695fcfdedd6bd142',
    'readouts.jsonl': 'e5b73d34de5b910fc04f775369b97afe41413295ee67aba5028366eb3c94ff67'}
assert all(sha(PARENT / name) == digest for name, digest in parent_hashes.items())
parent_config = json.loads((PARENT / 'config.json').read_text(encoding='utf8'))
assert parent_config['source_sha256']['arbitrary_plane_one_shot_model.py'] == sha(
    SOURCE / 'arbitrary_plane_one_shot_model.py')
assert parent_config['protocol_sha256'] == sha(PROTOCOL)
assert parent_config['source_sha256']['arbitrary_plane_postartifact_gauge_v1.py'] == \
    'ea591afab155df059ba212b668871e7005d584268c2cadf3b01aa71a873aba32'
checkpoint = torch.load(PARENT / 'pose_step_02000.pt', map_location='cpu', weights_only=False)
assert checkpoint['step'] == 2000 and json.loads(json.dumps(checkpoint['config'])) == parent_config
parent_training = next(json.loads(line) for line in (PARENT / 'training.jsonl').open(
    encoding='utf8') if json.loads(line)['step'] == 2000)
parent_next = next(json.loads(line) for line in (PARENT / 'training.jsonl').open(
    encoding='utf8') if json.loads(line)['step'] == 2001)
assert parent_training['accepted'] == 8000
parent_readout_rows = [json.loads(line) for line in (PARENT / 'readouts.jsonl').open(
    encoding='utf8') if json.loads(line)['step'] <= 2000]
assert [row['step'] for row in parent_readout_rows] == [0, 2000]


config = {'candidate': 305, 'stage': 'scratch_pose', 'seed': SEED, 'side': SIDE,
    'batch_size': BATCH, 'optimizer_batches': UPDATES, 'readouts': READOUTS,
    'model': {'modes': 16, 'normal_anchor_count': 64,
              'atlas_conditioning': True, 'vector_refinement': True,
              'spatial_feedback': True},
    'gradient_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'init': 'randomly initialized in verified 305 pose run 001; resumed its step-2000 state',
    'resume_parent': str(PARENT), 'resume_step': 2000,
    'resume_parent_sha256': parent_hashes,
    'resume_rule': 'rank-deficient nonfragment sections excluded from pose supervision',
    'synthetic_provenance': context['provenance'],
    'protocol_sha256': sha(PROTOCOL),
    'source_sha256': {name: sha(SOURCE / name) for name in (
        'train_joint_spatial_pose_305.py', 'arbitrary_plane_one_shot_model.py',
        'arbitrary_plane_postartifact_gauge_v1.py',
        'arbitrary_plane_one_shot_slide_artifacts_v9.py',
        'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py')},
    'no_external_weights_features_or_pseudolabels': True,
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False,
    'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

for name in ('panel_draws.jsonl', 'inner_panel.pt', 'pose_step_00000.pt',
             'pose_step_02000.pt'):
    shutil.copyfile(PARENT / name, RUN / name)
panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(panel) == 64 and len({row['physical_section_id'] for row in panel}) == 64
for name in ('training_draws.jsonl', 'training.jsonl', 'readouts.jsonl'):
    with (PARENT / name).open(encoding='utf8') as source, \
            (RUN / name).open('w', encoding='utf8') as output:
        for line in source:
            if json.loads(line)['step'] > 2000:
                break
            output.write(line)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, vector_refinement=True, spatial_feedback=True).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
draw_rng = np.random.default_rng(SEED + 2)
model.load_state_dict(checkpoint['model'])
model.train()
optimizer.load_state_dict(checkpoint['optimizer'])
torch.set_rng_state(checkpoint['torch_rng'])
torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
draw_rng.bit_generator.state = checkpoint['draw_rng_state']
del checkpoint
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
reflection_options = torch.arange(2, device='cuda')[None, None]
attempted, accepted = parent_training['attempted'], parent_training['accepted']
started = time.perf_counter()
readouts = {str(row['step']): row['summary'] for row in parent_readout_rows}
print(json.dumps({'event': 'verified_resume', 'step': 2000,
    'attempted': attempted, 'accepted': accepted}), flush=True)
with (RUN / 'training_draws.jsonl').open('a', encoding='utf8') as draws, \
        (RUN / 'training.jsonl').open('a', encoding='utf8') as training, \
        (RUN / 'readouts.jsonl').open('a', encoding='utf8') as readout_file:
    for step in range(2001, UPDATES + 1):
        if step:
            optimizer.param_groups[0]['lr'] = 2e-5 + 8e-5 * (1 + math.cos(math.pi * step / UPDATES)) / 2
            images, states, reflections, masks, centres, offsets, weights = [], [], [], [], [], [], []
            while len(images) < BATCH:
                attempted += 1
                base = int(draw_rng.integers(56))
                variant = int(draw_rng.integers(VARIANTS))
                subject = base * VARIANTS + variant
                seed = SEED + 2_000_000_000 + attempted
                with torch.no_grad():
                    sample = sample_postartifact_gauge_v1(context, [subject], seed, SIDE)
                used = bool(sample['eligible'][0])
                draws.write(json.dumps({'step': step, 'attempt': attempted,
                    'base': base, 'variant': variant, 'seed': seed, 'accepted': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    accepted += 1
                    images.append(sample['inputs'][0].clone())
                    states.append(sample['state'][0].clone())
                    reflections.append(sample['reflection'][0].clone())
                    masks.append(sample['valid_mask'][0].clone())
                    centres.append(sample['centre'][0].clone())
                    offsets.append(sample['offsets'][0].clone())
                    weights.append(sample['weights'][0].clone())
                del sample
            inputs, states, reflections, valid, centres, axial, masses = map(torch.stack,
                (images, states, reflections, masks, centres, offsets, weights))
            pixels = torch.multinomial(valid.flatten(1).float(), 128, replacement=True)
            chart = torch.stack((pixels.remainder(SIDE),
                pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            truth = centres.flatten(1, 2).gather(1, pixels[..., None].expand(-1, -1, 3))
            optimizer.zero_grad(set_to_none=True)
            prediction = model.predict(inputs)
            proposed_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1).float()
            proposed_reflections = reflection_options.expand(BATCH, model.modes, 2)
            prior = (prediction['log_mass'].float()[..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit'].float()),
                F.logsigmoid(prediction['reflection_logit'].float())), -1)).flatten(1)
            proposed = rigid_points_090(proposed_states, proposed_reflections,
                chart[:, None, None], (SIDE, SIDE))
            five = rigid_points_090(proposed_states, proposed_reflections,
                corners, (SIDE, SIDE))
            true_five = rigid_points_090(states, reflections, corners, (SIDE, SIDE))
            site_um = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1)
            five_um = (five - true_five[:, None, None]).norm(dim=-1).mean(-1)
            normal = full_frame_state_to_components(prediction['state'].float())[1][..., :, 2]
            true_normal = full_frame_state_to_components(states)[1][..., :, 2]
            normal_um = 4000 * (1 - (normal * true_normal[:, None]).sum(-1).abs().clamp_max(1))
            physical = (.75 * site_um + .25 * five_um + normal_um[..., None]).flatten(1) / 1000
            target_mass = F.softmax(-physical.detach() / .7, -1)
            nearest = (true_normal @ model.normal_anchor_frames[:, :, 2].T).abs().topk(4, -1)
            near_mode = model.base_modes + nearest.indices
            near_cost = physical.reshape(BATCH, model.modes, 2).min(-1).values.gather(1, near_mode)
            neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
            direct_loss = (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
                + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
                + .5 * physical[:, :2 * model.base_modes].min(-1).values
                + (neighbourhood * near_cost).sum(-1)).mean()
            reliability_loss = model.source_reliability_loss(prediction, valid)
            loss = direct_loss + .1 * reliability_loss
            teacher_error = None
            if step % 4 == 0:
                teacher_prediction = {**prediction, 'state': states[:, None]}
                index = torch.zeros((BATCH, 1), device='cuda', dtype=torch.long)
                teacher = model.map(teacher_prediction, axial, index,
                    reflections[:, None], (96, 96), context['atlas'], masses,
                    feature_side=96, source_shape=(SIDE, SIDE), max_derivative=.35)
                surface = teacher['centre_surface_ccf_ap_dv_ml_um'][:, 0].permute(0, 3, 1, 2)
                grid = ((chart + .5 / SIDE) * 2 - 1).reshape(BATCH, 1, 128, 2)
                fitted = F.grid_sample(surface, grid, mode='bilinear',
                    padding_mode='border', align_corners=False)[:, :, 0].transpose(1, 2)
                teacher_error = (fitted - truth).norm(dim=-1).mean() / 1000
                local = teacher['local_displacement_um'] / 1000
                loss = loss + 5 * teacher_error + .05 * local.square().mean()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            if step == 2001:
                assert attempted == parent_next['attempted'] and accepted == parent_next['accepted']
                assert abs(float(direct_loss.detach()) - parent_next['direct_loss']) < .01
                assert abs(float(loss.detach()) - parent_next['loss']) < .01
                print(json.dumps({'event': 'resume_replay_verified', 'step': step,
                    'direct_loss': float(direct_loss.detach()), 'attempted': attempted,
                    'accepted': accepted}), flush=True)
            training.write(json.dumps({'step': step, 'attempted': attempted,
                'accepted': accepted, 'direct_loss': float(direct_loss.detach()),
                'reliability_loss': float(reliability_loss.detach()),
                'teacher_error_mm': None if teacher_error is None else float(teacher_error.detach()),
                'loss': float(loss.detach()), 'gradient_norm': float(gradient),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
            if step == 1 or step % 500 == 0:
                draws.flush(); training.flush()
                print(json.dumps({'event': 'pose_batch', 'step': step,
                    'direct_loss': float(direct_loss.detach()), 'accepted': accepted}), flush=True)
            del inputs, states, reflections, valid, centres, prediction, loss

        if step in READOUTS:
            draws.flush()
            training.flush()
            model.eval()
            rows = []
            with torch.inference_mode():
                for item in panel:
                    prediction = model.predict(item['inputs'][None].cuda())
                    prior = (prediction['log_mass'][..., None] + torch.stack((
                        F.logsigmoid(-prediction['reflection_logit']),
                        F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                    branches = prior.topk(16).indices
                    states = prediction['state'][0, branches // 2][None]
                    chart = item['chart'].cuda()
                    truth = item['truth'].cuda()
                    points = rigid_points_090(states, (branches % 2)[None],
                        chart[None], (SIDE, SIDE))[0]
                    error = (points - truth[None]).norm(dim=-1).mean(-1) / 1000
                    rows.append({'physical_section_id': item['physical_section_id'],
                        'base': item['base'], 'slot': item['slot'],
                        'obliquity_deg': item['obliquity_deg'],
                        'visible_fraction': item['visible_fraction'],
                        'damage_fraction': item['damage_fraction'],
                        'exterior_style': item['exterior_style'],
                        'branch_ids': branches.tolist(), 'rigid_site_error_mm': error.tolist(),
                        'source_reliability_loss': float(model.source_reliability_loss(
                            prediction, item['valid_mask'][None].cuda()))})
            capture = sum(min(row['rigid_site_error_mm']) <= 1.5 for row in rows)
            mean_oracle = float(np.mean([min(row['rigid_site_error_mm']) for row in rows]))
            mean_first = float(np.mean([row['rigid_site_error_mm'][0] for row in rows]))
            strata = {'obliquity_ge45': [row for row in rows if row['obliquity_deg'] >= 45],
                'visible_lt0p1': [row for row in rows if row['visible_fraction'] < .1],
                'damage_ge0p02': [row for row in rows if row['damage_fraction'] >= .02]}
            strata.update({style: [row for row in rows if row['exterior_style'] == style]
                for style in ('exact_black', 'near_black', 'gray', 'exact_black_brush')})
            strata_metrics = {name: {'n': len(group),
                'capture_1p5mm': sum(min(row['rigid_site_error_mm']) <= 1.5 for row in group),
                'oracle_top16_mean_error_mm': (float(np.mean([
                    min(row['rigid_site_error_mm']) for row in group])) if group else None),
                'first_choice_mean_error_mm': (float(np.mean([
                    row['rigid_site_error_mm'][0] for row in group])) if group else None)}
                for name, group in strata.items()}
            readouts[str(step)] = {'capture_1p5mm': capture,
                'oracle_top16_mean_error_mm': mean_oracle,
                'first_choice_mean_error_mm': mean_first, 'strata': strata_metrics}
            readout_file.write(json.dumps({'step': step, 'summary': readouts[str(step)],
                'rows': rows}, allow_nan=False) + '\n')
            readout_file.flush()
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'config': config, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
                'draw_rng_state': draw_rng.bit_generator.state,
                'calibrated': False}, RUN / f'pose_step_{step:05d}.pt')
            print(json.dumps({'event': 'pose_readout', 'step': step,
                **readouts[str(step)]}), flush=True)
            model.train()

selected = max(READOUTS, key=lambda step: (
    int(readouts[str(step)]['capture_1p5mm'] >= 45
        and readouts[str(step)]['oracle_top16_mean_error_mm'] <= 1.5),
    readouts[str(step)]['capture_1p5mm'],
    -readouts[str(step)]['oracle_top16_mean_error_mm'],
    -readouts[str(step)]['first_choice_mean_error_mm']))
summary = {'candidate': 305, 'stage': 'scratch_pose', 'selected_step': selected,
    'readouts': readouts, 'attempted': attempted, 'accepted': accepted,
    'joint_stage_capture_gate': (readouts[str(selected)]['capture_1p5mm'] >= 45
        and readouts[str(selected)]['oracle_top16_mean_error_mm'] <= 1.5),
    'calibrated': False, 'no_dev_cal_final_or_public_benchmark': True}
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
receipt = {'summary': summary, 'output_sha256': {path.name: sha(path) for path in RUN.iterdir()
    if path.name != 'completed.json'}, 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'summary': summary}), flush=True)
