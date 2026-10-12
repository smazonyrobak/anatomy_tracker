"""Bounded v9 pose continuation with a new full-sphere TRAIN-inner panel."""
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
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


SEED, SIDE, BATCH, UPDATES = 20261012275, 192, 4, 8000
STEPS = (0, 2000, 4000, 8000)
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/POSE_CAPTURE_275_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_style_274_train_inner_001'
TRANSFER = ROOT / 'runs/acquired_train_transfer_272_001'
RUN = ROOT / 'runs/pose_capture_275_v9_continuation_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent_receipt = json.loads((PARENT / 'completed.json').read_text())
transfer_receipt = json.loads((TRANSFER / 'completed.json').read_text())
checkpoint_path = PARENT / 'control_v9_step_02000.pt'
with checkpoint_path.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == parent_receipt['checkpoint_sha256']['control_v9_2000']
assert parent_receipt['candidate'] == 274 and transfer_receipt['candidate'] == 272
parent = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
assert parent['step'] == 2000 and parent['arm'] == 'control_v9'
panel_270_path = ROOT / 'runs/pose_capture_270_v9_continuation_001/inner_panel.pt'
panel_268_path = ROOT / 'runs/pose_capture_268_v8_train_inner_002/inner_panel.pt'
panel_270 = torch.load(panel_270_path, map_location='cpu', weights_only=False)
panel_268 = torch.load(panel_268_path, map_location='cpu', weights_only=False)
assert len(panel_270) == len(panel_268) == 64
context = load_streaming_synthetic_v7_64(device='cuda')

sources = {
    'coronal': (ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
                ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
                'model_pixel_to_ap_dv_ml_um'),
    'sagittal': (ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
                 ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
                 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
}
acquired = []
transfer_cases = [json.loads(line) for line in (TRANSFER / 'cases.jsonl').open()]
assert hashlib.sha256((TRANSFER / 'cases.jsonl').read_bytes()).hexdigest() == transfer_receipt['cases_sha256']
for domain, (records_path, images_path, affine_key) in sources.items():
    by_row = {record['array_row_index']: record
              for record in (json.loads(line) for line in records_path.open())}
    array = np.load(images_path, mmap_mode='r')
    for case in transfer_cases:
        if case['domain'] != domain or hashlib.sha256(
                f'{domain}|{case["donor_id"]}'.encode()).digest()[0] % 2 != 1:
            continue
        record = by_row[case['array_row_index']]
        assert record['section_id'] == case['section_id']
        image = torch.from_numpy(np.asarray(array[case['array_row_index'], 0],
                                            dtype=np.float32).copy())[None, None].cuda()
        if domain == 'sagittal':
            image = F.interpolate(image, (SIDE, SIDE), mode='area')
        matrix = np.asarray(record[affine_key], dtype=np.float64)
        normal = np.cross(matrix[:, 0], matrix[:, 1])
        normal /= np.linalg.norm(normal)
        acquired.append({'case': case, 'image': image, 'normal': normal,
                         'axis': int(np.abs(normal).argmax())})
assert sum(item['case']['domain'] == 'coronal' for item in acquired) == 28
assert sum(item['case']['domain'] == 'sagittal' for item in acquired) == 22

config = {'candidate': 275, 'parent': 274, 'seed': SEED, 'side': SIDE,
    'batch_size': BATCH, 'added_optimizer_batches': UPDATES, 'readouts': STEPS,
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                      for name in ('train_pose_capture_275.py',
                                   'arbitrary_plane_one_shot_slide_artifacts_v9.py')},
    'parent_checkpoint_sha256': parent_receipt['checkpoint_sha256']['control_v9_2000'],
    'panel_270_sha256': hashlib.sha256(panel_270_path.read_bytes()).hexdigest(),
    'panel_268_sha256': hashlib.sha256(panel_268_path.read_bytes()).hexdigest(),
    'transfer_cases_sha256': transfer_receipt['cases_sha256'],
    'synthetic_provenance': context['provenance'],
    'gradient_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'loss': 'unchanged 270/274 physical-site+five-point+normal mixture',
    'optimizer': 'inherited AdamW moments, LR 5e-5 to 4000 then 3e-5, clip 2',
    'checkpoint_selection': 'highest new-panel top16 then lowest new-panel top1 mean',
    'weak_acquired_affines_used_for_gradients': False,
    'no_prior_external_weights_features_pseudolabels': True,
    'no_dev_cal_final_or_public_benchmark': True,
    'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2))

inner_rng = np.random.default_rng(SEED + 1)
fresh_panel, panel_attempts = [], 0
with (RUN / 'inner_draws.jsonl').open('w') as output:
    for base in range(56, 64):
        for slot in range(8):
            while True:
                panel_attempts += 1
                variant = int(inner_rng.integers(VARIANTS))
                subject_index = base * VARIANTS + variant
                draw_seed = SEED + 1_000_000_000 + panel_attempts
                with torch.no_grad():
                    sample = sample_one_shot_slide_artifacts_v9(context, [subject_index],
                                                                 draw_seed, SIDE)
                used = bool(sample['eligible'][0])
                output.write(json.dumps({'base_index': base, 'slot': slot,
                    'subject_index': subject_index, 'draw_seed': draw_seed, 'used': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    break
                del sample
            valid = sample['valid_mask'][0]
            pixels = valid.flatten().nonzero()[:, 0]
            selected = pixels[torch.linspace(0, len(pixels) - 1,
                min(1024, len(pixels)), device='cuda').long()]
            chart = torch.stack((selected.remainder(SIDE),
                selected.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            true_normal = full_frame_state_to_components(sample['state'][0])[1][:, 2]
            angle = float(torch.rad2deg(torch.acos(true_normal.abs().max().clamp(0, 1))))
            fresh_panel.append({'inputs': sample['inputs'][0].cpu(),
                'state': sample['state'][0].cpu(), 'reflection': sample['reflection'][0].cpu(),
                'chart': chart.cpu(), 'base_index': base, 'slot': slot,
                'physical_section_id': sample['provenance'][0]['physical_section_id'],
                'valid_fraction': float(valid.float().mean()),
                'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                'obliquity_deg': angle, 'provenance': sample['provenance'][0]})
            del sample
assert len(fresh_panel) == 64
torch.save(fresh_panel, RUN / 'inner_panel.pt')

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
subject_rng = np.random.default_rng(SEED + 2)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda()
model.load_state_dict(parent['model'])
model.requires_grad_(False)
for module in (model.encoder, model.lateral, model.pose, model.anchor_pose, model.anchor_global):
    module.requires_grad_(True)
parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=1e-4)
optimizer.load_state_dict(parent['optimizer'])
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
reflection_options = torch.arange(2, device='cuda')[None, None]
readouts, attempted, accepted = {}, 0, 0
started = time.perf_counter()
with (RUN / 'draws.jsonl').open('w') as draws, (RUN / 'training.jsonl').open('w') as training, \
        (RUN / 'readouts.jsonl').open('w') as output:
    for step in range(UPDATES + 1):
        if step:
            optimizer.param_groups[0]['lr'] = 5e-5 if step <= 4000 else 3e-5
            images, states, reflections, masks = [], [], [], []
            while len(images) < BATCH:
                attempted += 1
                base = int(subject_rng.integers(56))
                variant = int(subject_rng.integers(VARIANTS))
                subject_index = base * VARIANTS + variant
                draw_seed = SEED + 2_000_000_000 + attempted
                with torch.no_grad():
                    sample = sample_one_shot_slide_artifacts_v9(context, [subject_index], draw_seed, SIDE)
                used = bool(sample['eligible'][0])
                draws.write(json.dumps({'step': step, 'attempt': attempted, 'base_index': base,
                    'subject_index': subject_index, 'draw_seed': draw_seed, 'used': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    accepted += 1
                    images.append(sample['inputs'][0].clone())
                    states.append(sample['state'][0].clone())
                    reflections.append(sample['reflection'][0].clone())
                    masks.append(sample['valid_mask'][0].clone())
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
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
            optimizer.step()
            training.write(json.dumps({'step': step, 'attempted': attempted, 'accepted': accepted,
                'loss': float(loss.detach()), 'gradient_norm': float(gradient),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
            if step == 1 or step % 500 == 0:
                draws.flush()
                training.flush()
                print(json.dumps({'event': 'train_milestone', 'step': step,
                                  'loss': float(loss.detach()), 'accepted': accepted}), flush=True)
            del inputs, states, reflections, valid, prediction, loss

        if step in STEPS:
            draws.flush()
            training.flush()
            model.eval()
            readouts[step] = {}
            with torch.inference_mode():
                for panel_name, panel in (('new_v9', fresh_panel), ('frozen_270_v9', panel_270),
                                          ('frozen_268_v8', panel_268)):
                    rows = []
                    for item in panel:
                        prediction = model.predict(item['inputs'][None].cuda())
                        prior = (prediction['log_mass'][..., None] + torch.stack((
                            F.logsigmoid(-prediction['reflection_logit']),
                            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                        order = prior.argsort(descending=True)
                        chart = item['chart'].cuda()
                        truth = rigid_points_090(item['state'][None].cuda(),
                            item['reflection'][None].cuda(), chart, (SIDE, SIDE))
                        candidate = prediction['state'][0, order[:16] // 2][None]
                        points = rigid_points_090(candidate, (order[:16] % 2)[None],
                            chart[None], (SIDE, SIDE))
                        errors = ((points - truth[:, None]).norm(dim=-1).mean(-1) / 1000)[0]
                        true_normal = full_frame_state_to_components(item['state'].cuda())[1][:, 2]
                        selected_normal = full_frame_state_to_components(
                            prediction['state'][0, order[0] // 2])[1][:, 2]
                        angle = float(torch.rad2deg(torch.acos((true_normal * selected_normal)
                                                     .sum().abs().clamp(0, 1))))
                        rows.append({'physical_section_id': item['physical_section_id'],
                            'base_index': item['base_index'], 'top16_hit': bool((errors <= 1.5).any()),
                            'top1_mm': float(errors[0]), 'normal_axis_error_deg': angle,
                            'visible_fraction': item['visible_fraction']})
                    metrics = {'n': len(rows), 'top16_hits': sum(row['top16_hit'] for row in rows),
                        'mean_top1_mm': float(np.mean([row['top1_mm'] for row in rows])),
                        'mean_normal_axis_error_deg': float(np.mean([
                            row['normal_axis_error_deg'] for row in rows]))}
                    readouts[step][panel_name] = metrics
                    output.write(json.dumps({'step': step, 'panel': panel_name,
                        'metrics': metrics, 'rows': rows}, allow_nan=False) + '\n')
                for domain in ('coronal', 'sagittal'):
                    rows = []
                    for item in acquired:
                        case = item['case']
                        if case['domain'] != domain:
                            continue
                        network_input = torch.cat((item['image'],
                            torch.zeros(1, 4, SIDE, SIDE, device='cuda')), 1)
                        prediction = model.predict(network_input)
                        prior = (prediction['log_mass'][..., None] + torch.stack((
                            F.logsigmoid(-prediction['reflection_logit']),
                            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                        first = int(prior.argmax()) // 2
                        normal = full_frame_state_to_components(prediction['state'][0, first])[1][:, 2]
                        angle = float(torch.rad2deg(torch.acos((normal * torch.as_tensor(
                            item['normal'], device='cuda')).sum().abs().clamp(0, 1))))
                        rows.append({'donor_id': case['donor_id'],
                            'specimen_id': case['specimen_id'], 'experiment_id': case['experiment_id'],
                            'section_id': case['section_id'], 'source_image_sha256': case['source_image_sha256'],
                            'first_axis_correct': int(normal.abs().argmax()) == item['axis'],
                            'normal_gap_to_weak_affine_deg': angle})
                    metrics = {'n': len(rows),
                        'first_axis_correct': sum(row['first_axis_correct'] for row in rows),
                        'median_weak_normal_angle_deg': float(np.median([
                            row['normal_gap_to_weak_affine_deg'] for row in rows]))}
                    readouts[step][domain] = metrics
                    output.write(json.dumps({'step': step, 'panel': domain,
                        'metrics': metrics, 'rows': rows}, allow_nan=False) + '\n')
            model.train()
            output.flush()
            checkpoint = RUN / f'pose_step_{step:05d}.pt'
            temporary = checkpoint.with_suffix('.tmp')
            torch.save({'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'subject_rng': subject_rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'attempted': attempted, 'accepted': accepted, 'config': config,
                'calibrated': False}, temporary)
            os.replace(temporary, checkpoint)
            print(json.dumps({'event': 'readout', 'step': step, 'metrics': readouts[step]}), flush=True)

selected_step = max(STEPS, key=lambda step: (readouts[step]['new_v9']['top16_hits'],
    -readouts[step]['new_v9']['mean_top1_mm']))
selected = readouts[selected_step]
gate = {'new_top16_at_least_52': selected['new_v9']['top16_hits'] >= 52,
    'new_mean_top1_at_most_2.3mm': selected['new_v9']['mean_top1_mm'] <= 2.3,
    'frozen_270_top16_at_least_48': selected['frozen_270_v9']['top16_hits'] >= 48,
    'frozen_268_top16_at_least_50': selected['frozen_268_v8']['top16_hits'] >= 50,
    'acquired_coronal_at_least_26': selected['coronal']['first_axis_correct'] >= 26,
    'acquired_sagittal_at_least_16': selected['sagittal']['first_axis_correct'] >= 16}
receipt = {'candidate': 275, 'selected_step': selected_step,
    'attempted_draws': attempted, 'accepted_presentations': accepted,
    'panel_attempts': panel_attempts, 'readouts': readouts,
    'gate': gate, 'gate_pass': all(gate.values()),
    'config_sha256': hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest(),
    'inner_panel_sha256': hashlib.sha256((RUN / 'inner_panel.pt').read_bytes()).hexdigest(),
    'inner_draws_sha256': hashlib.sha256((RUN / 'inner_draws.jsonl').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((RUN / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((RUN / 'training.jsonl').read_bytes()).hexdigest(),
    'readouts_sha256': hashlib.sha256((RUN / 'readouts.jsonl').read_bytes()).hexdigest(),
    'checkpoint_sha256': {str(step): hashlib.sha256(
        (RUN / f'pose_step_{step:05d}.pt').read_bytes()).hexdigest() for step in STEPS},
    'weak_affines_used_for_training_or_selection': False,
    'calibrated': False, 'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({'event': 'complete', 'selected_step': selected_step,
                  'gate': gate, 'gate_pass': all(gate.values())}), flush=True)
