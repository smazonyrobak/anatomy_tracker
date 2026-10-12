"""Small predeclared label-free acquired-style versus v9 pose continuation."""
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
from training.arbitrary_plane_one_shot_slide_artifacts_v10 import sample_one_shot_slide_artifacts_v10
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


SIDE, BATCH, UPDATES = 192, 4, 2000
STEPS = (0, 1000, 2000)
SEEDS = {'control_v9': 202610122741, 'fourier_v10': 202610122742}
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/FOURIER_STYLE_274_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_270_v9_continuation_001'
TRANSFER = ROOT / 'runs/acquired_train_transfer_272_001'
RUN = ROOT / 'runs/pose_style_274_train_inner_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent_receipt = json.loads((PARENT / 'completed.json').read_text())
transfer_receipt = json.loads((TRANSFER / 'completed.json').read_text())
checkpoint_path = PARENT / 'pose_step_08000.pt'
assert parent_receipt['selected_step'] == 8000
with checkpoint_path.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == parent_receipt['checkpoint_sha256']['8000']
assert hashlib.sha256((TRANSFER / 'cases.jsonl').read_bytes()).hexdigest() == transfer_receipt['cases_sha256']
parent = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
fresh_panel = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
old_panel = torch.load(ROOT / 'runs/pose_capture_268_v8_train_inner_002/inner_panel.pt',
                       map_location='cpu', weights_only=False)
assert len(fresh_panel) == len(old_panel) == 64
context = load_streaming_synthetic_v7_64(device='cuda')

sources = {
    'coronal': (ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
                ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
                'animal_id', 'model_pixel_to_ap_dv_ml_um'),
    'sagittal': (ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
                 ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
                 'donor_id', 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
}
style_banks, eval_images, source_hashes = {}, {}, {}
for domain, (records_path, images_path, donor_key, affine_key) in sources.items():
    records = [json.loads(line) for line in records_path.open()]
    records = [row for row in records if row['training_split' if domain == 'coronal' else 'split'] == 'train']
    by_row = {row['array_row_index']: row for row in records}
    donors = sorted({row[donor_key] for row in records})
    even = [donor for donor in donors if hashlib.sha256(
        f'{domain}|{donor}'.encode()).digest()[0] % 2 == 0]
    odd = [donor for donor in donors if donor not in even]
    assert (len(even), len(odd)) == ((30, 28) if domain == 'coronal' else (18, 22))
    array = np.load(images_path, mmap_mode='r')
    selected = []
    for donor in even:
        cohort = sorted((row for row in records if row[donor_key] == donor),
                        key=lambda row: (row['section_number'], row['section_id']))
        selected += [cohort[index] for index in np.linspace(
            0, len(cohort) - 1, min(16, len(cohort)), dtype=int)]
    bank = torch.from_numpy(np.stack([np.asarray(array[row['array_row_index'], 0],
                                              dtype=np.float32) for row in selected]))[:, None].cuda()
    if domain == 'sagittal':
        bank = F.interpolate(bank, (SIDE, SIDE), mode='area')
    style_banks[domain] = {'images': bank[:, 0], 'records': [{
        'donor_id': row[donor_key], 'specimen_id': row['specimen_id'],
        'experiment_id': row['experiment_id'], 'section_id': row['section_id'],
        'source_image_sha256': row.get('source_image_sha256', row.get('image_sha256'))}
        for row in selected]}
    eval_images[domain] = (array, by_row)
    with records_path.open('rb') as stream:
        records_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    with images_path.open('rb') as stream:
        images_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    source_hashes[domain] = {'records_sha256': records_hash,
                             'images_sha256': images_hash,
                             'style_donors': len(even), 'eval_donors': len(odd),
                             'style_images': len(selected)}

eval_cases = []
for case in (json.loads(line) for line in (TRANSFER / 'cases.jsonl').open()):
    domain, donor = case['domain'], case['donor_id']
    if hashlib.sha256(f'{domain}|{donor}'.encode()).digest()[0] % 2 != 1:
        continue
    array, by_row = eval_images[domain]
    record = by_row[case['array_row_index']]
    assert record['section_id'] == case['section_id']
    image = torch.from_numpy(np.asarray(array[case['array_row_index'], 0],
                                        dtype=np.float32).copy())[None, None].cuda()
    if domain == 'sagittal':
        image = F.interpolate(image, (SIDE, SIDE), mode='area')
    matrix = np.asarray(record[sources[domain][3]], dtype=np.float64)
    weak_normal = np.cross(matrix[:, 0], matrix[:, 1])
    weak_normal /= np.linalg.norm(weak_normal)
    eval_cases.append({'case': case, 'image': image, 'weak_normal': weak_normal,
                       'weak_axis': int(np.abs(weak_normal).argmax())})
assert sum(item['case']['domain'] == 'coronal' for item in eval_cases) == 28
assert sum(item['case']['domain'] == 'sagittal' for item in eval_cases) == 22

config = {'candidate': 274, 'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
          'source_sha256': {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                            for name in ('train_pose_style_274.py',
                                         'arbitrary_plane_one_shot_slide_artifacts_v9.py',
                                         'arbitrary_plane_one_shot_slide_artifacts_v10.py')},
          'parent_checkpoint_sha256': parent_receipt['checkpoint_sha256']['8000'],
          'parent_fresh_panel_sha256': parent_receipt['inner_panel_sha256'],
          'parent_old_panel_sha256': parent['config']['parent_inner_panel_sha256'],
          'transfer_cases_sha256': transfer_receipt['cases_sha256'],
          'source_data': source_hashes, 'synthetic_provenance': context['provenance'],
          'arms': SEEDS, 'optimizer_batches_per_arm': UPDATES, 'batch_size': BATCH,
          'learning_rate': 5e-5, 'gradient_base_indices': list(range(56)),
          'acquired_affine_role': 'odd-hash TRAIN donor diagnostic readout only',
          'no_prior_model_weights_features_pseudolabels': True,
          'no_dev_cal_final_or_public_benchmark': True,
          'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2))

corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                        [95.5, 95.5]], device='cuda') / SIDE
reflection_options = torch.arange(2, device='cuda')[None, None]
readouts, counters = {}, {}
started = time.perf_counter()
with (RUN / 'draws.jsonl').open('w') as draws, (RUN / 'training.jsonl').open('w') as training, \
        (RUN / 'readouts.jsonl').open('w') as output:
    for arm, seed in SEEDS.items():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        subject_rng = np.random.default_rng(seed)
        model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda()
        model.load_state_dict(parent['model'])
        model.requires_grad_(False)
        for module in (model.encoder, model.lateral, model.pose,
                       model.anchor_pose, model.anchor_global):
            module.requires_grad_(True)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=1e-4)
        optimizer.load_state_dict(parent['optimizer'])
        optimizer.param_groups[0]['lr'] = 5e-5
        attempted = accepted = 0
        readouts[arm] = {}
        for step in range(UPDATES + 1):
            if step:
                images, states, reflections, masks = [], [], [], []
                while len(images) < BATCH:
                    attempted += 1
                    base = int(subject_rng.integers(56))
                    variant = int(subject_rng.integers(VARIANTS))
                    subject_index = base * VARIANTS + variant
                    draw_seed = seed + 2_000_000_000 + attempted
                    with torch.no_grad():
                        sample = (sample_one_shot_slide_artifacts_v9(context, [subject_index],
                                  draw_seed, SIDE) if arm == 'control_v9' else
                                  sample_one_shot_slide_artifacts_v10(context, [subject_index],
                                  draw_seed, style_banks, SIDE))
                    used = bool(sample['eligible'][0])
                    draws.write(json.dumps({'arm': arm, 'step': step, 'attempt': attempted,
                        'base_index': base, 'subject_index': subject_index, 'draw_seed': draw_seed,
                        'used': used, 'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
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
                training.write(json.dumps({'arm': arm, 'step': step, 'attempted': attempted,
                    'accepted': accepted, 'loss': float(loss.detach()),
                    'gradient_norm': float(gradient),
                    'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
                if step == 1 or step % 500 == 0:
                    draws.flush()
                    training.flush()
                    print(json.dumps({'event': 'train_milestone', 'arm': arm, 'step': step,
                                      'loss': float(loss.detach()), 'accepted': accepted}), flush=True)
                del inputs, states, reflections, valid, prediction, loss

            if step in STEPS:
                draws.flush()
                training.flush()
                model.eval()
                panel_scores = {}
                with torch.inference_mode():
                    for panel_name, panel in (('fresh_v9', fresh_panel), ('old_v8', old_panel)):
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
                            rows.append({'physical_section_id': item['physical_section_id'],
                                'base_index': item['base_index'],
                                'top16_hit': bool((errors <= 1.5).any()),
                                'top1_mm': float(errors[0])})
                        panel_scores[panel_name] = {'n': len(rows),
                            'top16_hits': sum(row['top16_hit'] for row in rows),
                            'mean_top1_mm': float(np.mean([row['top1_mm'] for row in rows]))}
                        output.write(json.dumps({'arm': arm, 'step': step, 'panel': panel_name,
                            'metrics': panel_scores[panel_name], 'rows': rows}, allow_nan=False) + '\n')
                    for domain in ('coronal', 'sagittal'):
                        rows = []
                        for item in eval_cases:
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
                            normal = full_frame_state_to_components(
                                prediction['state'][0, first])[1][:, 2]
                            angle = float(torch.rad2deg(torch.acos((normal * torch.as_tensor(
                                item['weak_normal'], device='cuda')).sum().abs().clamp(0, 1))))
                            rows.append({'donor_id': case['donor_id'],
                                'specimen_id': case['specimen_id'],
                                'experiment_id': case['experiment_id'],
                                'section_id': case['section_id'],
                                'source_image_sha256': case['source_image_sha256'],
                                'first_axis_correct': int(normal.abs().argmax()) == item['weak_axis'],
                                'normal_gap_to_weak_affine_deg': angle})
                        panel_scores[domain] = {'n': len(rows),
                            'first_axis_correct': sum(row['first_axis_correct'] for row in rows),
                            'median_weak_normal_angle_deg': float(np.median([
                                row['normal_gap_to_weak_affine_deg'] for row in rows]))}
                        output.write(json.dumps({'arm': arm, 'step': step, 'panel': domain,
                            'metrics': panel_scores[domain], 'rows': rows}, allow_nan=False) + '\n')
                readouts[arm][step] = panel_scores
                model.train()
                output.flush()
                if step:
                    checkpoint = RUN / f'{arm}_step_{step:05d}.pt'
                    temporary = checkpoint.with_suffix('.tmp')
                    torch.save({'arm': arm, 'step': step, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'config': config,
                        'subject_rng': subject_rng.bit_generator.state,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'attempted': attempted, 'accepted': accepted,
                        'calibrated': False}, temporary)
                    os.replace(temporary, checkpoint)
                print(json.dumps({'event': 'readout', 'arm': arm, 'step': step,
                                  'metrics': panel_scores}), flush=True)
        counters[arm] = {'attempted': attempted, 'accepted': accepted}
        del model, optimizer
        torch.cuda.empty_cache()

control, style = readouts['control_v9'][UPDATES], readouts['fourier_v10'][UPDATES]
gate = {'sagittal_gain_at_least_8':
            style['sagittal']['first_axis_correct'] - control['sagittal']['first_axis_correct'] >= 8,
        'coronal_loss_at_most_2':
            control['coronal']['first_axis_correct'] - style['coronal']['first_axis_correct'] <= 2,
        'fresh_top16_loss_at_most_4':
            control['fresh_v9']['top16_hits'] - style['fresh_v9']['top16_hits'] <= 4,
        'old_top16_at_least_40': style['old_v8']['top16_hits'] >= 40}
receipt = {'candidate': 274, 'steps': STEPS, 'counters': counters,
    'readouts': readouts, 'gate': gate, 'gate_pass': all(gate.values()),
    'result_role': 'small TRAIN-inner matched-budget pilot; no checkpoint promotion',
    'config_sha256': hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest(),
    'draws_sha256': hashlib.sha256((RUN / 'draws.jsonl').read_bytes()).hexdigest(),
    'training_sha256': hashlib.sha256((RUN / 'training.jsonl').read_bytes()).hexdigest(),
    'readouts_sha256': hashlib.sha256((RUN / 'readouts.jsonl').read_bytes()).hexdigest(),
    'checkpoint_sha256': {f'{arm}_{step}': hashlib.sha256(
        (RUN / f'{arm}_step_{step:05d}.pt').read_bytes()).hexdigest()
        for arm in SEEDS for step in STEPS[1:]},
    'weak_affines_used_for_training': False, 'calibrated': False,
    'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({'event': 'complete', 'gate': gate, 'gate_pass': all(gate.values()),
                  'readouts_2000': {arm: readouts[arm][UPDATES] for arm in SEEDS}}), flush=True)
