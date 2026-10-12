"""Matched 500-batch TRAIN-inner synthetic/control versus weak-volume mixture pilot."""
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
from training.arbitrary_plane_reserved_real_oblique_v3 import sample_reserved_real_oblique_v3
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


SEED, SIDE, BATCH, UPDATES, WEAK_WEIGHT = 20261012297, 192, 4, 500, .10
READOUTS = (0, 250, 500)
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/POSE_REAL_VOLUME_MIX_297_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
TRANSFER = ROOT / 'runs/acquired_train_transfer_272_001'
AUDIT = ROOT / 'runs/reserved_train_volume_geometry_105_audit'
RESERVED = ROOT / 'data/joint_v7_reserved_train_images_192_001'
RUN = ROOT / 'runs/pose_real_volume_mix_297_train_inner_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parent_receipt = json.loads((PARENT / 'completed.json').read_text())
parent_path = PARENT / 'pose_step_08000.pt'
parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
assert parent_receipt['candidate'] == 275
assert parent_sha == parent_receipt['checkpoint_sha256']['8000']
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
assert parent['step'] == 8000 and parent['calibrated'] is False
transfer_receipt = json.loads((TRANSFER / 'completed.json').read_text())
transfer_cases_path = TRANSFER / 'cases.jsonl'
assert hashlib.sha256(transfer_cases_path.read_bytes()).hexdigest() == transfer_receipt['cases_sha256']
transfer_cases = [json.loads(line) for line in transfer_cases_path.open()]

audit_rows_path, audit_summary_path = AUDIT / 'rows.jsonl', AUDIT / 'summary.json'
audit_rows = [json.loads(line) for line in audit_rows_path.open()]
audit_summary = json.loads(audit_summary_path.read_text())
assert len(audit_rows) == audit_summary['sampled_donors'] == 64
regular = [row for row in audit_rows if row['train_only'] and row['section_number_gaps'] == 0
    and row['inplane_basis_max_variation_um'] < 1e-5
    and row['section_step_max_residual_um'] < 1e-4
    and row['image_shape'] == [row['sections'], 1, 192, 192]]
assert len(regular) == audit_summary['affinely_regular_complete_stacks'] == 63
excluded = {'16207', '14855', '15975'}
assert {row['donor'].removeprefix('donor_') for row in audit_rows if row not in regular} == {'16207'}
selected = [row for row in regular if row['donor'].removeprefix('donor_') not in excluded]
assert len(selected) == 61
virtual_readout_indices = sorted(np.random.default_rng(SEED + 4).choice(
    len(selected), 8, replace=False).tolist())
virtual_gradient_indices = [index for index in range(len(selected))
    if index not in virtual_readout_indices]
assert len(virtual_gradient_indices) == 53
reserved_manifest_path = RESERVED / 'completed.json'
reserved_manifest = json.loads(reserved_manifest_path.read_text())
assert reserved_manifest['training_role'] == 'train_only'
assert reserved_manifest['development_calibration_final_or_benchmark_downloads'] == 0
donors, bindings = [], {str(reserved_manifest_path): hashlib.sha256(
    reserved_manifest_path.read_bytes()).hexdigest()}
for row in selected:
    directory = RESERVED / row['donor']
    animal_id = row['donor'].removeprefix('donor_')
    donor_receipt_path = directory / 'completed.json'
    donor_receipt_sha = hashlib.sha256(donor_receipt_path.read_bytes()).hexdigest()
    assert donor_receipt_sha == reserved_manifest['donor_receipts'][animal_id]
    donor_receipt = json.loads(donor_receipt_path.read_text())
    assert str(donor_receipt['animal_id']) == animal_id
    assert donor_receipt['image_count'] == row['sections']
    bindings[str(donor_receipt_path)] = donor_receipt_sha
    for name in ('images.npy', 'geometry.npz', 'records.jsonl'):
        path = directory / name
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == donor_receipt['output_sha256'][name]
        bindings[str(path)] = sha
    assert bindings[str(directory / 'records.jsonl')] == row['records_sha256']
    records = [json.loads(line) for line in (directory / 'records.jsonl').open()]
    with np.load(directory / 'geometry.npz') as geometry:
        states = geometry['state'].astype(np.float32)
    assert states.shape == (row['sections'], 12)
    assert len(records) == row['sections']
    assert all(record['training_split'] == 'train' and str(record['animal_id']) == animal_id
        for record in records)
    donors.append({'animal_id': animal_id, 'images': directory / 'images.npy',
        'state': states, 'identities': [{key: record[key] for key in
        ('animal_id', 'specimen_id', 'experiment_id', 'section_id')} for record in records]})
real_context = {'donors': donors, 'bindings': bindings, 'label_role': 'weak Allen affine'}

sources = {
    'coronal': (ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
        ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
        'model_pixel_to_ap_dv_ml_um'),
    'sagittal': (ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
        ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
        'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
}
acquired, acquired_bindings = [], {}
for domain, (records_path, images_path, affine_key) in sources.items():
    acquired_bindings[str(records_path)] = hashlib.sha256(records_path.read_bytes()).hexdigest()
    acquired_bindings[str(images_path)] = hashlib.sha256(images_path.read_bytes()).hexdigest()
    by_row = {record['array_row_index']: record
        for record in (json.loads(line) for line in records_path.open())}
    images = np.load(images_path, mmap_mode='r')
    for case in transfer_cases:
        if case['domain'] != domain or hashlib.sha256(
                f'{domain}|{case["donor_id"]}'.encode()).digest()[0] % 2 != 1:
            continue
        record = by_row[case['array_row_index']]
        assert record['section_id'] == case['section_id']
        image = torch.from_numpy(np.asarray(images[case['array_row_index'], 0],
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
assert not {str(donor['animal_id']) for donor in donors} & {
    str(item['case']['donor_id']) for item in acquired}

old_panel_paths = {
    'frozen_270_v9': ROOT / 'runs/pose_capture_270_v9_continuation_001/inner_panel.pt',
    'frozen_268_v8': ROOT / 'runs/pose_capture_268_v8_train_inner_002/inner_panel.pt',
}
old_panels = {name: torch.load(path, map_location='cpu', weights_only=False)
    for name, path in old_panel_paths.items()}
assert all(len(panel) == 64 for panel in old_panels.values())
context = load_streaming_synthetic_v7_64(device='cuda')
config = {'candidate': 297, 'parent': 275, 'seed': SEED, 'side': SIDE,
    'synthetic_per_batch': BATCH, 'updates_per_arm': UPDATES,
    'readout_batches': READOUTS, 'virtual_weak_loss_weight': WEAK_WEIGHT,
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        for name in ('train_pose_real_volume_mix_297.py',
            'arbitrary_plane_reserved_real_oblique_v3.py',
            'arbitrary_plane_one_shot_slide_artifacts_v3.py',
            'arbitrary_plane_one_shot_slide_artifacts_v9.py',
            'arbitrary_plane_streaming_synthetic_v7_64.py',
            'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
            'arbitrary_plane_one_shot_model.py',
            'arbitrary_plane_full_frame_primitives.py',
            'global_atlas_contrast_090.py')},
    'parent_checkpoint_sha256': parent_sha,
    'transfer_cases_sha256': transfer_receipt['cases_sha256'],
    'geometry_audit_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (audit_rows_path, audit_summary_path)},
    'reserved_source_sha256': bindings, 'acquired_source_sha256': acquired_bindings,
    'old_panel_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in old_panel_paths.items()},
    'synthetic_source_provenance': context['provenance'],
    'virtual_train_animal_ids': [donors[index]['animal_id'] for index in virtual_gradient_indices],
    'virtual_inner_readout_animal_ids': [donors[index]['animal_id']
        for index in virtual_readout_indices],
    'excluded_audit_animal_ids': sorted(excluded),
    'acquired_readout_donor_ids': sorted({str(item['case']['donor_id']) for item in acquired}),
    'gradient_synthetic_bases': list(range(56)), 'inner_synthetic_bases': list(range(56, 64)),
    'loss': 'unchanged 275 physical-site+five-point+normal per-case loss; real weight 0.10',
    'optimizer': 'inherited 275 AdamW moments, LR 3e-5, clip 2',
    'weak_affine_role': 'virtual TRAIN-donor label and fixed acquired diagnostic, never expert truth',
    'no_external_weights_features_pseudolabels': True,
    'no_dev_cal_final_or_public_benchmark': True,
    'calibrated': False, 'torch': torch.__version__, 'numpy': np.__version__}
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'config.json').write_text(json.dumps(config, indent=2))

panel_rng = np.random.default_rng(SEED + 1)
fresh_panel, panel_attempts = [], 0
with (RUN / 'inner_draws.jsonl').open('w') as output:
    for base in range(56, 64):
        for slot in range(8):
            while True:
                panel_attempts += 1
                variant = int(panel_rng.integers(VARIANTS))
                subject_index = base * VARIANTS + variant
                draw_seed = SEED + 1_000_000_000 + panel_attempts
                with torch.no_grad():
                    sample = sample_one_shot_slide_artifacts_v9(
                        context, [subject_index], draw_seed, SIDE)
                used = bool(sample['eligible'][0])
                output.write(json.dumps({'base_index': base, 'slot': slot,
                    'subject_index': subject_index, 'draw_seed': draw_seed,
                    'used': used, 'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    break
                del sample
            valid = sample['valid_mask'][0]
            pixels = valid.flatten().nonzero()[:, 0]
            selected_pixels = pixels[torch.linspace(0, len(pixels) - 1,
                min(1024, len(pixels)), device='cuda').long()]
            chart = torch.stack((selected_pixels.remainder(SIDE),
                selected_pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            true_normal = full_frame_state_to_components(sample['state'][0])[1][:, 2]
            angle = float(torch.rad2deg(torch.acos(true_normal.abs().max().clamp(0, 1))))
            fresh_panel.append({'inputs': sample['inputs'][0].cpu(),
                'state': sample['state'][0].cpu(),
                'reflection': sample['reflection'][0].cpu(), 'chart': chart.cpu(),
                'base_index': base, 'slot': slot,
                'physical_section_id': sample['provenance'][0]['physical_section_id'],
                'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                'obliquity_deg': angle, 'provenance': sample['provenance'][0]})
            del sample
torch.save(fresh_panel, RUN / 'inner_panel.pt')

virtual_panel, virtual_panel_attempts = [], 0
with (RUN / 'virtual_inner_draws.jsonl').open('w') as output:
    for donor_index in virtual_readout_indices:
        for slot in range(3):
            while True:
                virtual_panel_attempts += 1
                draw_seed = SEED + 4_000_000_000 + virtual_panel_attempts
                with torch.no_grad():
                    sample = sample_reserved_real_oblique_v3(real_context,
                        donor_index, draw_seed, side=SIDE, device='cuda')
                used = bool(sample['eligible'][0])
                output.write(json.dumps({'donor_index': donor_index, 'slot': slot,
                    'draw_seed': draw_seed, 'used': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    break
                del sample
            valid = sample['valid_mask'][0]
            pixels = valid.flatten().nonzero()[:, 0]
            selected_pixels = pixels[torch.linspace(0, len(pixels) - 1,
                min(1024, len(pixels)), device='cuda').long()]
            chart = torch.stack((selected_pixels.remainder(SIDE),
                selected_pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            true_normal = full_frame_state_to_components(sample['state'][0])[1][:, 2]
            angle = float(torch.rad2deg(torch.acos(true_normal.abs().max().clamp(0, 1))))
            virtual_panel.append({'inputs': sample['inputs'][0].cpu(),
                'state': sample['state'][0].cpu(),
                'reflection': sample['reflection'][0].cpu(), 'chart': chart.cpu(),
                'donor_id': donors[donor_index]['animal_id'],
                'section_id': sample['provenance'][0]['section_id'],
                'visible_fraction': float((sample['visible'][0] > .25).float().mean()),
                'obliquity_deg': angle, 'provenance': sample['provenance'][0]})
            del sample
assert len(virtual_panel) == 24
torch.save(virtual_panel, RUN / 'virtual_inner_panel.pt')

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
models, optimizers = {}, {}
for arm in ('control', 'mixed'):
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda()
    model.load_state_dict(parent['model'])
    model.requires_grad_(False)
    for module in (model.encoder, model.lateral, model.pose,
                   model.anchor_pose, model.anchor_global):
        module.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-5, weight_decay=1e-4)
    optimizer.load_state_dict(parent['optimizer'])
    optimizer.param_groups[0]['lr'] = 3e-5
    models[arm], optimizers[arm] = model, optimizer
del parent
corners = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
    [95.5, 95.5]], device='cuda') / SIDE
reflection_options = torch.arange(2, device='cuda')[None, None]


def pose_loss(model, inputs, states, reflections, charts):
    prediction = model.predict(inputs)
    count = len(inputs)
    proposed_states = prediction['state'][:, :, None].expand(-1, -1, 2, -1).float()
    proposed_reflections = reflection_options.expand(count, model.modes, 2)
    prior = (prediction['log_mass'].float()[..., None] + torch.stack((
        F.logsigmoid(-prediction['reflection_logit'].float()),
        F.logsigmoid(prediction['reflection_logit'].float())), -1)).flatten(1)
    true = rigid_points_090(states, reflections, charts, (SIDE, SIDE))
    proposed = rigid_points_090(proposed_states, proposed_reflections,
        charts[:, None, None], (SIDE, SIDE))
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
    near_cost = physical.reshape(count, model.modes, 2).min(-1).values.gather(1, near_mode)
    neighbourhood = F.softmax(40 * (nearest.values - nearest.values[:, :1]), -1)
    return (-1.5 * torch.logsumexp(prior - physical / 1.5, -1)
        + .5 * F.kl_div(prior, target_mass, reduction='none').sum(-1)
        + .5 * physical[:, :2 * model.base_modes].min(-1).values
        + (neighbourhood * near_cost).sum(-1))


synthetic_rng = np.random.default_rng(SEED + 2)
real_rng = np.random.default_rng(SEED + 3)
attempted, accepted, virtual_attempted = 0, 0, 0
readouts = {}
started = time.perf_counter()
with (RUN / 'draws.jsonl').open('w') as draws, (RUN / 'training.jsonl').open('w') as training, \
        (RUN / 'readouts.jsonl').open('w') as output:
    for step in range(UPDATES + 1):
        if step:
            images, states, reflections, masks = [], [], [], []
            while len(images) < BATCH:
                attempted += 1
                base = int(synthetic_rng.integers(56))
                variant = int(synthetic_rng.integers(VARIANTS))
                subject_index = base * VARIANTS + variant
                draw_seed = SEED + 2_000_000_000 + attempted
                with torch.no_grad():
                    sample = sample_one_shot_slide_artifacts_v9(
                        context, [subject_index], draw_seed, SIDE)
                used = bool(sample['eligible'][0])
                draws.write(json.dumps({'step': step, 'role': 'matched_synthetic',
                    'attempt': attempted, 'base_index': base, 'subject_index': subject_index,
                    'draw_seed': draw_seed, 'used': used,
                    'provenance': sample['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    accepted += 1
                    images.append(sample['inputs'][0].clone())
                    states.append(sample['state'][0].clone())
                    reflections.append(sample['reflection'][0].clone())
                    masks.append(sample['valid_mask'][0].clone())
                del sample
            synthetic_inputs, synthetic_states, synthetic_reflections, synthetic_valid = map(
                torch.stack, (images, states, reflections, masks))
            while True:
                virtual_attempted += 1
                donor_index = virtual_gradient_indices[int(real_rng.integers(
                    len(virtual_gradient_indices)))]
                real_seed = SEED + 3_000_000_000 + virtual_attempted
                with torch.no_grad():
                    real = sample_reserved_real_oblique_v3(
                        real_context, donor_index, real_seed, side=SIDE, device='cuda')
                used = bool(real['eligible'][0])
                draws.write(json.dumps({'step': step, 'role': 'mixed_weak_virtual_train',
                    'attempt': virtual_attempted, 'used': used,
                    'provenance': real['provenance'][0]}, allow_nan=False) + '\n')
                if used:
                    break
                del real
            synthetic_pixels = torch.multinomial(
                synthetic_valid.flatten(1).float(), 128, replacement=True)
            real_pixels = torch.multinomial(
                real['valid_mask'].flatten(1).float(), 128, replacement=True)
            pixels = torch.cat((synthetic_pixels, real_pixels))
            charts = torch.stack((pixels.remainder(SIDE),
                pixels.div(SIDE, rounding_mode='floor')), -1).float() / SIDE
            mixed_inputs = torch.cat((synthetic_inputs, real['inputs']))
            mixed_states = torch.cat((synthetic_states, real['state']))
            mixed_reflections = torch.cat((synthetic_reflections, real['reflection']))
            step_rng, step_cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            row = {'step': step, 'accepted_synthetic': accepted,
                'attempted_synthetic': attempted, 'attempted_virtual': virtual_attempted}
            for arm in ('control', 'mixed'):
                torch.set_rng_state(step_rng)
                torch.cuda.set_rng_state_all(step_cuda_rng)
                model, optimizer = models[arm], optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                if arm == 'control':
                    per_case = pose_loss(model, synthetic_inputs, synthetic_states,
                        synthetic_reflections, charts[:BATCH])
                    loss = per_case.mean()
                else:
                    per_case = pose_loss(model, mixed_inputs, mixed_states,
                        mixed_reflections, charts)
                    loss = per_case[:BATCH].mean() + WEAK_WEIGHT * per_case[BATCH]
                    row['weak_virtual_pose_loss'] = float(per_case[BATCH].detach())
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    2., error_if_nonfinite=True)
                optimizer.step()
                row[f'{arm}_loss'] = float(loss.detach())
                row[f'{arm}_synthetic_pose_loss'] = float(per_case[:BATCH].mean().detach())
                row[f'{arm}_gradient_norm'] = float(gradient)
            row['elapsed_seconds'] = time.perf_counter() - started
            training.write(json.dumps(row, allow_nan=False) + '\n')
            if step == 1 or step % 100 == 0:
                training.flush()
                draws.flush()
                print(json.dumps({'event': 'train_milestone', 'step': step,
                    'accepted_synthetic': accepted, 'control_loss': row['control_loss'],
                    'mixed_loss': row['mixed_loss']}), flush=True)
            del real, synthetic_inputs, synthetic_states, synthetic_reflections
            del synthetic_valid, mixed_inputs, mixed_states, mixed_reflections, per_case, loss

        if step in READOUTS:
            draws.flush()
            training.flush()
            readouts[step] = {}
            for arm, model in models.items():
                model.eval()
                readouts[step][arm] = {}
                with torch.inference_mode():
                    for panel_name, panel in (('new_v9', fresh_panel), *old_panels.items(),
                                              ('virtual_weak', virtual_panel)):
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
                            candidates = prediction['state'][0, order[:16] // 2][None]
                            points = rigid_points_090(candidates, (order[:16] % 2)[None],
                                chart[None], (SIDE, SIDE))
                            errors = ((points - truth[:, None]).norm(dim=-1).mean(-1) / 1000)[0]
                            true_normal = full_frame_state_to_components(
                                item['state'].cuda())[1][:, 2]
                            first_normal = full_frame_state_to_components(
                                prediction['state'][0, order[0] // 2])[1][:, 2]
                            angle = float(torch.rad2deg(torch.acos(
                                (true_normal * first_normal).sum().abs().clamp(0, 1))))
                            rows.append({'section_id': item.get('physical_section_id',
                                item.get('section_id')), 'base_index': item.get('base_index'),
                                'donor_id': item.get('donor_id'),
                                'top16_hit': bool((errors <= 1.5).any()),
                                'top1_mm': float(errors[0]), 'normal_axis_error_deg': angle,
                                'visible_fraction': item['visible_fraction'],
                                'obliquity_deg': item.get('obliquity_deg')})
                        metrics = {'n': len(rows),
                            'top16_hits': sum(row['top16_hit'] for row in rows),
                            'mean_top1_mm': float(np.mean([row['top1_mm'] for row in rows])),
                            'mean_normal_axis_error_deg': float(np.mean([
                                row['normal_axis_error_deg'] for row in rows]))}
                        readouts[step][arm][panel_name] = {'metrics': metrics, 'rows': rows}
                        output.write(json.dumps({'step': step, 'arm': arm,
                            'panel': panel_name, 'metrics': metrics, 'rows': rows},
                            allow_nan=False) + '\n')
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
                            normal = full_frame_state_to_components(
                                prediction['state'][0, first])[1][:, 2]
                            angle = float(torch.rad2deg(torch.acos((normal * torch.as_tensor(
                                item['normal'], device='cuda')).sum().abs().clamp(0, 1))))
                            rows.append({'donor_id': case['donor_id'],
                                'specimen_id': case['specimen_id'],
                                'experiment_id': case['experiment_id'],
                                'section_id': case['section_id'],
                                'source_image_sha256': case['source_image_sha256'],
                                'first_axis_correct': int(normal.abs().argmax()) == item['axis'],
                                'normal_gap_to_weak_affine_deg': angle})
                        metrics = {'n': len(rows),
                            'first_axis_correct': sum(row['first_axis_correct'] for row in rows),
                            'median_weak_normal_angle_deg': float(np.median([
                                row['normal_gap_to_weak_affine_deg'] for row in rows]))}
                        readouts[step][arm][domain] = {'metrics': metrics, 'rows': rows}
                        output.write(json.dumps({'step': step, 'arm': arm,
                            'panel': domain, 'metrics': metrics, 'rows': rows},
                            allow_nan=False) + '\n')
                model.train()
                checkpoint = RUN / f'{arm}_step_{step:05d}.pt'
                temporary = checkpoint.with_suffix('.tmp')
                torch.save({'step': step, 'arm': arm, 'model': model.state_dict(),
                    'optimizer': optimizers[arm].state_dict(), 'config': config,
                    'synthetic_rng': synthetic_rng.bit_generator.state,
                    'real_rng': real_rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all(),
                    'attempted_synthetic': attempted,
                    'accepted_synthetic': accepted,
                    'attempted_virtual': virtual_attempted,
                    'calibrated': False}, temporary)
                os.replace(temporary, checkpoint)
            output.flush()
            print(json.dumps({'event': 'readout', 'step': step,
                'control': {name: row['metrics'] for name, row in readouts[step]['control'].items()},
                'mixed': {name: row['metrics'] for name, row in readouts[step]['mixed'].items()}}),
                flush=True)

final = readouts[UPDATES]
control, mixed = final['control'], final['mixed']
base_gains = {str(base): float(np.mean([
    control['new_v9']['rows'][i]['top1_mm'] - mixed['new_v9']['rows'][i]['top1_mm']
    for i in range(64) if control['new_v9']['rows'][i]['base_index'] == base]))
    for base in range(56, 64)}
virtual_donor_gains = {str(donor_id): float(np.mean([
    control['virtual_weak']['rows'][i]['top1_mm'] - mixed['virtual_weak']['rows'][i]['top1_mm']
    for i in range(24) if str(control['virtual_weak']['rows'][i]['donor_id']) == str(donor_id)]))
    for donor_id in config['virtual_inner_readout_animal_ids']}
gate = {
    'new_top1_gain_at_least_0p15mm':
        control['new_v9']['metrics']['mean_top1_mm'] - mixed['new_v9']['metrics']['mean_top1_mm'] >= .15,
    'at_least_six_positive_bases': sum(gain > 0 for gain in base_gains.values()) >= 6,
    'new_top16_loses_at_most_two': mixed['new_v9']['metrics']['top16_hits'] >=
        control['new_v9']['metrics']['top16_hits'] - 2,
    'frozen_270_top16_loses_at_most_two': mixed['frozen_270_v9']['metrics']['top16_hits'] >=
        control['frozen_270_v9']['metrics']['top16_hits'] - 2,
    'frozen_268_top16_loses_at_most_two': mixed['frozen_268_v8']['metrics']['top16_hits'] >=
        control['frozen_268_v8']['metrics']['top16_hits'] - 2,
    'virtual_weak_top1_gain_at_least_0p15mm':
        control['virtual_weak']['metrics']['mean_top1_mm'] -
        mixed['virtual_weak']['metrics']['mean_top1_mm'] >= .15,
    'at_least_six_virtual_readout_donors_positive':
        sum(gain > 0 for gain in virtual_donor_gains.values()) >= 6,
    'sagittal_weak_family_gain_at_least_four': mixed['sagittal']['metrics']['first_axis_correct'] >=
        control['sagittal']['metrics']['first_axis_correct'] + 4,
    'coronal_weak_family_loses_at_most_one': mixed['coronal']['metrics']['first_axis_correct'] >=
        control['coronal']['metrics']['first_axis_correct'] - 1,
}
receipt = {'candidate': 297, 'parent_checkpoint_sha256': parent_sha,
    'updates_per_arm': UPDATES, 'accepted_synthetic_presentations_per_arm': accepted,
    'attempted_synthetic_draws': attempted, 'attempted_virtual_draws': virtual_attempted,
    'virtual_train_donors': len(virtual_gradient_indices),
    'virtual_inner_readout_donors': len(virtual_readout_indices),
    'virtual_inner_readout_cases': len(virtual_panel),
    'acquired_readout_cases': len(acquired),
    'panel_attempts': panel_attempts,
    'virtual_panel_attempts': virtual_panel_attempts,
    'base_gains_mm': base_gains, 'virtual_donor_gains_mm': virtual_donor_gains,
    'gate': gate, 'gate_pass': all(gate.values()), 'readouts': {
        str(step): {arm: {name: panel['metrics'] for name, panel in arm_rows.items()}
            for arm, arm_rows in step_rows.items()}
        for step, step_rows in readouts.items()},
    'output_sha256': {name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        for name in ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
            'virtual_inner_draws.jsonl', 'virtual_inner_panel.pt',
            'draws.jsonl', 'training.jsonl', 'readouts.jsonl')},
    'checkpoint_sha256': {f'{arm}_{step}': hashlib.sha256(
        (RUN / f'{arm}_step_{step:05d}.pt').read_bytes()).hexdigest()
        for arm in ('control', 'mixed') for step in READOUTS},
    'virtual_affines_are_weak_labels': True, 'calibrated': False,
    'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({'event': 'complete', 'gate': gate, 'gate_pass': all(gate.values())}),
    flush=True)
