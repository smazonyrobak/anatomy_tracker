"""Bounded v9 versus v11 anatomy-contrast pose continuation from frozen 275."""
import copy
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

from training.arbitrary_plane_allen_atlas_binding_v6 import (
    ANNOTATION_PATH_V6, ANNOTATION_RAW_SHA256_V6, _decode_and_preprocess_allen_v6,
)
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_one_shot_slide_artifacts_v11 import sample_one_shot_slide_artifacts_v11
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090


SIDE, BATCH, UPDATES = 192, 4, 1000
STEPS = (0, 500, 1000)
SEEDS = {'control_v9': 2026101227901, 'anatomy_v11': 2026201227901}
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/ANATOMY_CONTRAST_279_PILOT_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/pose_capture_275_v9_continuation_001'
OLD = ROOT / 'runs/pose_capture_270_v9_continuation_001'
TRANSFER = ROOT / 'runs/acquired_train_transfer_272_001'
PREVIEW = ROOT / 'previews/anatomy_contrast_278_actual_generator_001'
CENSUS = ROOT / 'runs/anatomy_contrast_278_texture_census_001'
RUN = ROOT / 'runs/anatomy_contrast_279_pilot_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

source_names = (
    'train_pose_anatomy_contrast_279.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_one_shot_slide_artifacts_v11.py',
    'arbitrary_plane_streaming_synthetic_v7_appearance_v3.py',
    'arbitrary_plane_one_shot_slide_artifacts_v7.py',
    'arbitrary_plane_one_shot_slide_artifacts_v6.py',
    'arbitrary_plane_streaming_synthetic_v7_64.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_full_frame_primitives.py',
    'global_atlas_contrast_090.py', 'arbitrary_plane_allen_atlas_binding_v6.py',
)
source_hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                 for name in source_names}
parent_receipt = json.loads((PARENT / 'completed.json').read_text())
old_receipt = json.loads((OLD / 'completed.json').read_text())
transfer_receipt = json.loads((TRANSFER / 'completed.json').read_text())
preview_receipt = json.loads((PREVIEW / 'completed.json').read_text())
census_receipt = json.loads((CENSUS / 'completed.json').read_text())
assert parent_receipt['candidate'] == 275 and parent_receipt['selected_step'] == 8000
assert old_receipt['candidate'] == 270 and transfer_receipt['candidate'] == 272
checkpoint_path = PARENT / 'pose_step_08000.pt'
with checkpoint_path.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == parent_receipt['checkpoint_sha256']['8000']
with (PARENT / 'config.json').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == parent_receipt['config_sha256']
with (PARENT / 'inner_panel.pt').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == parent_receipt['inner_panel_sha256']
with (OLD / 'inner_panel.pt').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == old_receipt['inner_panel_sha256']
with (TRANSFER / 'cases.jsonl').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == transfer_receipt['cases_sha256']
with (TRANSFER / 'config.json').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == transfer_receipt['config_sha256']
transfer_config = json.loads((TRANSFER / 'config.json').read_text())
with (PREVIEW / 'selection.json').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == preview_receipt['selection_sha256']
with (PREVIEW / 'atlas_source_input.png').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == preview_receipt['image_sha256']
with (CENSUS / 'rows.jsonl').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == census_receipt['rows_sha256']
with (CENSUS / 'summary.json').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == census_receipt['summary_sha256']
preview_selection = json.loads((PREVIEW / 'selection.json').read_text())
assert preview_selection['generator_sha256'] == census_receipt['generator_sha256'] == \
    source_hashes['arbitrary_plane_one_shot_slide_artifacts_v11.py']
assert census_receipt['preview_selection_sha256'] == preview_receipt['selection_sha256']

parent = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
assert parent['step'] == 8000 and parent['config']['candidate'] == 275
new_panel = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
old_panel = torch.load(OLD / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(new_panel) == len(old_panel) == 64
context = load_streaming_synthetic_v7_64(device='cuda')
atlas_array, annotation_array = _decode_and_preprocess_allen_v6()
annotation = torch.from_numpy(annotation_array.astype(np.int64)).cuda()
del atlas_array, annotation_array

sources = {
    'coronal': (ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
                ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
                'model_pixel_to_ap_dv_ml_um'),
    'sagittal': (ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
                 ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
                 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
}
transfer_cases = [json.loads(line) for line in (TRANSFER / 'cases.jsonl').open()]
assert len(transfer_cases) == 98
acquired, source_data = [], {}
for domain, (records_path, images_path, affine_key) in sources.items():
    records = {row['array_row_index']: row for row in
               (json.loads(line) for line in records_path.open())}
    images = np.load(images_path, mmap_mode='r')
    with records_path.open('rb') as stream:
        records_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    with images_path.open('rb') as stream:
        images_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    source_data[domain] = {'records_sha256': records_hash, 'images_sha256': images_hash}
    assert records_hash == transfer_config['datasets'][domain]['records_sha256']
    assert images_hash == transfer_config['datasets'][domain]['images_sha256']
    for case in transfer_cases:
        if case['domain'] != domain:
            continue
        record = records[case['array_row_index']]
        assert record['section_id'] == case['section_id']
        assert record.get('source_image_sha256', record.get('image_sha256')) == case['source_image_sha256']
        assert record['training_split' if domain == 'coronal' else 'split'] == 'train'
        image = torch.from_numpy(np.asarray(images[case['array_row_index'], 0],
                                            dtype=np.float32).copy())[None, None].cuda()
        if domain == 'sagittal':
            image = F.interpolate(image, (SIDE, SIDE), mode='area')
        matrix = np.asarray(record[affine_key], dtype=np.float64)
        weak_normal = np.cross(matrix[:, 0], matrix[:, 1])
        weak_normal /= np.linalg.norm(weak_normal)
        weak_axis = int(np.abs(weak_normal).argmax())
        assert case['weak_axis'] == ('AP', 'DV', 'ML')[weak_axis]
        assert weak_axis == (0 if domain == 'coronal' else 2)
        parity = 'even' if hashlib.sha256(f'{domain}|{case["donor_id"]}'.encode()).digest()[0] % 2 == 0 else 'odd'
        acquired.append({'case': case, 'image': image, 'weak_normal': weak_normal,
                         'weak_axis': weak_axis, 'donor_parity': parity})
assert sum(item['case']['domain'] == 'coronal' for item in acquired) == 58
assert sum(item['case']['domain'] == 'sagittal' for item in acquired) == 40
assert len({(item['case']['domain'], item['case']['donor_id']) for item in acquired}) == 98
assert {domain: {parity: sum(item['case']['domain'] == domain and item['donor_parity'] == parity
                             for item in acquired) for parity in ('even', 'odd')}
        for domain in sources} == {'coronal': {'even': 30, 'odd': 28},
                                   'sagittal': {'even': 18, 'odd': 22}}

config = {'candidate': 279, 'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
          'source_sha256': source_hashes,
          'parent_completed_sha256': hashlib.sha256((PARENT / 'completed.json').read_bytes()).hexdigest(),
          'parent_config_sha256': parent_receipt['config_sha256'],
          'parent_checkpoint_sha256': parent_receipt['checkpoint_sha256']['8000'],
          'parent_selected_step': 8000,
          'new_275_panel_sha256': parent_receipt['inner_panel_sha256'],
          'frozen_270_panel_sha256': old_receipt['inner_panel_sha256'],
          'transfer_cases_sha256': transfer_receipt['cases_sha256'],
          'preview_278_selection_sha256': preview_receipt['selection_sha256'],
          'preview_278_image_sha256': preview_receipt['image_sha256'],
          'census_278_rows_sha256': census_receipt['rows_sha256'],
          'census_278_summary_sha256': census_receipt['summary_sha256'],
          'annotation_path': str(ANNOTATION_PATH_V6),
          'annotation_raw_sha256': ANNOTATION_RAW_SHA256_V6,
          'source_data': source_data, 'synthetic_provenance': context['provenance'],
          'arms': SEEDS, 'optimizer_batches_per_arm': UPDATES, 'batch_size': BATCH,
          'readout_batches': STEPS, 'learning_rate': 3e-5,
          'gradient_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
          'loss': 'unchanged 275 physical-site+five-point+normal mixture',
          'optimizer': 'same frozen 275 AdamW moments for both arms, LR 3e-5, clip 2',
          'sampling': 'one newly drawn eligible full-sphere physical plane per presentation; no paired appearances',
          'acquired_affine_role': '98 TRAIN donor median sections, weak orientation diagnostic and fixed gate only',
          'donor_parity': 'first SHA-256 byte of domain|donor_id, modulo 2',
          'gate_checkpoint': 1000, 'checkpoint_selection': 'none',
          'weak_affines_or_acquired_images_used_for_gradients': False,
          'no_prior_external_weights_features_pseudolabels': True,
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
        optimizer = torch.optim.AdamW(parameters, lr=3e-5, weight_decay=1e-4)
        optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        optimizer.param_groups[0]['lr'] = 3e-5
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
                                  sample_one_shot_slide_artifacts_v11(context, annotation,
                                  [subject_index], draw_seed, SIDE))
                    used = bool(sample['eligible'][0])
                    draws.write(json.dumps({'arm': arm, 'step': step, 'attempt': attempted,
                        'base_index': base, 'variant': variant, 'subject_index': subject_index,
                        'draw_seed': draw_seed, 'used': used,
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
                    for panel_name, panel in (('new_275_v9', new_panel), ('frozen_270_v9', old_panel)):
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
                                'base_index': item['base_index'],
                                'top16_hit': bool((errors <= 1.5).any()),
                                'top1_mm': float(errors[0]), 'normal_axis_error_deg': angle,
                                'visible_fraction': item['visible_fraction']})
                        metrics = {'n': len(rows), 'top16_hits': sum(row['top16_hit'] for row in rows),
                            'mean_top1_mm': float(np.mean([row['top1_mm'] for row in rows])),
                            'mean_normal_axis_error_deg': float(np.mean([
                                row['normal_axis_error_deg'] for row in rows]))}
                        panel_scores[panel_name] = metrics
                        output.write(json.dumps({'arm': arm, 'step': step, 'panel': panel_name,
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
                            normal = full_frame_state_to_components(
                                prediction['state'][0, first])[1][:, 2]
                            angle = float(torch.rad2deg(torch.acos((normal * torch.as_tensor(
                                item['weak_normal'], device='cuda')).sum().abs().clamp(0, 1))))
                            selected_axis = int(normal.abs().argmax())
                            rows.append({'donor_id': case['donor_id'],
                                'specimen_id': case['specimen_id'],
                                'experiment_id': case['experiment_id'],
                                'section_id': case['section_id'],
                                'array_row_index': case['array_row_index'],
                                'source_image_sha256': case['source_image_sha256'],
                                'donor_parity': item['donor_parity'],
                                'weak_axis': ('AP', 'DV', 'ML')[item['weak_axis']],
                                'selected_axis': ('AP', 'DV', 'ML')[selected_axis],
                                'first_axis_correct': selected_axis == item['weak_axis'],
                                'normal_gap_to_weak_affine_deg': angle})
                        metrics = {}
                        for parity in ('total', 'even', 'odd'):
                            stratum = rows if parity == 'total' else [row for row in rows
                                if row['donor_parity'] == parity]
                            metrics[parity] = {'n': len(stratum),
                                'first_axis_correct': sum(row['first_axis_correct'] for row in stratum),
                                'median_weak_normal_angle_deg': float(np.median([
                                    row['normal_gap_to_weak_affine_deg'] for row in stratum]))}
                        panel_scores[domain] = metrics
                        output.write(json.dumps({'arm': arm, 'step': step, 'panel': domain,
                            'metrics': metrics, 'rows': rows}, allow_nan=False) + '\n')
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
        assert accepted == UPDATES * BATCH
        del model, optimizer
        torch.cuda.empty_cache()

control, anatomy = readouts['control_v9'][UPDATES], readouts['anatomy_v11'][UPDATES]
deltas = {
    'new_275_mean_top1_drop_mm': control['new_275_v9']['mean_top1_mm'] - anatomy['new_275_v9']['mean_top1_mm'],
    'new_275_top16_loss': control['new_275_v9']['top16_hits'] - anatomy['new_275_v9']['top16_hits'],
    'frozen_270_mean_top1_worsening_mm': anatomy['frozen_270_v9']['mean_top1_mm'] - control['frozen_270_v9']['mean_top1_mm'],
    'sagittal_total_family_gain': anatomy['sagittal']['total']['first_axis_correct'] - control['sagittal']['total']['first_axis_correct'],
    'sagittal_odd_family_gain': anatomy['sagittal']['odd']['first_axis_correct'] - control['sagittal']['odd']['first_axis_correct'],
    'sagittal_even_family_gain': anatomy['sagittal']['even']['first_axis_correct'] - control['sagittal']['even']['first_axis_correct'],
    'sagittal_median_weak_angle_drop_deg': control['sagittal']['total']['median_weak_normal_angle_deg'] - anatomy['sagittal']['total']['median_weak_normal_angle_deg'],
    'coronal_total_family_loss': control['coronal']['total']['first_axis_correct'] - anatomy['coronal']['total']['first_axis_correct'],
}
gate = {'new_275_mean_top1_drop_at_least_0_20mm': deltas['new_275_mean_top1_drop_mm'] >= .20,
    'new_275_top16_loss_at_most_2': deltas['new_275_top16_loss'] <= 2,
    'frozen_270_mean_top1_worsening_at_most_0_10mm': deltas['frozen_270_mean_top1_worsening_mm'] <= .10,
    'sagittal_total_family_gain_at_least_6': deltas['sagittal_total_family_gain'] >= 6,
    'sagittal_odd_family_gain_at_least_2': deltas['sagittal_odd_family_gain'] >= 2,
    'sagittal_even_family_gain_at_least_2': deltas['sagittal_even_family_gain'] >= 2,
    'sagittal_median_weak_angle_drop_at_least_5deg': deltas['sagittal_median_weak_angle_drop_deg'] >= 5,
    'coronal_total_family_loss_at_most_2': deltas['coronal_total_family_loss'] <= 2}
checkpoint_hashes = {}
for arm in SEEDS:
    for step in STEPS[1:]:
        with (RUN / f'{arm}_step_{step:05d}.pt').open('rb') as stream:
            checkpoint_hashes[f'{arm}_{step}'] = hashlib.file_digest(stream, 'sha256').hexdigest()
output_hashes = {}
for name in ('config.json', 'draws.jsonl', 'training.jsonl', 'readouts.jsonl'):
    with (RUN / name).open('rb') as stream:
        output_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
receipt = {'candidate': 279, 'steps': STEPS, 'counters': counters,
    'readouts': readouts, 'gate_deltas': deltas, 'gate': gate, 'gate_pass': all(gate.values()),
    'result_role': 'bounded TRAIN-inner mechanism pilot; pass permits confirmation only, not GUI promotion',
    'protocol_sha256': config['protocol_sha256'],
    'source_sha256': source_hashes,
    'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
    'output_sha256': output_hashes, 'checkpoint_sha256': checkpoint_hashes,
    'weak_affines_or_acquired_images_used_for_gradients': False,
    'calibrated': False, 'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'completed.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({'event': 'complete', 'gate': gate, 'gate_pass': all(gate.values()),
                  'gate_deltas': deltas, 'readouts_1000': {arm: readouts[arm][UPDATES]
                                                     for arm in SEEDS}}), flush=True)
