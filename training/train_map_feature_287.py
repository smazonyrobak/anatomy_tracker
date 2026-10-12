"""Matched TRAIN-only source-feature and atlas-conditioning map pilot."""
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

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_ribbon_v6 import ribbon_derivative_frobenius_bound
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS, load_streaming_synthetic_v7_64
from training.train_image_map_282 import (LIMIT, MAP_MODULES, PARENT, POSE_BUFFERS,
    POSE_MODULES, SEED as SEED282, SIDE, digest, exact_map, make_model,
    mapped_sites, rigid_surface, sites_from_sample, tensor_digest)


SEED, UPDATES = 20261012287, 3000
STEPS = (0, 750, 1500, 3000)
ARMS = ('trainable_atlas', 'trainable_direct')
CONTROL = ROOT / 'runs/pose_gauge_map_286_001'
RUN = ROOT / 'runs/map_feature_287_001'
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/MAP_FEATURE_287_PROTOCOL_20261012.md'
EVENTS = ('fragment', 'fold', 'bubble', 'tile_seam')
SOURCE_NAMES = ('train_map_feature_287.py', 'train_image_map_282.py',
    'arbitrary_plane_one_shot_model.py', 'arbitrary_plane_postartifact_gauge_v1.py',
    'arbitrary_plane_one_shot_slide_artifacts_v9.py')


def pair(context, subject_index, draw_seed):
    old = sample_one_shot_slide_artifacts_v9(context, [subject_index], draw_seed, SIDE)
    new = sample_postartifact_gauge_v1(context, [subject_index], draw_seed, SIDE,
                                        source_sample=old)
    for key in ('inputs', 'centre', 'valid_mask', 'reflection', 'offsets', 'weights'):
        assert torch.equal(old[key], new[key]), key
    assert torch.equal(old['state'], new['pre_artifact_state'])
    return old, new


def new_model(pose_weights, pose_buffers, initial_map_weights, conditioned):
    if conditioned:
        model = make_model(pose_weights, pose_buffers, 16)
    else:
        torch.manual_seed(SEED282 + 2)
        torch.cuda.manual_seed_all(SEED282 + 2)
        model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
            atlas_conditioning=False, warp_control_side=16).cuda()
        for name in POSE_MODULES:
            prefix = name + '.'
            getattr(model, name).load_state_dict({key[len(prefix):]: value
                for key, value in pose_weights.items() if key.startswith(prefix)}, strict=True)
        with torch.no_grad():
            for name in POSE_BUFFERS:
                getattr(model, name).copy_(pose_buffers[name])
        model.requires_grad_(False)
    for name in MAP_MODULES:
        if hasattr(model, name):
            prefix = name + '.'
            getattr(model, name).load_state_dict({key[len(prefix):]: value
                for key, value in initial_map_weights.items() if key.startswith(prefix)}, strict=True)
            getattr(model, name).requires_grad_(True)
    for name in ('encoder', 'lateral'):
        getattr(model, name).requires_grad_(True)
    return model


def readout(model, atlas, panel, arm, step, stream):
    model.eval()
    rows = []
    with torch.inference_mode():
        for item in panel:
            inputs = item['inputs'][None].cuda()
            state = item['state'][None].cuda()
            reflection = item['reflection'][None].cuda()
            offsets = item['offsets'][None].cuda()
            weights = item['weights'][None].cuda()
            indices = item['site_indices'].cuda()
            target = item['site_target'].cuda()
            rigid, _, basis = rigid_surface(state, reflection)
            rigid_points = mapped_sites(
                {'centre_surface_ccf_ap_dv_ml_um': rigid[:, None]}, indices)[0, 0]
            mapped, points = exact_map(model, inputs, state, reflection,
                                        offsets, weights, atlas, indices)
            local = mapped['local_displacement_um'][:, 0]
            canonical = torch.where(reflection[:, None, None, None].bool(), local.flip(-1), local)
            derivative = ribbon_derivative_frobenius_bound(
                canonical, torch.zeros_like(canonical), basis, offsets)
            prediction = model.predict(inputs)
            log_reflection = torch.stack((F.logsigmoid(-prediction['reflection_logit']),
                                          F.logsigmoid(prediction['reflection_logit'])), -1)
            chosen = (prediction['log_mass'][..., None] + log_reflection).flatten(1).argmax(-1)
            chosen_state = prediction['state'][0, chosen[0] // 2][None]
            chosen_reflection = (chosen % 2).to(reflection)
            pose_surface, _, _ = rigid_surface(chosen_state, chosen_reflection)
            pose_points = mapped_sites(
                {'centre_surface_ccf_ap_dv_ml_um': pose_surface[:, None]}, indices)[0, 0]
            no_warp = float((rigid_points - target).norm(dim=-1).mean() / 1000)
            error = float((points - target).norm(dim=-1).mean() / 1000)
            row = {'arm': arm, 'step': step, 'base_index': item['base_index'],
                'slot': item['slot'], 'physical_section_id': item['physical_section_id'],
                'input_sha256': item['input_sha256'], 'target_sha256': item['target_sha256'],
                'no_warp_mm': no_warp, 'mapped_mm': error, 'map_gain_mm': no_warp - error,
                'surviving_site_shift_mm': float((points - rigid_points).norm(dim=-1).mean() / 1000),
                'pose_first_choice_mm': float((pose_points - target).norm(dim=-1).mean() / 1000),
                'postlimit_derivative': float(derivative[0]),
                'visible_fraction': item['visible_fraction'], 'event_flags': item['event_flags']}
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
    model.train()
    by_base = {str(base): float(np.mean([row['map_gain_mm'] for row in rows
                                       if row['base_index'] == base])) for base in range(56, 64)}
    def subgroup(selected):
        return {'n': len(selected), 'gain_mean_mm':
                float(np.mean([row['map_gain_mm'] for row in selected])) if selected else None}
    return {'sections': len(rows),
        'no_warp_mean_mm': float(np.mean([row['no_warp_mm'] for row in rows])),
        'mapped_mean_mm': float(np.mean([row['mapped_mm'] for row in rows])),
        'map_gain_mean_mm': float(np.mean([row['map_gain_mm'] for row in rows])),
        'pose_first_choice_mean_mm': float(np.mean([row['pose_first_choice_mm'] for row in rows])),
        'surviving_site_shift_mean_mm': float(np.mean([row['surviving_site_shift_mm'] for row in rows])),
        'max_postlimit_derivative': max(row['postlimit_derivative'] for row in rows),
        'nonnegative_gain_bases': sum(gain >= 0 for gain in by_base.values()),
        'gain_by_base_mm': by_base,
        'by_visible_support': {name: subgroup([row for row in rows if
            (row['visible_fraction'] < .15) == low])
            for name, low in (('under_15pct', True), ('at_least_15pct', False))},
        'by_event': {event: {str(flag).lower(): subgroup([row for row in rows if
            row['event_flags'][event] == flag]) for flag in (False, True)}
            for event in EVENTS}}


assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()
control_receipt = json.loads((CONTROL / 'completed.json').read_text(encoding='utf8'))
control_config = json.loads((CONTROL / 'config.json').read_text(encoding='utf8'))
assert control_receipt['candidate'] == 286 and not control_receipt['gate_pass']
assert digest(CONTROL / 'completed.json') == \
    '08b88ad1906303b33947dd55ce7e6d730883266f28682a8904b56b01a27551eb'
assert digest(CONTROL / 'training_schedule.jsonl') == \
    control_receipt['files_sha256']['training_schedule.jsonl']
control_checkpoint = CONTROL / 'postartifact_gauge_v1_step_03000.pt'
assert digest(control_checkpoint) == control_receipt['checkpoint_sha256']['postartifact_gauge_v1']['3000']
control_weights = torch.load(control_checkpoint, map_location='cpu', weights_only=False)['mapper']
initial_checkpoint = CONTROL / 'postartifact_gauge_v1_step_00000.pt'
assert digest(initial_checkpoint) == control_receipt['checkpoint_sha256']['postartifact_gauge_v1']['0']
initial_map_weights = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)['mapper']
parent_receipt = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
assert parent_receipt['candidate'] == 275 and parent_receipt['selected_step'] == 8000
parent_path = PARENT / 'pose_step_08000.pt'
assert digest(parent_path) == parent_receipt['checkpoint_sha256']['8000']
parent = torch.load(parent_path, map_location='cpu', weights_only=False)
pose_weights = {name: value for name, value in parent['model'].items()
                if name.split('.')[0] in POSE_MODULES}
pose_buffers = {name: parent['model'][name] for name in POSE_BUFFERS}
del parent
context = load_streaming_synthetic_v7_64(device='cuda')
for name, sha in control_config['source_sha256'].items():
    assert digest(SOURCE / name) == sha
schedule = [json.loads(line) for line in (CONTROL / 'training_schedule.jsonl').open(encoding='utf8')]
assert len(schedule) == UPDATES
prior_ids = {entry['physical_section_id'] for entry in schedule}
prior_ids.update(json.loads(line)['physical_section_id'] for line in
    (CONTROL / 'inner_draws.jsonl').open(encoding='utf8'))
RUN.mkdir(parents=True, exist_ok=False)

panel, attempts = [], 0
rng = np.random.default_rng(SEED + 1)
with (RUN / 'inner_draws.jsonl').open('w', encoding='utf8') as draws:
    for base in range(56, 64):
        for slot in range(8):
            while True:
                attempts += 1
                subject_index = base * VARIANTS + int(rng.integers(VARIANTS))
                draw_seed = SEED + 1_000_000_000 + attempts
                with torch.no_grad():
                    old, new = pair(context, subject_index, draw_seed)
                identity = old['provenance'][0]
                valid_pixels = int(old['valid_mask'][0].sum())
                used = bool(old['eligible'][0]) and valid_pixels >= 512
                draws.write(json.dumps({'base_index': base, 'slot': slot,
                    'attempt': attempts, 'subject_index': subject_index,
                    'draw_seed': draw_seed, 'physical_section_id': identity['physical_section_id'],
                    'valid_pixels': valid_pixels, 'used': used}, allow_nan=False) + '\n')
                if used:
                    break
                del old, new
            assert identity['split'] == 'train'
            assert identity['physical_section_id'] not in prior_ids
            prior_ids.add(identity['physical_section_id'])
            indices, _, target = sites_from_sample(old, 512)
            assert len(indices) == 512
            panel.append({'inputs': old['inputs'][0].cpu(), 'state': new['state'][0].cpu(),
                'reflection': old['reflection'][0].cpu(), 'offsets': old['offsets'][0].cpu(),
                'weights': old['weights'][0].cpu(), 'site_indices': indices.cpu(),
                'site_target': target.cpu(), 'base_index': base, 'slot': slot,
                'subject_index': subject_index, 'draw_seed': draw_seed,
                'physical_section_id': identity['physical_section_id'],
                'input_sha256': tensor_digest(old['inputs']),
                'target_sha256': tensor_digest(torch.cat((indices[:, None].float(), target), -1)),
                'visible_fraction': float((old['visible'][0] > .25).float().mean()),
                'event_flags': {event: bool(identity['one_shot_slide_artifacts_v3']['events'][event])
                                for event in EVENTS}, 'provenance': new['provenance'][0]})
            del old, new
assert len(panel) == 64
torch.save(panel, RUN / 'inner_panel.pt')
print(json.dumps({'event': 'panel_ready', 'sections': 64, 'attempts': attempts}), flush=True)

config = {'candidate': 287, 'seed': SEED, 'protocol_sha256': digest(PROTOCOL),
    'source_sha256': {name: digest(SOURCE / name) for name in SOURCE_NAMES},
    'parent_275_checkpoint_sha256': digest(parent_path),
    'parent_286_completed_sha256': digest(CONTROL / 'completed.json'),
    'parent_286_schedule_sha256': digest(CONTROL / 'training_schedule.jsonl'),
    'parent_286_control_checkpoint_sha256': digest(control_checkpoint),
    'parent_286_initial_checkpoint_sha256': digest(initial_checkpoint),
    'synthetic_provenance': context['provenance'], 'arms': ARMS,
    'frozen_control': '286 postartifact_gauge_v1 step 3000',
    'updates_per_arm': UPDATES, 'readout_steps': STEPS,
    'inner_panel_sha256': digest(RUN / 'inner_panel.pt'),
    'inner_draws_sha256': digest(RUN / 'inner_draws.jsonl'),
    'train_base_indices': list(range(56)), 'inner_base_indices': list(range(56, 64)),
    'sites_per_section': 512, 'map_derivative_limit': LIMIT,
    'source_lr': 2e-5, 'map_lr': 1e-4, 'weight_decay': 1e-4, 'clip': 2.,
    'feature_side': 64, 'map_control_side': 16,
    'no_external_weights_features_pseudolabels': True,
    'dev_cal_final_or_public_benchmark_used': False,
    'result_role': 'TRAIN-inner map-representation mechanism pilot; no blind-pose or GUI claim',
    'torch': torch.__version__, 'numpy': np.__version__}
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

control_model = make_model(pose_weights, pose_buffers, 16)
for name in MAP_MODULES:
    prefix = name + '.'
    getattr(control_model, name).load_state_dict({key[len(prefix):]: value
        for key, value in control_weights.items() if key.startswith(prefix)}, strict=True)
readouts, checkpoint_hashes, initial_common_hashes = {}, {}, {}
with (RUN / 'readouts.jsonl').open('w', encoding='utf8') as output, \
     (RUN / 'training.jsonl').open('w', encoding='utf8') as training:
    readouts['frozen_286_atlas'] = {'3000': readout(
        control_model, context['atlas'], panel, 'frozen_286_atlas', 3000, output)}
    print(json.dumps({'event': 'frozen_control',
        'gain_mm': readouts['frozen_286_atlas']['3000']['map_gain_mean_mm']}), flush=True)
    del control_model
    torch.cuda.empty_cache()
    for arm in ARMS:
        model = new_model(pose_weights, pose_buffers, initial_map_weights,
                          arm == 'trainable_atlas')
        common = {name: tensor_digest(value) for name, value in model.state_dict().items()
                  if name.split('.')[0] in ('encoder', 'lateral', 'warp_shared',
                                             'warp_condition', 'warp')}
        initial_common_hashes[arm] = common
        assert common == initial_common_hashes[ARMS[0]]
        source_parameters = list(model.encoder.parameters()) + list(model.lateral.parameters())
        map_names = MAP_MODULES if arm == 'trainable_atlas' else MAP_MODULES[:3]
        map_parameters = [parameter for name in map_names
                          for parameter in getattr(model, name).parameters()]
        parameters = source_parameters + map_parameters
        optimizer = torch.optim.AdamW([
            {'params': source_parameters, 'lr': 2e-5},
            {'params': map_parameters, 'lr': 1e-4}], weight_decay=1e-4)
        readouts[arm], checkpoint_hashes[arm] = {}, {}
        for step in range(UPDATES + 1):
            if step in STEPS:
                readouts[arm][str(step)] = readout(
                    model, context['atlas'], panel, arm, step, output)
                output.flush()
                path = RUN / f'{arm}_step_{step:05d}.pt'
                temporary = path.with_suffix('.tmp')
                torch.save({'candidate': 287, 'arm': arm, 'step': step,
                    'model': {name: value.cpu() for name, value in model.state_dict().items()
                              if name.split('.')[0] in ('encoder', 'lateral') + map_names},
                    'optimizer': optimizer.state_dict(),
                    'config_sha256': digest(RUN / 'config.json'),
                    'parent_275_checkpoint_sha256': config['parent_275_checkpoint_sha256'],
                    'calibrated': False}, temporary)
                os.replace(temporary, path)
                checkpoint_hashes[arm][str(step)] = digest(path)
                report = readouts[arm][str(step)]
                print(json.dumps({'event': 'readout', 'arm': arm, 'step': step,
                    'map_gain_mm': report['map_gain_mean_mm'],
                    'mapped_mm': report['mapped_mean_mm'],
                    'pose_first_choice_mm': report['pose_first_choice_mean_mm']}), flush=True)
            if step == UPDATES:
                break
            entry = schedule[step]
            with torch.no_grad():
                old, new = pair(context, entry['subject_index'], entry['draw_seed'])
            assert old['provenance'][0]['physical_section_id'] == entry['physical_section_id']
            indices, _, target = sites_from_sample(old, 512)
            assert len(indices) == 512
            assert tensor_digest(old['inputs']) == entry['input_sha256']
            assert tensor_digest(torch.cat((indices[:, None].float(), target), -1)) == \
                entry['target_sha256']
            assert tensor_digest(new['state']) == entry['postartifact_state_sha256']
            optimizer.zero_grad(set_to_none=True)
            mapped, points = exact_map(model, old['inputs'], new['state'],
                old['reflection'], old['offsets'], old['weights'], context['atlas'], indices)
            loss = (points - target).norm(dim=-1).mean() / 1000
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 2., error_if_nonfinite=True)
            optimizer.step()
            training.write(json.dumps({'arm': arm, 'step': step + 1,
                'physical_section_id': entry['physical_section_id'],
                'input_sha256': entry['input_sha256'], 'target_sha256': entry['target_sha256'],
                'loss_mm_before_update': float(loss.detach()),
                'gradient_norm': float(gradient),
                'elapsed_seconds': time.perf_counter() - started}, allow_nan=False) + '\n')
            if (step + 1) % 750 == 0:
                training.flush()
                print(json.dumps({'event': 'train_milestone', 'arm': arm,
                    'step': step + 1, 'loss_mm': float(loss.detach())}), flush=True)
            del old, new, mapped, points, loss
        del model, optimizer
        torch.cuda.empty_cache()

base = readouts['frozen_286_atlas']['3000']
gate = {}
for arm in ARMS:
    row = readouts[arm]['3000']
    initial = readouts[arm]['0']
    gate[arm] = {'gain_at_least_0_02mm': row['map_gain_mean_mm'] >= .02,
        'advantage_over_frozen_at_least_0_01mm':
            row['map_gain_mean_mm'] - base['map_gain_mean_mm'] >= .01,
        'at_least_six_nonnegative_bases': row['nonnegative_gain_bases'] >= 6,
        'derivative_at_most_0_35': row['max_postlimit_derivative'] <= LIMIT + 1e-5,
        'pose_regression_at_most_0_20mm':
            row['pose_first_choice_mean_mm'] - initial['pose_first_choice_mean_mm'] <= .20}
advance = {arm: all(gate[arm].values()) for arm in ARMS}
gap = readouts[ARMS[0]]['3000']['map_gain_mean_mm'] - \
    readouts[ARMS[1]]['3000']['map_gain_mean_mm']
preference = ('atlas' if gap >= .005 else 'direct' if gap <= -.005 else 'unresolved')
names = ('config.json', 'inner_draws.jsonl', 'inner_panel.pt',
         'training.jsonl', 'readouts.jsonl')
completed = {'candidate': 287, 'parent': 286, 'seed': SEED,
    'training_sections_per_arm': UPDATES, 'panel_sections': len(panel),
    'panel_attempts': attempts, 'arms': ARMS, 'readout_steps': STEPS,
    'readouts': readouts, 'gate': gate, 'advance': advance,
    'atlas_vs_direct_gain_gap_mm': gap, 'conditioning_preference': preference,
    'elapsed_seconds': time.perf_counter() - started,
    'source_sha256': config['source_sha256'],
    'initial_common_weight_sha256': initial_common_hashes,
    'files_sha256': {name: digest(RUN / name) for name in names},
    'checkpoint_sha256': checkpoint_hashes,
    'result_role': config['result_role'], 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'advance': advance,
    'conditioning_preference': preference,
    'gain_mm': {arm: readouts[arm]['3000']['map_gain_mean_mm'] for arm in ARMS},
    'frozen_gain_mm': base['map_gain_mean_mm']}), flush=True)
