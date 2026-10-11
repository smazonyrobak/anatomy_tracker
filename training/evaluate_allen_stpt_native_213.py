"""Post-training inner-held native STPT readout for matched 212 arms."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = os.environ['TMPDIR'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_from_components
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.sample_allen_stpt_native_211 import (
    load_allen_stpt_native_211, sample_allen_stpt_native_211)


source = Path(__file__).resolve()
repo = source.parents[1]
protocol = repo / 'docs/publication/ALLEN_NATIVE_STPT_212_MATCHED_PROTOCOL_20261011.md'
acquired = root / 'data/allen_stpt_native_208_train_001'
maps = root / 'data/allen_stpt_native_maps_210_train_001'
train = root / 'runs/allen_native_matched_212_001'
parent = root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt'
out = root / 'runs/allen_native_stpt_213_inner_held_001'
donors, arms, steps, side, map_side = (14599, 14600), ('control', 'native'), (0, 200, 400), 256, 96
virtual_seed_first, virtual_count = 213640000, 64
strata_edges = (0., .1, .3, .6, 1.)

assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
committed_protocol = subprocess.run(['git', '-C', str(repo), 'show',
    'HEAD:docs/publication/ALLEN_NATIVE_STPT_212_MATCHED_PROTOCOL_20261011.md'],
    capture_output=True, check=True).stdout
protocol_bytes = protocol.read_bytes()
assert committed_protocol == protocol_bytes.replace(b'\r\n', b'\n'), (
    '212 protocol must be committed unchanged before readout')
protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
assert (acquired / 'completed.json').is_file() and (maps / 'completed.json').is_file()
assert (train / 'completed.json').is_file()
assert all((train / arm / 'completed.json').is_file() for arm in arms)

acquisition_done = json.loads((acquired / 'completed.json').read_text())
map_done = json.loads((maps / 'completed.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
train_done = json.loads((train / 'completed.json').read_text())
assert all(train_done[f'{arm}_completion_sha256'] == hashlib.sha256(
    (train / arm / 'completed.json').read_bytes()).hexdigest() for arm in arms)
assert acquisition_done['donors'] == map_done['donors'] == 8
assert set(donors) <= set(acquisition_done['donor_ids']) == set(map_done['donor_ids'])
assert tuple(train_config['checkpoints']) == steps
assert train_config['map_side'] == map_side
assert train_config['protocol_sha256'] == protocol_sha
assert tuple(train_config['native_inner_held_donors']) == donors
assert not set(donors) & set(train_config['native_gradient_donors'])
assert train_config['native_map_completion_sha256'] == hashlib.sha256(
    (maps / 'completed.json').read_bytes()).hexdigest()
with parent.open('rb') as stream:
    parent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert train_config['parent_checkpoint_sha256'] == parent_sha

checkpoints, checkpoint_sha, bindings = {}, {}, {}
for arm in arms:
    done = json.loads((train / arm / 'completed.json').read_text())
    assert done['arm'] == arm and done['updates'] == 400
    assert not done['calibrated'] and not done['public_benchmark_used']
    assert not done['inner_held_donors_used_in_training']
    assert done['config_sha256'] == hashlib.sha256((train / 'config.json').read_bytes()).hexdigest()
    for step in steps:
        path = train / arm / f'joint_step_{step:05d}.pt'
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == done['checkpoint_sha256'][str(step)]
        checkpoints[arm, step], checkpoint_sha[arm, step] = path, digest

parent_saved = torch.load(parent, map_location='cpu', weights_only=False)
assert parent_saved['step'] == 800
for arm in arms:
    saved = torch.load(checkpoints[arm, 0], map_location='cpu', weights_only=False)
    assert saved['step'] == 0 and saved['arm'] == arm and saved['config'] == train_config
    assert parent_saved['model'].keys() == saved['model'].keys()
    assert all(torch.equal(value, saved['model'][name])
               for name, value in parent_saved['model'].items())
    del saved
del parent_saved

for donor in donors:
    binding_path = maps / f'donor_{donor}/binding.json'
    with binding_path.open('rb') as stream:
        binding_bytes = stream.read()
    assert hashlib.sha256(binding_bytes).hexdigest() == map_done['binding_sha256'][
        f'donor_{donor}/binding.json']
    binding = json.loads(binding_bytes)
    assert binding['donor_id'] == donor and binding['split'] == 'train'
    assert binding['shape_zyxc'][0] == len(binding['section_numbers']) == 140
    assert np.all(np.diff(binding['section_numbers']) == 1)
    assert binding['native_spacing_um_xyz'][2] == 100
    assert binding['weak_alignment_not_anatomical_truth']
    assert binding['acquisition_completion_sha256'] == hashlib.sha256(
        (acquired / 'completed.json').read_bytes()).hexdigest()
    with Path(binding['native_geometry_path']).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == binding['native_geometry_sha256']
    with Path(binding['native_stack_path']).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == binding['native_stack_sha256']
    with Path(binding['map_path']).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == binding['map_sha256']
    bindings[donor] = binding

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = 'cuda'
axis = torch.arange(side, device=device, dtype=torch.float32) / side
local_axis = (torch.arange(map_side, device=device, dtype=torch.float32) + .5
              ) / map_side - .5 / side
local_y, local_x = torch.meshgrid(local_axis, local_axis, indexing='ij')
local_chart = torch.stack((local_x, local_y), -1).reshape(-1, 2)
five_chart = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                           [127.5, 127.5]], device=device) / side
offsets = torch.linspace(-50., 50., 9, device=device)
weights = torch.tensor([1., 2., 2., 2., 2., 2., 2., 2., 1.], device=device) / 16
mode_zero = torch.zeros((1, 1), device=device, dtype=torch.long)
panel, virtual_draws = [], []

with torch.inference_mode():
    for donor in donors:
        binding = bindings[donor]
        stack = np.load(binding['native_stack_path'], mmap_mode='r')
        authored = np.load(binding['map_path'], mmap_mode='r')
        assert tuple(stack.shape) == tuple(binding['shape_zyxc'][:3])
        assert tuple(authored.shape) == tuple(binding['shape_zyxc'])
        for index, section_number in enumerate(binding['section_numbers']):
            native_red = np.asarray(stack[index])
            components, _ = label(native_red > 5)
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            tissue = (sizes >= 25)[components]
            image = F.interpolate(torch.from_numpy(np.array(native_red, copy=True))[None, None]
                .to(device, torch.float32), (side, side), mode='bilinear',
                align_corners=False)[0, 0] / 255
            weak = F.interpolate(torch.from_numpy(np.array(authored[index], copy=True))
                .permute(2, 0, 1)[None].to(device), (side, side), mode='bilinear',
                align_corners=False)[0].permute(1, 2, 0)
            support = F.interpolate(torch.from_numpy(tissue.astype(np.float32))[None, None]
                .to(device), (side, side), mode='bilinear', align_corners=False)[0, 0]
            tissue_fraction = float(support.mean())
            valid_weight = F.interpolate(support[None, None], (map_side, map_side),
                mode='area')[0, 0]
            weak_local = F.interpolate((weak * support[..., None]).permute(2, 0, 1)[None],
                (map_side, map_side), mode='area')[0].permute(1, 2, 0)
            weak_local = weak_local / valid_weight.clamp_min(1e-6)[..., None]
            centred = axis - axis.mean()
            edge_u = (weak * centred[None, :, None]).sum((0, 1)) / (
                side * centred.square().sum())
            edge_v = (weak * centred[:, None, None]).sum((0, 1)) / (
                side * centred.square().sum())
            origin = weak.mean((0, 1)) - axis.mean() * (edge_u + edge_v)
            state = full_frame_state_from_components(*physical_ouv_to_frame(
                torch.stack((origin, edge_u, edge_v))))
            rigid = rigid_points_090(state[None], torch.zeros(1, device=device,
                dtype=torch.long), local_chart)[0].reshape(map_side, map_side, 3)
            zero_sse = float(((rigid - weak_local).square().sum(-1) * valid_weight).sum())
            record = {'panel': 'physical_acquired_coronal',
                'sample_id': f'{donor}:section:{section_number}', 'donor_id': donor,
                'specimen_id': binding['specimen_id'], 'experiment_id': binding['experiment_id'],
                'section_number': section_number, 'native_section_index': index,
                'native_shape_hw': list(native_red.shape),
                'tissue_fraction': tissue_fraction,
                'valid_tissue_pixels': int((valid_weight > 0).sum()),
                'valid_tissue_mass_96': float(valid_weight.sum()),
                'label_role': 'Allen authored weak CCF registration',
                'source_kind': 'physically acquired original red section'}
            panel.append({'record': record, 'inputs': torch.stack((image, *[torch.zeros_like(image)
                for _ in range(4)])).cpu().half(), 'weak': weak_local.cpu(),
                'valid_weight': valid_weight.cpu(),
                'state': state.cpu(), 'reflection': 0, 'zero_sse_um2': zero_sse,
                'offsets': offsets.cpu(), 'weights': weights.cpu()})
        del stack, authored

    for donor in donors:
        binding = bindings[donor]
        context = load_allen_stpt_native_211(acquired / f'donor_{donor}',
            binding['map_path'], device=device, inner_role='inner_held')
        accepted, attempt = 0, 0
        while accepted < virtual_count:
            seed = virtual_seed_first + donor * 10000 + attempt
            sample = sample_allen_stpt_native_211(context, [donor], seed, side=side)
            provenance = sample['provenance'][0]
            visible = sample['visible'][0]
            valid_weight = F.interpolate(visible[None, None],
                (map_side, map_side), mode='area')[0, 0]
            use = bool(sample['eligible'][0]) and bool((valid_weight > 0).any())
            virtual_draws.append({'donor_id': donor, 'attempt': attempt,
                'accepted_index': accepted if use else None, 'used': use, **provenance})
            attempt += 1
            if not use:
                continue
            weak = F.interpolate((sample['centre'][0] * visible[..., None])
                .permute(2, 0, 1)[None], (map_side, map_side),
                mode='area')[0].permute(1, 2, 0)
            weak = weak / valid_weight.clamp_min(1e-6)[..., None]
            state = sample['state'][0]
            reflection = int(sample['reflection'][0])
            rigid = rigid_points_090(state[None], torch.tensor([reflection],
                device=device), local_chart)[0].reshape(map_side, map_side, 3)
            zero_sse = float(((rigid - weak).square().sum(-1) * valid_weight).sum())
            record = {'panel': 'virtual_oblique_reconstructed',
                'sample_id': f'{donor}:virtual:{accepted}', 'donor_id': donor,
                'specimen_id': binding['specimen_id'], 'experiment_id': binding['experiment_id'],
                'virtual_index': accepted, 'seed': seed,
                'native_ouv_um': provenance['native_ouv_um'],
                'native_unit_normal': provenance['native_unit_normal'],
                'virtual_thickness_um': provenance['virtual_thickness_um'],
                'source_section_numbers': provenance['source_section_numbers'],
                'tissue_fraction': float(visible.mean()),
                'valid_tissue_pixels': int((valid_weight > 0).sum()),
                'valid_tissue_mass_96': float(valid_weight.sum()),
                'label_role': 'Allen authored weak CCF map interpolated from acquired coronal sections',
                'source_kind': 'reconstructed virtual oblique, not physical-angle validation'}
            panel.append({'record': record, 'inputs': sample['inputs'][0].cpu().half(),
                'weak': weak.cpu(), 'valid_weight': valid_weight.cpu(), 'state': state.cpu(),
                'reflection': reflection, 'zero_sse_um2': zero_sse,
                'offsets': sample['offsets'][0].cpu(),
                'weights': sample['weights'][0].cpu()})
            accepted += 1
        del context, sample
        torch.cuda.empty_cache()

assert sum(item['record']['panel'] == 'physical_acquired_coronal' for item in panel) == 280
assert sum(item['record']['panel'] == 'virtual_oblique_reconstructed' for item in panel) == 128

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).to(device)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).to(device).eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
rows = []
with torch.inference_mode():
    for arm in arms:
        for step in steps:
            saved = torch.load(checkpoints[arm, step], map_location='cpu', weights_only=False)
            assert saved['step'] == step and saved['arm'] == arm and saved['config'] == train_config
            model.load_state_dict(saved['model'], strict=True)
            del saved
            for item in panel:
                inputs = item['inputs'][None].to(device, torch.float32)
                state = item['state'][None].to(device)
                reflection = torch.tensor([[item['reflection']]], device=device)
                weak = item['weak'].to(device)
                valid_weight = item['valid_weight'].to(device)
                prediction = model.predict(inputs)
                prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
                top_id = int(prior[0].argmax())
                proposed = rigid_points_090(prediction['state'][:, top_id // 2],
                    torch.tensor([top_id % 2], device=device), five_chart)[0]
                reference = rigid_points_090(state, reflection[:, 0], five_chart)[0]
                pose_mm = float((proposed - reference).norm(dim=-1).mean() / 1000)
                supplied = {**prediction, 'state': prediction['state'].clone()}
                supplied['state'][0, 0] = state[0]
                mapped = model.map(supplied, item['offsets'].to(device), mode_zero, reflection,
                    (map_side, map_side), atlas, item['weights'].to(device), feature_side=map_side,
                    source_shape=(side, side))['centre_surface_ccf_ap_dv_ml_um'][0, 0]
                map_sse = float(((mapped - weak).square().sum(-1) * valid_weight).sum())
                mass = item['record']['valid_tissue_mass_96']
                rows.append({**item['record'], 'arm': arm, 'step': step,
                    'checkpoint_sha256': checkpoint_sha[arm, step],
                    'top1_branch_id': top_id, 'top1_weak_affine_five_point_mm': pose_mm,
                    'zero_local_sse_um2': item['zero_sse_um2'],
                    'learned_local_sse_um2': map_sse,
                    'zero_local_rms_mm': (item['zero_sse_um2'] / mass) ** .5 / 1000
                        if mass else None,
                    'learned_local_rms_mm': (map_sse / mass) ** .5 / 1000
                        if mass else None})
            print(json.dumps({'event': 'checkpoint_read', 'arm': arm, 'step': step,
                'physical_sections': 280, 'virtual_planes': 128}), flush=True)

parent_sse = {(row['panel'], row['sample_id']): row['learned_local_sse_um2']
    for row in rows if row['arm'] == 'control' and row['step'] == 0}
for row in rows:
    row['parent_local_sse_um2'] = parent_sse[row['panel'], row['sample_id']]
    row['parent_local_rms_mm'] = (row['parent_local_sse_um2'] /
        row['valid_tissue_mass_96']) ** .5 / 1000 if row['valid_tissue_mass_96'] else None
assert all(np.isclose(row['learned_local_sse_um2'], row['parent_local_sse_um2'])
           for row in rows if row['step'] == 0)

summary = {'scope': 'two inner held TRAIN donors; descriptive gate only',
    'primary': '280 physically acquired coronal sections at native XY canvas resized once to 256',
    'secondary': '128 independently seeded reconstructed virtual obliques; not physical-angle validation',
    'pose': 'prior top-1 mode/reflection five-point CCF disagreement with full-canvas weak Allen affine',
    'local': 'teacher-forced correct affine state; tissue-mass-weighted CCF-coordinate RMS at the trained 96x96 map grid with area-pooled visible tissue and authored coordinates',
    'physical': {}, 'virtual': {}, 'native_component_gate_step400': {},
    'full_advancement_gate': 'pending independent synthetic/AL1A/old-real retention readout',
    'calibrated': False, 'official_dev_cal_final_used': False,
    'public_benchmark_used': False}
for donor in donors:
    for panel_name, summary_name in (('physical_acquired_coronal', 'physical'),
                                     ('virtual_oblique_reconstructed', 'virtual')):
        donor_summary = summary[summary_name].setdefault(str(donor), {})
        for step in steps:
            step_summary = donor_summary.setdefault(str(step), {})
            for arm in arms:
                selected = [row for row in rows if row['panel'] == panel_name
                    and row['donor_id'] == donor and row['step'] == step and row['arm'] == arm]
                pose = np.array([row['top1_weak_affine_five_point_mm'] for row in selected])
                mass = sum(row['valid_tissue_mass_96'] for row in selected)
                item = {'count': len(selected), 'top1_p50_mm': float(np.median(pose)),
                    'top1_p90_mm': float(np.quantile(pose, .9)),
                    'top1_gt5mm_count': int((pose > 5).sum()),
                    'learned_local_rms_mm': (sum(row['learned_local_sse_um2']
                        for row in selected) / mass) ** .5 / 1000,
                    'zero_local_rms_mm': (sum(row['zero_local_sse_um2']
                        for row in selected) / mass) ** .5 / 1000,
                    'parent_local_rms_mm': (sum(row['parent_local_sse_um2']
                        for row in selected) / mass) ** .5 / 1000}
                if summary_name == 'physical':
                    item['tissue_fraction_strata'] = {}
                    for index, (lower, upper) in enumerate(zip(strata_edges[:-1], strata_edges[1:])):
                        band = [row['top1_weak_affine_five_point_mm'] for row in selected
                            if lower <= row['tissue_fraction'] and
                            (row['tissue_fraction'] < upper if index < 3
                             else row['tissue_fraction'] <= upper)]
                        item['tissue_fraction_strata'][f'{lower:g}-{upper:g}'] = {
                            'count': len(band),
                            'top1_p50_mm': float(np.median(band)) if band else None,
                            'top1_p90_mm': float(np.quantile(band, .9)) if band else None,
                            'top1_gt5mm_count': int((np.array(band) > 5).sum())}
                step_summary[arm] = item

gate = summary['native_component_gate_step400']
for donor in donors:
    control = summary['physical'][str(donor)]['400']['control']
    native = summary['physical'][str(donor)]['400']['native']
    required = max(.5, .1 * control['top1_p50_mm'])
    gate[str(donor)] = {'required_p50_improvement_mm': required,
        'observed_p50_improvement_mm': control['top1_p50_mm'] - native['top1_p50_mm'],
        'p50_improves_enough': native['top1_p50_mm'] <= control['top1_p50_mm'] - required,
        'p90_regression_le_0p25mm': native['top1_p90_mm'] <= control['top1_p90_mm'] + .25,
        'local_beats_zero': native['learned_local_rms_mm'] < native['zero_local_rms_mm'],
        'local_beats_unchanged_parent': native['learned_local_rms_mm'] < native['parent_local_rms_mm']}
gate['passes_native_component_only'] = all(all(value for key, value in gate[str(donor)].items()
    if key not in ('required_p50_improvement_mm', 'observed_p50_improvement_mm')) for donor in donors)

config = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'protocol_sha256': protocol_sha, 'acquisition_completion_sha256': hashlib.sha256(
        (acquired / 'completed.json').read_bytes()).hexdigest(),
    'map_completion_sha256': hashlib.sha256((maps / 'completed.json').read_bytes()).hexdigest(),
    'training_config_sha256': hashlib.sha256((train / 'config.json').read_bytes()).hexdigest(),
    'training_completion_sha256': {arm: hashlib.sha256((train / arm / 'completed.json')
        .read_bytes()).hexdigest() for arm in arms},
    'checkpoint_sha256': {arm: {str(step): checkpoint_sha[arm, step] for step in steps}
        for arm in arms}, 'parent_checkpoint_sha256': parent_sha,
    'held_binding_sha256': {str(donor): map_done['binding_sha256'][f'donor_{donor}/binding.json']
        for donor in donors}, 'physical_sections_per_donor': 140,
    'native_geometry': {str(donor): {'spacing_um_xyz': bindings[donor]['native_spacing_um_xyz'],
        'section_number_start_end': [bindings[donor]['section_numbers'][0],
                                     bindings[donor]['section_numbers'][-1]],
        'shape_zyx': bindings[donor]['shape_zyxc'][:3]} for donor in donors},
    'virtual_accepted_per_donor': virtual_count, 'virtual_seed_first': virtual_seed_first,
    'virtual_seed_rule': '213640000 + donor_id * 10000 + zero_based_attempt; first 64 eligible per donor',
    'tissue_fraction_strata': strata_edges, 'model_side': side, 'map_side': map_side,
    'axial_offsets_um': offsets.cpu().tolist(), 'axial_weights': weights.cpu().tolist(),
    'no_checkpoint_selection': True, 'calibrated': False,
    'official_dev_cal_final_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
for name, selected in (('physical_rows.jsonl', [row for row in rows
                        if row['panel'] == 'physical_acquired_coronal']),
                       ('virtual_rows.jsonl', [row for row in rows
                        if row['panel'] == 'virtual_oblique_reconstructed']),
                       ('virtual_draws.jsonl', virtual_draws)):
    with (out / name).open('w') as stream:
        for row in selected:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({'physical_sections': 280,
    'virtual_planes': 128, 'steps': steps, 'arms': arms,
    'output_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ('config.json', 'summary.json', 'physical_rows.jsonl',
                     'virtual_rows.jsonl', 'virtual_draws.jsonl')},
    'calibrated': False, 'official_dev_cal_final_used': False,
    'public_benchmark_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'readout_complete', 'native_component_gate_step400': gate}), flush=True)
