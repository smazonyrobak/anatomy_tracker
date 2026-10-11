"""Fixed post-213 candidate-capture readout on the 128 inner-held native virtual planes."""

import hashlib
import json
import math
import os
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.sample_allen_stpt_native_211 import (
    load_allen_stpt_native_211, sample_allen_stpt_native_211)


source = Path(__file__).resolve()
repo = source.parents[1]
acquired = root / 'data/allen_stpt_native_208_train_001'
maps = root / 'data/allen_stpt_native_maps_210_train_001'
train = root / 'runs/allen_native_matched_212_001'
readout = root / 'runs/allen_native_stpt_213_inner_held_001'
out = root / 'runs/allen_native_capture_215_001'
protocol = repo / 'docs/publication/ALLEN_NATIVE_STPT_212_MATCHED_PROTOCOL_20261011.md'
donors, arms, steps = (14599, 14600), ('control', 'native'), (0, 400)
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()
assert torch.cuda.is_available()

readout_done = json.loads((readout / 'completed.json').read_text())
readout_config = json.loads((readout / 'config.json').read_text())
train_config = json.loads((train / 'config.json').read_text())
train_done = json.loads((train / 'completed.json').read_text())
map_done = json.loads((maps / 'completed.json').read_text())
acquisition_done = json.loads((acquired / 'completed.json').read_text())
assert readout_done['virtual_planes'] == 128
assert readout_config['virtual_accepted_per_donor'] == 64
assert tuple(train_config['native_inner_held_donors']) == donors
assert not set(donors) & set(train_config['native_gradient_donors'])
assert not readout_done['calibrated'] and not readout_done['official_dev_cal_final_used']
assert not readout_done['public_benchmark_used']
assert not train_done['development_calibration_final_read']
assert train_config['native_map_completion_sha256'] == readout_config['map_completion_sha256']

paths = {
    'protocol': protocol,
    'source_208': repo / 'training/acquire_allen_stpt_native_208.py',
    'source_210': repo / 'training/build_allen_stpt_native_maps_210.py',
    'source_211': repo / 'training/sample_allen_stpt_native_211.py',
    'source_212': repo / 'training/train_allen_native_matched_212.py',
    'source_213': repo / 'training/evaluate_allen_stpt_native_213.py',
    'acquisition_completion': acquired / 'completed.json',
    'map_completion': maps / 'completed.json',
    'training_config': train / 'config.json',
    'training_completion': train / 'completed.json',
    'readout_config': readout / 'config.json',
    'readout_completion': readout / 'completed.json',
    'readout_virtual_rows': readout / 'virtual_rows.jsonl',
    'readout_virtual_draws': readout / 'virtual_draws.jsonl',
    'parent_checkpoint': root / 'runs/lupori_fluorescence_pose_187_inner_AL1A/joint_step_00800.pt',
}
for arm in arms:
    paths[f'{arm}_completion'] = train / arm / 'completed.json'
    for step in steps:
        paths[f'{arm}_step{step}'] = train / arm / f'joint_step_{step:05d}.pt'
for donor in donors:
    paths[f'binding_{donor}'] = maps / f'donor_{donor}/binding.json'
    paths[f'acquisition_receipt_{donor}'] = acquired / f'donor_{donor}/receipt.json'
input_sha256 = {}
for name, path in paths.items():
    with path.open('rb') as stream:
        input_sha256[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
assert input_sha256['protocol'] == train_config['protocol_sha256'] == readout_config['protocol_sha256']
assert input_sha256['source_208'] == acquisition_done['acquisition_source_sha256']
assert input_sha256['source_210'] == map_done['source_sha256']
assert input_sha256['source_211'] == train_config['source_sha256']['sample_allen_stpt_native_211.py']
assert input_sha256['source_212'] == train_config['source_sha256']['train_allen_native_matched_212.py']
assert input_sha256['source_213'] == readout_config['source_sha256']
assert input_sha256['acquisition_completion'] == readout_config['acquisition_completion_sha256'] == map_done['acquisition_completion_sha256']
assert input_sha256['map_completion'] == readout_config['map_completion_sha256']
assert input_sha256['training_config'] == readout_config['training_config_sha256']
assert input_sha256['parent_checkpoint'] == train_config['parent_checkpoint_sha256'] == readout_config['parent_checkpoint_sha256']
assert input_sha256['readout_config'] == readout_done['output_sha256']['config.json']
assert input_sha256['readout_virtual_rows'] == readout_done['output_sha256']['virtual_rows.jsonl']
assert input_sha256['readout_virtual_draws'] == readout_done['output_sha256']['virtual_draws.jsonl']

bindings = {}
for donor in donors:
    binding = json.loads(paths[f'binding_{donor}'].read_text())
    assert binding['donor_id'] == donor and binding['split'] == 'train'
    assert binding['weak_alignment_not_anatomical_truth']
    assert input_sha256[f'binding_{donor}'] == map_done['binding_sha256'][f'donor_{donor}/binding.json'] == readout_config['held_binding_sha256'][str(donor)]
    assert input_sha256[f'acquisition_receipt_{donor}'] == acquisition_done['donor_receipt_sha256'][f'donor_{donor}/receipt.json']
    assert binding['acquisition_completion_sha256'] == input_sha256['acquisition_completion']
    for name, key in (('native_geometry_path', 'native_geometry_sha256'),
                      ('native_stack_path', 'native_stack_sha256'), ('map_path', 'map_sha256')):
        with Path(binding[name]).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == binding[key]
        input_sha256[f'{donor}_{key}'] = digest
        paths[f'{donor}_{key}'] = Path(binding[name])
    bindings[donor] = binding
for arm in arms:
    done = json.loads(paths[f'{arm}_completion'].read_text())
    assert done['arm'] == arm and done['updates'] == 400
    assert not done['inner_held_donors_used_in_training']
    assert input_sha256[f'{arm}_completion'] == readout_config['training_completion_sha256'][arm] == train_done[f'{arm}_completion_sha256']
    assert done['config_sha256'] == input_sha256['training_config']
    for step in steps:
        assert input_sha256[f'{arm}_step{step}'] == done['checkpoint_sha256'][str(step)] == readout_config['checkpoint_sha256'][arm][str(step)]

source_rows = [json.loads(line) for line in paths['readout_virtual_rows'].open()]
draws = [json.loads(line) for line in paths['readout_virtual_draws'].open() if '"used": true' in line]
rows_by_id = {}
for row in source_rows:
    rows_by_id.setdefault(row['sample_id'], []).append(row)
assert len(source_rows) == 128 * 2 * 3 and len(rows_by_id) == len(draws) == 128
assert {str(d['donor_id']) + ':virtual:' + str(d['accepted_index']) for d in draws} == set(rows_by_id)
draw_by_id = {str(d['donor_id']) + ':virtual:' + str(d['accepted_index']): d for d in draws}
for sample_id, group in rows_by_id.items():
    assert {(r['arm'], r['step']) for r in group} == {(a, s) for a in arms for s in (0, 200, 400)}
    assert len({r['seed'] for r in group}) == 1
    assert group[0]['seed'] == draw_by_id[sample_id]['seed']
    assert group[0]['native_ouv_um'] == draw_by_id[sample_id]['native_ouv_um']
    assert group[0]['native_unit_normal'] == draw_by_id[sample_id]['native_unit_normal']

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device, side = 'cuda', 256
five_chart = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                           [127.5, 127.5]], device=device) / side
axis16 = (torch.arange(16, device=device, dtype=torch.float32) + .5) / 16 - .5 / side
y16, x16 = torch.meshgrid(axis16, axis16, indexing='ij')
tissue_chart = torch.stack((x16, y16), -1).reshape(-1, 2)
panel = []
with torch.inference_mode():
    for donor in donors:
        binding = bindings[donor]
        context = load_allen_stpt_native_211(acquired / f'donor_{donor}',
            binding['map_path'], device=device, inner_role='inner_held')
        for index in range(64):
            sample_id = f'{donor}:virtual:{index}'
            record = rows_by_id[sample_id][0]
            draw = draw_by_id[sample_id]
            sample = sample_allen_stpt_native_211(context, [donor], record['seed'], side=side)
            provenance = sample['provenance'][0]
            visible = sample['visible'][0]
            valid96 = F.interpolate(visible[None, None], (96, 96), mode='area')[0, 0]
            assert bool(sample['eligible'][0]) and bool((valid96 > 0).any())
            assert provenance == {key: draw[key] for key in provenance}
            assert int(sample['reflection'][0]) == int(draw['horizontal_reflection'])
            assert np.isclose(float(visible.mean()), record['tissue_fraction'], atol=1e-6)
            assert int((valid96 > 0).sum()) == record['valid_tissue_pixels']
            assert np.isclose(float(valid96.sum()), record['valid_tissue_mass_96'], atol=1e-3)
            panel.append({'record': record, 'inputs': sample['inputs'][0].cpu().half(),
                'state': sample['state'][0].cpu(), 'reflection': int(sample['reflection'][0]),
                'support16': F.interpolate(visible[None, None], (16, 16), mode='area')[0, 0].cpu()})
        del context, sample
        torch.cuda.empty_cache()
        print(json.dumps({'event': 'reconstructed', 'donor_id': donor, 'count': 64}), flush=True)

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).to(device).eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
branch_id = torch.arange(160, device=device)
reflection = branch_id % 2
rows = []
with torch.inference_mode():
    for arm, step in (('control', 0), ('control', 400), ('native', 400)):
        saved = torch.load(paths[f'{arm}_step{step}'], map_location='cpu', weights_only=False)
        assert saved['step'] == step and saved['arm'] == arm and saved['config'] == train_config
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for item in panel:
            source_row = next(r for r in rows_by_id[item['record']['sample_id']]
                              if r['arm'] == arm and r['step'] == step)
            prediction = model.predict(item['inputs'][None].to(device, torch.float32))
            assert prediction['state'].shape == (1, 80, 12)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            states = prediction['state'][0].repeat_interleave(2, 0)
            truth_state = item['state'][None].to(device)
            truth_reflection = torch.tensor([item['reflection']], device=device)
            proposed5 = rigid_points_090(states, reflection, five_chart)
            reference5 = rigid_points_090(truth_state, truth_reflection, five_chart)[0]
            error5 = (proposed5 - reference5).norm(dim=-1).mean(-1) / 1000
            proposed_tissue = rigid_points_090(states, reflection, tissue_chart)
            reference_tissue = rigid_points_090(truth_state, truth_reflection, tissue_chart)[0]
            support = item['support16'].to(device).flatten()
            error_tissue = ((proposed_tissue - reference_tissue).norm(dim=-1) * support[None]).sum(-1) / support.sum() / 1000
            normal = full_frame_state_to_components(prediction['state'][0])[1][..., 2]
            truth_normal = full_frame_state_to_components(truth_state)[1][0, :, 2]
            angle = torch.acos((normal * truth_normal).sum(-1).abs().clamp(0, 1)) * 180 / math.pi
            angle = angle.repeat_interleave(2)
            top = int(prior.argmax())
            best5 = int(error5.argmin())
            best_tissue = int(error_tissue.argmin())
            prior_rank = torch.argsort(torch.argsort(-prior, stable=True), stable=True) + 1
            mass = prior.exp()
            assert top == source_row['top1_branch_id']
            assert np.isclose(float(error5[top]), source_row['top1_weak_affine_five_point_mm'], atol=1e-4)
            rows.append({'sample_id': item['record']['sample_id'],
                'donor_id': item['record']['donor_id'], 'seed': item['record']['seed'],
                'virtual_index': item['record']['virtual_index'],
                'tissue_fraction': item['record']['tissue_fraction'],
                'tissue_support_mass_16': float(support.sum()),
                'arm': arm, 'step': step,
                'checkpoint_sha256': input_sha256[f'{arm}_step{step}'],
                'top1_branch_id': top, 'best_five_branch_id': best5,
                'best_tissue_branch_id': best_tissue,
                'top1_five_mm': float(error5[top]), 'best_five_mm': float(error5[best5]),
                'top1_tissue_mm': float(error_tissue[top]),
                'best_tissue_mm': float(error_tissue[best_tissue]),
                'five_best_tissue_mm': float(error_tissue[best5]),
                'tissue_best_five_mm': float(error5[best_tissue]),
                'top1_normal_axis_deg': float(angle[top]),
                'five_best_normal_axis_deg': float(angle[best5]),
                'tissue_best_normal_axis_deg': float(angle[best_tissue]),
                'top1_prior_mass': float(mass[top]),
                'five_best_prior_mass': float(mass[best5]),
                'five_best_prior_rank': int(prior_rank[best5]),
                'tissue_best_prior_mass': float(mass[best_tissue]),
                'tissue_best_prior_rank': int(prior_rank[best_tissue]),
                'all_160_five_mm': error5.cpu().tolist(),
                'all_160_tissue_mm': error_tissue.cpu().tolist(),
                'all_160_prior_mass': mass.cpu().tolist(),
                'all_160_normal_axis_deg': angle.cpu().tolist()})
        print(json.dumps({'event': 'checkpoint_scored', 'arm': arm, 'step': step,
                          'virtual_planes': len(panel)}), flush=True)

summary = {'metric': 'rigid candidate versus reconstructed weak-affine truth; five-point full canvas as 213, plus fixed 16x16 area-pooled visible-support-weighted chart',
    'normal_axis': 'acute unoriented normal-axis angle, degrees',
    'rank': '1-based descending normalized mode/reflection prior mass; no reranking or fitting',
    'scope': '128 virtual obliques from two inner-held TRAIN donors; not physical-angle validation',
    'groups': {}, 'calibrated': False, 'official_dev_cal_final_used': False,
    'public_benchmark_used': False}
for arm, step in (('control', 0), ('control', 400), ('native', 400)):
    for donor in (None, *donors):
        for label, lower, upper in (('all', 0., 1.), ('0-0.1', 0., .1),
                                    ('0.1-0.3', .1, .3), ('0.3-0.6', .3, .6),
                                    ('0.6-1', .6, 1.)):
            selected = [r for r in rows if r['arm'] == arm and r['step'] == step
                and (donor is None or r['donor_id'] == donor)
                and (label == 'all' or lower <= r['tissue_fraction'] < upper)]
            result = {'count': len(selected)}
            for key in ('top1_five_mm', 'best_five_mm', 'top1_tissue_mm', 'best_tissue_mm',
                        'top1_normal_axis_deg', 'five_best_normal_axis_deg',
                        'five_best_prior_rank', 'five_best_prior_mass'):
                values = np.array([r[key] for r in selected])
                result[key] = {'p50': float(np.median(values)),
                               'p90': float(np.quantile(values, .9))} if len(values) else None
            for key in ('top1_five_mm', 'best_five_mm', 'top1_tissue_mm', 'best_tissue_mm'):
                result[key + '_le_1_2_5_count'] = [sum(r[key] <= bound for r in selected)
                                                    for bound in (1, 2, 5)]
            result['five_best_prior_rank_le_5_10_20_count'] = [sum(
                r['five_best_prior_rank'] <= bound for r in selected) for bound in (5, 10, 20)]
            summary['groups'][f'{arm}:step{step}:{donor if donor is not None else "both"}:{label}'] = result

config = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'input_path_sha256': {name: {'path': str(paths[name]), 'sha256': digest}
                          for name, digest in input_sha256.items()},
    'source_213_output_sha256': readout_done['output_sha256'],
    'source_212_sha256': train_config['source_sha256'],
    'donor_ids': donors, 'sample_count': len(panel), 'checkpoints': [('control', 0),
        ('control', 400), ('native', 400)], 'branch_order': '2*mode+reflection; 80 modes, 2 reflections',
    'five_chart_xy': five_chart.cpu().tolist(), 'tissue_chart': '16x16 area-pool support on original 256x256 visible image; coordinate centers (index+0.5)/16 - 0.5/256',
    'tissue_metric': 'support-weighted mean rigid CCF point distance to reconstructed affine state, mm',
    'normal_axis': 'acos(abs(dot(predicted normal, truth normal))) degrees',
    'calibrated': False, 'official_dev_cal_final_used': False,
    'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
with (out / 'rows.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
(out / 'completed.json').write_text(json.dumps({'virtual_planes': len(panel),
    'checkpoint_reads': len(rows), 'output_sha256': {name: hashlib.sha256(
        (out / name).read_bytes()).hexdigest() for name in ('config.json', 'summary.json', 'rows.jsonl')},
    'calibrated': False, 'official_dev_cal_final_used': False,
    'public_benchmark_used': False}, indent=2, allow_nan=False))
print(json.dumps({'event': 'capture_diagnostic_complete', 'output': str(out)}), flush=True)
