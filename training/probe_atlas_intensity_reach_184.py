"""TRAIN-only intact-versus-zero atlas mapping at known and blind planes."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'cache/tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v3 import sample_one_shot_slide_artifacts_v3
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_166 import sample_panel_geometry_train_166
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_plane_matcher_120 import attach_global_plane_matcher
from training.joint_pose_map_121 import joint_target_loss_121

source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/ATLAS_INTENSITY_REACH_184_TRAIN_PROTOCOL_20261011.md'
parent = root / 'runs/mapped_surface_mixture_182_train_001'
output = root / 'runs/atlas_intensity_reach_184_train_001'
checkpoint_path = parent / 'full_step_00300.pt'
assert source.drive.upper() == protocol.drive.upper() == output.drive.upper() == 'I:'
assert torch.cuda.is_available() and not output.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

completed = json.loads((parent / 'completed.json').read_text())
config = json.loads((parent / 'config.json').read_text())
with checkpoint_path.open('rb') as stream:
    checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert checkpoint_sha == completed['output_sha256'][checkpoint_path.name]
assert completed['paired_updates'] == 300 and completed['inner_physical_sections'] == 128
assert not any(completed[key] for key in ('calibrated', 'development_used',
    'final_animals_used', 'public_benchmark_used'))
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 300 and checkpoint['arm'] == 'full'
assert checkpoint['config'] == config and checkpoint['calibrated'] is False

context = load_streaming_synthetic_v7_64(device='cuda')
assert json.loads(json.dumps(context['provenance'])) == config['synthetic_provenance']
assert len(context['bases']) == 64
atlas = context['atlas']
zero_atlas = atlas.clone()
zero_atlas[0].zero_()

model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(checkpoint['model'], strict=True)

panel = [json.loads(line) for line in (parent / 'draws.jsonl').open()
    if '"kind": "inner_attempt"' in line and '"used": true' in line]
baseline = [json.loads(line) for line in (parent / 'inner_rows.jsonl').open()
    if '"arm": "full"' in line and '"step": 300' in line]
assert len(panel) == len(baseline) == 128
baseline = {(row['appearance'], row['physical_section_id']): row for row in baseline}
assert len(baseline) == 128

output.mkdir(parents=True, exist_ok=False)
rows = []
with (output / 'rows.jsonl').open('w') as stream, torch.inference_mode():
    for item in panel:
        appearance = item['appearance']
        geometry = sample_panel_geometry_train_166(context, [item['base_index']], item['draw_seed'], 256)
        sampler = (sample_one_shot_slide_artifacts_v4 if appearance == 'v4'
                   else sample_one_shot_slide_artifacts_v3)
        sample = sampler(context, [item['base_index']], item['draw_seed'], 256,
                         source_sample=geometry)
        physical_id = sample['provenance'][0]['physical_section_id']
        assert physical_id == item['provenance']['physical_section_id']
        image_sha = hashlib.sha256(sample['inputs'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        assert image_sha == item['model_inputs_sha256']
        valid = sample['valid_mask'][0].flatten().nonzero()[:, 0]
        pixel = valid[torch.linspace(0, len(valid) - 1, 128, device='cuda').long()][None]
        prediction = model.predict(sample['inputs'])
        exact_prediction = {'state': sample['state'][:, None],
            'feature': prediction['feature'], 'global_feature': prediction['global_feature']}
        mode = torch.zeros((1, 1), device='cuda', dtype=torch.long)
        reflection = sample['reflection'][:, None]
        intact = model.map(exact_prediction, sample['offsets'], mode, reflection,
            (96, 96), atlas, sample['weights'], feature_side=48, source_shape=(256, 256))
        zero = model.map(exact_prediction, sample['offsets'], mode, reflection,
            (96, 96), zero_atlas, sample['weights'], feature_side=48,
            source_shape=(256, 256))
        intact_mm = float(joint_target_loss_121(intact, sample['valid_mask'],
            sample['centre'], sample['state'], sample['reflection'], pixel)['map_loss'][0, 0])
        zero_mm = float(joint_target_loss_121(zero, sample['valid_mask'],
            sample['centre'], sample['state'], sample['reflection'], pixel)['map_loss'][0, 0])
        old = baseline[(appearance, physical_id)]
        assert abs(intact_mm - old['exact_mm']) < 1e-5
        row = {'appearance': appearance, 'physical_section_id': physical_id,
            'model_inputs_sha256': image_sha, 'exact_intact_mm': intact_mm,
            'exact_zero_mm': zero_mm, 'exact_penalty_mm': zero_mm - intact_mm,
            'top1_intact_mm': old['selected_mm'], 'top1_zero_mm': old['null_mm'],
            'top1_penalty_mm': old['null_mm'] - old['selected_mm'],
            'top1_rigid_mm': old['direct_rigid_mm']}
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        rows.append(row)
        if len(rows) % 32 == 0:
            print(f'184 {len(rows)}/128 cases', flush=True)

summary = {'cases': len(rows), 'checkpoint_sha256': checkpoint_sha,
    'parent_completion_sha256': hashlib.sha256((parent / 'completed.json').read_bytes()).hexdigest(),
    'groups': {}, 'calibrated': False, 'development_used': False,
    'final_animals_used': False, 'public_benchmark_used': False}
for appearance in ('v3', 'v4'):
    group = [row for row in rows if row['appearance'] == appearance]
    summary['groups'][appearance] = {
        'n': len(group),
        'exact_intact_mean_mm': float(np.mean([row['exact_intact_mm'] for row in group])),
        'exact_zero_mean_mm': float(np.mean([row['exact_zero_mm'] for row in group])),
        'exact_penalty_mean_mm': float(np.mean([row['exact_penalty_mm'] for row in group])),
        'top1_penalty_mean_mm': float(np.mean([row['top1_penalty_mm'] for row in group])),
        'rigid_bins': {}}
    for label, lower, upper in (('lt_0p5', 0, .5), ('0p5_to_1p5', .5, 1.5),
                                ('ge_1p5', 1.5, float('inf'))):
        subset = [row for row in group if lower <= row['top1_rigid_mm'] < upper]
        summary['groups'][appearance]['rigid_bins'][label] = {
            'n': len(subset), 'top1_penalty_mean_mm': None if not subset else
                float(np.mean([row['top1_penalty_mm'] for row in subset]))}
(output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
output_sha = {}
for name in ('rows.jsonl', 'summary.json'):
    with (output / name).open('rb') as stream:
        output_sha[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(output / 'completed.json').write_text(json.dumps({
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'output_sha256': output_sha, 'cases': len(rows),
    'checkpoint_sha256': checkpoint_sha, 'train_only': True}, indent=2) + '\n')
print(json.dumps(summary['groups'], allow_nan=False), flush=True)
