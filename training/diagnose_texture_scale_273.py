"""Raw versus two fixed Gaussian input scales on matched acquired and synthetic cases."""
import hashlib
import json
import os
import sys
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
from training.global_atlas_contrast_090 import rigid_points_090


PARENT = ROOT / 'runs/pose_capture_270_v9_continuation_001'
TRANSFER = ROOT / 'runs/acquired_train_transfer_272_001'
OUT = ROOT / 'runs/texture_scale_273_train_001'
SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/TEXTURE_SCALE_273_PROTOCOL_20261012.md'
assert SOURCE.drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
parent = json.loads((PARENT / 'completed.json').read_text())
transfer = json.loads((TRANSFER / 'completed.json').read_text())
assert parent['selected_step'] == 8000 and transfer['candidate'] == 272
checkpoint_path = PARENT / 'pose_step_08000.pt'
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == parent['checkpoint_sha256']['8000']
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64).cuda().eval()
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model'])
real_cases = [json.loads(line) for line in (TRANSFER / 'cases.jsonl').open()]
synthetic = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(real_cases) == 98 and len(synthetic) == 64
sources = {
    'coronal': (ROOT / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl',
                ROOT / 'data/joint_v7_allen_fullcanvas_192_001/images.npy',
                'model_pixel_to_ap_dv_ml_um'),
    'sagittal': (ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/geometry.jsonl',
                 ROOT / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008/model_input.npy',
                 'model_pixel_to_ccf_ref9_ap_dv_ml_um'),
}
records = {domain: {row['array_row_index']: row for row in
            [json.loads(line) for line in records_path.open()]}
           for domain, (records_path, _, _) in sources.items()}
images = {domain: np.load(images_path, mmap_mode='r')
          for domain, (_, images_path, _) in sources.items()}
config = {'candidate': 273, 'parent_checkpoint_sha256': parent['checkpoint_sha256']['8000'],
    'parent_panel_sha256': parent['inner_panel_sha256'],
    'transfer_cases_sha256': transfer['cases_sha256'],
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'arms': ('raw', 'gaussian_sigma_1', 'gaussian_sigma_2'),
    'acquired_weak_label_role': 'orientation diagnostic only',
    'synthetic_panel': '270 fresh v9 no-brush only'}
OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w') as output:
    for case in real_cases + [item for item in synthetic if not bool(item['inputs'][2, 0, 0])]:
        acquired = 'domain' in case
        if acquired:
            domain = case['domain']
            record = records[domain][case['array_row_index']]
            assert record['section_id'] == case['section_id']
            image = torch.from_numpy(np.array(images[domain][case['array_row_index'], 0],
                                               dtype=np.float32, copy=True))[None, None].cuda()
            if domain == 'sagittal':
                image = F.interpolate(image, (192, 192), mode='area')
            base = torch.cat((image, torch.zeros(1, 4, 192, 192, device='cuda')), 1)
            matrix = np.asarray(record[sources[domain][2]], dtype=np.float64)
            weak_normal = np.cross(matrix[:, 0], matrix[:, 1])
            weak_normal /= np.linalg.norm(weak_normal)
            true_normal = torch.as_tensor(weak_normal, device='cuda')
        else:
            domain = 'synthetic'
            base = case['inputs'][None].cuda()
            chart = case['chart'].cuda()
            truth = rigid_points_090(case['state'][None].cuda(),
                                     case['reflection'][None].cuda(), chart, (192, 192))
        for arm, sigma in (('raw', 0), ('gaussian_sigma_1', 1), ('gaussian_sigma_2', 2)):
            if sigma:
                axis = torch.arange(-3, 4, device='cuda', dtype=base.dtype)
                gaussian = torch.exp(-.5 * (axis / sigma).square())
                gaussian /= gaussian.sum()
                kernel = torch.outer(gaussian, gaussian)[None, None]
                channel = F.conv2d(F.pad(base[:, :1], (3, 3, 3, 3), mode='reflect'), kernel)
                network_input = torch.cat((channel, base[:, 1:]), 1)
            else:
                network_input = base
            prediction = model.predict(network_input)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            order = prior.argsort(descending=True)
            selected = (order[:16] // 2).tolist()
            result = {'domain': domain, 'arm': arm,
                'section_id': case['section_id'] if acquired else case['physical_section_id'],
                'donor_id': case['donor_id'] if acquired else None,
                'base_index': None if acquired else case['base_index']}
            if acquired:
                normal = full_frame_state_to_components(prediction['state'][0])[1][:, :, 2]
                result['first_axis_correct'] = int(normal[selected[0]].abs().argmax()) == int(np.abs(weak_normal).argmax())
                result['first_weak_normal_angle_deg'] = float(torch.rad2deg(torch.acos(
                    (normal[selected[0]] * true_normal).sum().abs().clamp(0, 1))))
                if arm == 'raw':
                    assert result['first_axis_correct'] == case['first_axis_correct']
                    assert abs(result['first_weak_normal_angle_deg'] - case['first_weak_normal_angle_deg']) < 1e-3
            else:
                states = prediction['state'][0, order[:16] // 2][None]
                reflections = (order[:16] % 2)[None]
                points = rigid_points_090(states, reflections, chart[None], (192, 192))
                errors = ((points - truth[:, None]).norm(dim=-1).mean(-1) / 1000)[0]
                result['top16_hit'] = bool((errors <= 1.5).any())
                result['top1_mm'] = float(errors[0])
            rows.append(result)
            output.write(json.dumps(result, allow_nan=False) + '\n')
summary = {'real_cases': len(real_cases), 'synthetic_no_brush_cases':
    sum(not bool(item['inputs'][2, 0, 0]) for item in synthetic), 'arms': {}}
for arm in config['arms']:
    summary['arms'][arm] = {}
    for domain in ('coronal', 'sagittal', 'synthetic'):
        group = [row for row in rows if row['arm'] == arm and row['domain'] == domain]
        summary['arms'][arm][domain] = (
            {'n': len(group), 'first_axis_correct': sum(row['first_axis_correct'] for row in group),
             'median_normal_angle_deg': float(np.median([row['first_weak_normal_angle_deg'] for row in group]))}
            if domain != 'synthetic' else
            {'n': len(group), 'top16_hits': sum(row['top16_hit'] for row in group),
             'mean_top1_mm': float(np.mean([row['top1_mm'] for row in group]))})
raw, sigma1, sigma2 = (summary['arms'][name] for name in config['arms'])
summary['future_training_comparison_justified'] = {
    name: (data['sagittal']['first_axis_correct'] - raw['sagittal']['first_axis_correct'] >= 10
           and raw['coronal']['first_axis_correct'] - data['coronal']['first_axis_correct'] <= 3
           and raw['synthetic']['top16_hits'] - data['synthetic']['top16_hits'] <= 4)
    for name, data in (('gaussian_sigma_1', sigma1), ('gaussian_sigma_2', sigma2))}
(OUT / 'config.json').write_text(json.dumps(config, indent=2))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'completed.json').write_text(json.dumps({'candidate': 273,
    'rows_sha256': hashlib.sha256((OUT / 'rows.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((OUT / 'config.json').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((OUT / 'summary.json').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', **summary}), flush=True)
