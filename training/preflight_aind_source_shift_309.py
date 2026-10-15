"""Paired TRAIN-inner source-shift probe; AIND grid is not a pose label."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import rigid_points_090

torch.set_num_threads(4)
run = ROOT / 'runs/first_choice_exposure_309_001'
panel = run / 'panels.pt'
checkpoint = run / 'single/pose_local_20000.pt'
source = ROOT / 'data/aind_smartspim_568105_registered'
volume_path = source / 'registered_ml_dv_ap_float32.npy'
source_receipt = json.loads((source / 'source.json').read_text(encoding='utf8'))
with volume_path.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == source_receipt['npy_sha256']
volume = np.load(volume_path, mmap_mode='r')
assert volume.shape == (456, 320, 528)
sample = volume[::4, ::4, ::4]
positive = sample[np.isfinite(sample) & (sample > 0)]
lo, hi = np.quantile(np.log1p(positive), [.01, .99])

cases = torch.load(panel, map_location='cpu', weights_only=True)['v10']
model, _ = load_one_shot_checkpoint(checkpoint)
model.requires_grad_(False)
context = load_fov_balance_v10('cuda')
registered = torch.from_numpy(np.array(volume.transpose(2, 1, 0))).cuda()
registered.log1p_().sub_(float(lo)).div_(float(hi - lo)).clamp_(0, 1)
rows = []

with torch.inference_mode():
    for ml_direction in ('as_stored', 'mirrored'):
        context['atlas'][0].copy_(registered if ml_direction == 'as_stored'
                                  else registered.flip(-1))
        for start in range(0, len(cases), 4):
            batch = cases[start:start + 4]
            images = []
            for case in batch:
                meta = case['meta']
                draw = sample_fov_balance_v10(
                    context, [meta['base'] * VARIANTS + meta['variant']],
                    meta['seed'], 192)
                assert draw['provenance'][0]['physical_section_id'] == meta['physical_section_id']
                assert torch.equal(draw['state'][0].cpu(), case['state'])
                sites = (case['chart'] * 192).long()
                indices = sites[:, 1] * 192 + sites[:, 0]
                assert torch.equal(draw['centre'][0].reshape(-1, 3)[indices.cuda()].cpu(),
                                   case['truth'])
                images.append(draw['inputs'][0])
            prediction = model.predict(torch.stack(images))
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = torch.arange(2, device='cuda')[None, None].expand(
                len(batch), model.modes, -1)
            chart = torch.stack([case['chart'] for case in batch]).cuda()
            truth = torch.stack([case['truth'] for case in batch]).cuda()
            proposed = rigid_points_090(states, reflections,
                                        chart[:, None, None], (192, 192))
            error = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
            first = mass.argmax(-1)
            top16 = mass.topk(16, -1).indices
            oracle = error.gather(1, top16).min(-1).values
            normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
            true_normal = full_frame_state_to_components(torch.stack(
                [case['state'] for case in batch]).cuda())[1][:, :, 2]
            angle = torch.rad2deg(torch.acos((normals * true_normal[:, None]
                                             ).sum(-1).abs().clamp(0, 1)))
            for i, case in enumerate(batch):
                rows.append({'id': case['meta']['physical_section_id'],
                             'ml_direction': ml_direction,
                             'base': case['meta']['base'],
                             'visible_fraction': case['meta']['postartifact_visible_fraction'],
                             'obliquity_deg': case['meta']['obliquity_deg'],
                             'brush': bool(case['inputs'][2, 0, 0]),
                             'first_error_mm': float(error[i, first[i]]),
                             'top16_oracle_mm': float(oracle[i]),
                             'first_normal_error_deg': float(angle[i, first[i] // 2])})

summary = {}
for direction in ('as_stored', 'mirrored'):
    selected = [row for row in rows if row['ml_direction'] == direction]
    error = np.array([row['first_error_mm'] for row in selected])
    oracle = np.array([row['top16_oracle_mm'] for row in selected])
    angle = np.array([row['first_normal_error_deg'] for row in selected])
    summary[direction] = {
        'n': len(selected), 'first_mean_mm': float(error.mean()),
        'first_median_mm': float(np.median(error)),
        'first_p90_mm': float(np.percentile(error, 90)),
        'first_over5mm': int((error > 5).sum()),
        'top16_oracle_mean_mm': float(oracle.mean()),
        'top16_capture_le1p5mm': int((oracle <= 1.5).sum()),
        'first_normal_median_deg': float(np.median(angle)),
    }
out = ROOT / 'reports/aind_source_shift_309_train_568105'
out.mkdir(parents=True, exist_ok=False)
(out / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
(out / 'summary.json').write_text(json.dumps({
    'summary': summary, 'source': source_receipt,
    'intensity': {'transform': 'clip((log1p(raw)-lo)/(hi-lo),0,1)',
                  'lo': float(lo), 'hi': float(hi),
                  'quantiles': '1,99 percent of positive level-0 values sampled every fourth voxel'},
    'panel_sha256': hashlib.sha256(panel.read_bytes()).hexdigest(),
    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'scope': 'same 256 TRAIN-inner virtual cuts; alternate single-animal source intensity; no physical histology or pose labels',
}, indent=2) + '\n', encoding='utf8')
print(summary)
