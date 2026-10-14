"""Frozen 309 model on the same TRAIN-inner cuts with one registered STPT red volume."""
import hashlib
import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

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
cases = torch.load(run / 'panels.pt', map_location='cpu', weights_only=True)['v10']
with (run / 'single/readouts.jsonl').open(encoding='utf8') as stream:
    readouts = [json.loads(line) for line in stream]
baseline = {row['physical_section_id']: row for record in readouts if record['step'] == 20000
            for row in record['rows'] if row['cohort'] == 'v10'}
assert len(cases) == len(baseline) == 256
model, _ = load_one_shot_checkpoint(run / 'single/pose_local_20000.pt')
model.requires_grad_(False)
context = load_fov_balance_v10('cuda')

archive = ROOT / 'data/allen_stpt_red_310_001/156198187_607436902.zip'
with archive.open('rb') as stream:
    archive_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
with ZipFile(archive) as source:
    header = source.read('resampled_red.mhd').decode()
    assert 'DimSize = 528 320 456' in header and 'ElementSpacing = 25 25 25' in header
    assert 'Offset = 0 0 0' in header and 'ElementType = MET_USHORT' in header
    red = np.frombuffer(source.read('resampled_red.raw'), dtype='<u2').reshape(
        (456, 320, 528)).transpose(2, 1, 0)
brain = context['atlas'][1].cpu().numpy() > 0
low, high = np.quantile(red[brain], (0.01, 0.99))
red = np.clip((red.astype(np.float32) - low) / (high - low), 0, 1)
red[~brain] = 0
context['atlas'][0].copy_(torch.from_numpy(red).cuda())
del red, brain

rows = []
with torch.inference_mode():
    for start in range(0, len(cases), 4):
        batch = cases[start:start + 4]
        images = []
        for case in batch:
            meta = case['meta']
            sample = sample_fov_balance_v10(context, [meta['base'] * VARIANTS + meta['variant']],
                                            meta['seed'], 192)
            assert sample['provenance'][0]['physical_section_id'] == meta['physical_section_id']
            assert torch.equal(sample['state'][0].cpu(), case['state'])
            sites = (case['chart'] * 192).long()
            indices = sites[:, 1] * 192 + sites[:, 0]
            assert torch.equal(sample['centre'][0].reshape(-1, 3)[indices.cuda()].cpu(), case['truth'])
            images.append(sample['inputs'][0])
        prediction = model.predict(torch.stack(images))
        mass = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        reflections = torch.arange(2, device='cuda')[None, None].expand(len(batch), model.modes, -1)
        chart = torch.stack([case['chart'] for case in batch]).cuda()
        truth = torch.stack([case['truth'] for case in batch]).cuda()
        proposed = rigid_points_090(states, reflections, chart[:, None, None], (192, 192))
        error = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1).flatten(1) / 1000
        first = mass.argmax(-1)
        oracle = error.gather(1, mass.topk(16, -1).indices).min(-1).values
        normals = full_frame_state_to_components(prediction['state'])[1][..., :, 2]
        true_normal = full_frame_state_to_components(torch.stack(
            [case['state'] for case in batch]).cuda())[1][:, :, 2]
        angle = torch.rad2deg(torch.acos((normals * true_normal[:, None]
            ).sum(-1).abs().clamp(0, 1)))
        for i, case in enumerate(batch):
            key = case['meta']['physical_section_id']
            rows.append({'id': key, 'baseline_first_mm': baseline[key]['first_rigid_site_gap_mm'],
                         'baseline_top16_oracle_mm': baseline[key]['top16_oracle_gap_mm'],
                         'red_first_mm': float(error[i, first[i]]),
                         'red_top16_oracle_mm': float(oracle[i]),
                         'red_first_normal_deg': float(angle[i, first[i] // 2])})

base = np.array([row['baseline_first_mm'] for row in rows])
red = np.array([row['red_first_mm'] for row in rows])
oracle = np.array([row['red_top16_oracle_mm'] for row in rows])
summary = {'n': len(rows), 'baseline_first_mean_mm': float(base.mean()),
           'red_first_mean_mm': float(red.mean()), 'paired_change_mm': float((red - base).mean()),
           'red_first_median_mm': float(np.median(red)), 'red_first_p90_mm': float(np.percentile(red, 90)),
           'baseline_over5mm': int((base > 5).sum()), 'red_over5mm': int((red > 5).sum()),
           'red_top16_oracle_mean_mm': float(oracle.mean()),
           'red_top16_capture_le1p5mm': int((oracle <= 1.5).sum()),
           'red_first_normal_median_deg': float(np.median([row['red_first_normal_deg'] for row in rows]))}
out = ROOT / 'reports/allen_stpt_red_source_shift_310_train'
out.mkdir(parents=True, exist_ok=True)
(out / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
(out / 'summary.json').write_text(json.dumps({'summary': summary,
    'source': {'section_data_set_id': 156198187, 'well_known_file_id': 607436902,
               'archive_sha256': archive_sha256, 'red_brain_q01_q99': [float(low), float(high)],
               'grid': 'AP posterior, DV inferior, ML right; 25 um; gross QC, not exact landmarks'},
    'scope': 'frozen model, same TRAIN-inner virtual planes; contrast and specimen anatomy change together; not physical histology'},
    indent=2) + '\n', encoding='utf8')
print(summary)
