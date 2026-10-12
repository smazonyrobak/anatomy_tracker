"""Read-only post-hoc oracle move/stay ceiling for frozen 292 map directions."""
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

from training.arbitrary_plane_multiscale_pair_map_v1 import MultiScalePairJointModel
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.train_image_map_282 import digest, mapped_sites


SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/MOVE_GATE_CEILING_293_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/multiscale_pair_map_292_001'
RUN = ROOT / 'reports/move_gate_ceiling_293_001'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
receipt = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
assert digest(PARENT / 'completed.json') == \
    '5823e81dd08ff3d1eb316a30662c451057a365814a0f5b962070a49e4d008306'
assert receipt['selected_step'] == 2400 and receipt['advance'] is False
checkpoint = PARENT / 'joint_multiscale_pair_step_02400.pt'
assert digest(checkpoint) == receipt['checkpoint_sha256']['2400']
panel_path = PARENT / 'inner_panel.pt'
assert digest(panel_path) == receipt['files_sha256']['inner_panel.pt']
panel = torch.load(panel_path, map_location='cpu', weights_only=False)
context = load_streaming_synthetic_v7_64(device='cuda')
model = MultiScalePairJointModel().cuda().eval()
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
RUN.mkdir(parents=True, exist_ok=False)
config = {'candidate': 293, 'role': 'post-hoc TRAIN-inner oracle feasibility, not inference',
    'parent_receipt_sha256': digest(PARENT / 'completed.json'),
    'checkpoint_sha256': digest(checkpoint), 'panel_sha256': digest(panel_path),
    'protocol_sha256': digest(PROTOCOL),
    'source_sha256': digest(SOURCE / 'audit_move_gate_ceiling_293.py'),
    'model_source_sha256': digest(SOURCE / 'arbitrary_plane_multiscale_pair_map_v1.py'),
    'synthetic_provenance': context['provenance'],
    'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
groups = {name: {'rigid': [], 'raw': [], 'oracle': [], 'benefit': [],
                 'gate': [], 'move_mm': [], 'peak': []}
          for name in ('all', 'intact', 'fragment')}
with torch.inference_mode():
    for item in panel:
        mapped = model.map_at_pose(item['inputs'][None].cuda(), item['state'][None].cuda(),
            item['reflection'][None].cuda(), item['offsets'][None].cuda(),
            item['weights'][None].cuda(), context['atlas'])
        peak_field = mapped['levels'][192]['logits'].softmax(1).amax(1).flatten()
        for name, key in (('all', 'site_indices'), ('intact', 'intact_indices'),
                          ('fragment', 'fragment_indices')):
            indices = item[key].cuda()
            if not len(indices):
                continue
            target = item['centre'].reshape(-1, 3)[item[key]].cuda()
            raw = mapped_sites(mapped, indices)[0, 0]
            rigid = mapped_sites({'centre_surface_ccf_ap_dv_ml_um':
                mapped['rigid_surface_ccf_ap_dv_ml_um'][:, None]}, indices)[0, 0]
            move = raw - rigid
            gate = ((target - rigid) * move).sum(-1) / move.square().sum(-1).clamp_min(1e-8)
            gate = gate.clamp(0, 1)
            oracle = rigid + gate[:, None] * move
            rigid_mm = (rigid - target).norm(dim=-1) / 1000
            raw_mm = (raw - target).norm(dim=-1) / 1000
            oracle_mm = (oracle - target).norm(dim=-1) / 1000
            result = groups[name]
            for field, value in (('rigid', rigid_mm), ('raw', raw_mm), ('oracle', oracle_mm),
                ('benefit', rigid_mm - raw_mm), ('gate', gate),
                ('move_mm', move.norm(dim=-1) / 1000), ('peak', peak_field[indices])):
                result[field].extend(value.cpu().tolist())
summary = {}
for name, values in groups.items():
    summary[name] = {'sites': len(values['rigid']),
        'rigid_mean_mm': float(np.mean(values['rigid'])),
        'raw_mean_mm': float(np.mean(values['raw'])),
        'oracle_gated_mean_mm': float(np.mean(values['oracle'])),
        'raw_beneficial_fraction': float(np.mean(np.array(values['benefit']) > 0)),
        'oracle_gate_above_0_1_fraction': float(np.mean(np.array(values['gate']) > .1)),
        'raw_move_median_mm': float(np.median(values['move_mm'])),
        'posterior_peak_median': float(np.median(values['peak']))}
benefit = np.asarray(groups['all']['benefit'])
peak = np.asarray(groups['all']['peak'])
summary['all_peak_quartiles'] = [{'low': float(peak[chunk].min()),
    'high': float(peak[chunk].max()), 'sites': len(chunk),
    'raw_benefit_mean_mm': float(benefit[chunk].mean())}
    for chunk in np.array_split(np.argsort(peak), 4)]
(RUN / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
completed = {'candidate': 293, 'config_sha256': digest(RUN / 'config.json'),
    'summary_sha256': digest(RUN / 'summary.json'), 'summary': summary,
    'role': config['role'], 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'all': summary['all'],
                  'fragment': summary['fragment']}), flush=True)
