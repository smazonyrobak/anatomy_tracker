"""No-training exact-pose local image/atlas information assay."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import render_finite_thickness_coordinate_grid
from training.arbitrary_plane_one_shot_slide_artifacts_v9 import sample_one_shot_slide_artifacts_v9
from training.arbitrary_plane_postartifact_gauge_v1 import sample_postartifact_gauge_v1
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.train_image_map_282 import SIDE, digest, mapped_sites, rigid_surface, tensor_digest


SEED = 20261012288
SOURCE = Path(__file__).resolve().parent
PROTOCOL = SOURCE.parent / 'docs/publication/MAP_INTENSITY_INFORMATION_288_PROTOCOL_20261012.md'
PARENT = ROOT / 'runs/map_feature_287_001'
RUN = ROOT / 'reports/map_intensity_information_288_001'
SOURCE_NAMES = ('audit_map_intensity_information_288.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_one_shot_slide_artifacts_v9.py',
    'arbitrary_plane_postartifact_gauge_v1.py', 'arbitrary_plane_streaming_synthetic_v7_64.py',
    'train_image_map_282.py', 'run_exact_map_capacity_281.py', 'train_blind_fit_score_280.py')


def absolute_correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return None if scale < 1e-8 else float(abs(np.dot(a, b) / scale))


assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:' and not RUN.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.perf_counter()
receipt = json.loads((PARENT / 'completed.json').read_text(encoding='utf8'))
parent_config = json.loads((PARENT / 'config.json').read_text(encoding='utf8'))
assert receipt['candidate'] == 287 and not any(receipt['advance'].values())
assert digest(PARENT / 'completed.json') == \
    '239d6d4f008029e331234bec0bcc088f3d5ab71c5fa5d856ddbd1e0120a6f63d'
assert digest(PARENT / 'inner_panel.pt') == receipt['files_sha256']['inner_panel.pt']
context = load_streaming_synthetic_v7_64(device='cuda')
for name, sha in parent_config['source_sha256'].items():
    assert digest(SOURCE / name) == sha
assert json.loads(json.dumps(context['provenance'])) == parent_config['synthetic_provenance']
panel = torch.load(PARENT / 'inner_panel.pt', map_location='cpu', weights_only=False)
assert len(panel) == 64
RUN.mkdir(parents=True, exist_ok=False)
config = {'candidate': 288, 'seed': SEED, 'parent_287_completed_sha256': digest(PARENT / 'completed.json'),
    'parent_287_panel_sha256': digest(PARENT / 'inner_panel.pt'), 'protocol_sha256': digest(PROTOCOL),
    'source_sha256': {name: digest(SOURCE / name) for name in SOURCE_NAMES},
    'section_count': 64, 'sites_per_section': 512, 'permutations': 16,
    'atlas_support_threshold': .9, 'minimum_common_sites': 64,
    'atlas_origin_ap_dv_ml_um': [0., 0., 0.], 'atlas_spacing_ap_dv_ml_um': [25., 25., 25.],
    'score': 'absolute Pearson correlation on fixed common atlas-supported surviving sites',
    'source_role': 'synthetic TRAIN-inner exact-coordinate information assay; no model training',
    'dev_cal_final_or_public_benchmark_used': False}
(RUN / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf8')

rows = []
with (RUN / 'rows.jsonl').open('w', encoding='utf8') as output, torch.no_grad():
    for case, item in enumerate(panel):
        old = sample_one_shot_slide_artifacts_v9(
            context, [item['subject_index']], item['draw_seed'], SIDE)
        new = sample_postartifact_gauge_v1(
            context, [item['subject_index']], item['draw_seed'], SIDE, source_sample=old)
        assert old['provenance'][0]['physical_section_id'] == item['physical_section_id']
        assert torch.equal(old['inputs'][0].cpu(), item['inputs'])
        assert torch.equal(new['state'][0].cpu(), item['state'])
        assert tensor_digest(old['inputs']) == item['input_sha256']
        indices = item['site_indices'].cuda()
        target = old['centre'][0].reshape(-1, 3)[indices]
        assert torch.equal(target.cpu(), item['site_target'])
        assert tensor_digest(torch.cat((indices[:, None].float(), target), -1)) == item['target_sha256']
        surface, frame, _ = rigid_surface(new['state'], old['reflection'])
        rigid = mapped_sites({'centre_surface_ccf_ap_dv_ml_um': surface[:, None]}, indices)[0, 0]
        displacement = target - rigid
        points = [target, rigid]
        rng = np.random.default_rng(SEED + case)
        points.extend(rigid + displacement[torch.as_tensor(rng.permutation(512), device='cuda')]
                      for _ in range(16))
        intensities, supports = [], []
        normal = frame[0, :, 2]
        offsets, weights = old['offsets'], old['weights']
        for locations in points:
            slab = (locations[None, None, None] +
                    offsets[:, :, None, None, None] * normal[None, None, None, None])
            rendered = render_finite_thickness_coordinate_grid(
                context['atlas'], slab, (0., 0., 0.), (25., 25., 25.), weights)
            support = rendered[0, 1, 0]
            intensities.append((rendered[0, 0, 0] / support.clamp_min(1e-6)).cpu().numpy())
            supports.append(support.cpu().numpy())
        common = (np.stack(supports) > .9).all(0)
        source = old['inputs'][0, 0].flatten()[indices].cpu().numpy()[common]
        scores = ([absolute_correlation(source, image[common]) for image in intensities]
                  if common.sum() >= 64 else [None] * len(points))
        eligible = all(score is not None for score in scores)
        shuffled = scores[2:] if eligible else []
        record = {'case': case, 'physical_section_id': item['physical_section_id'],
            'base_index': item['base_index'], 'slot': item['slot'],
            'input_sha256': item['input_sha256'], 'target_sha256': item['target_sha256'],
            'common_sites': int(common.sum()), 'eligible': eligible,
            'target_rigid_shift_mm': float(displacement.norm(dim=-1).mean() / 1000),
            'true_abs_corr': scores[0], 'rigid_abs_corr': scores[1],
            'true_minus_rigid_abs_corr': scores[0] - scores[1] if eligible else None,
            'shuffled_abs_corr': shuffled,
            'shuffled_p95_abs_corr': float(np.percentile(shuffled, 95)) if eligible else None,
            'true_above_shuffled_p95': bool(scores[0] > np.percentile(shuffled, 95)) if eligible else None,
            'brush_available': bool(old['inputs'][0, 2, 0, 0]),
            'exterior_style': old['provenance'][0]['one_shot_slide_artifacts_v9']['exterior_style'],
            'event_flags': item['event_flags'], 'visible_fraction': item['visible_fraction']}
        output.write(json.dumps(record, allow_nan=False) + '\n')
        rows.append(record)
        if (case + 1) % 16 == 0:
            output.flush()
            print(json.dumps({'event': 'case_milestone', 'cases': case + 1,
                'eligible': sum(row['eligible'] for row in rows)}), flush=True)

eligible_rows = [row for row in rows if row['eligible']]
by_base = {str(base): float(np.mean([row['true_minus_rigid_abs_corr'] for row in eligible_rows
                                   if row['base_index'] == base]))
           for base in range(56, 64)
           if any(row['base_index'] == base for row in eligible_rows)}
summary = {'cases': len(rows), 'eligible_cases': len(eligible_rows),
    'true_above_rigid_cases': sum(row['true_minus_rigid_abs_corr'] > 0 for row in eligible_rows),
    'true_above_shuffled_p95_cases': sum(row['true_above_shuffled_p95'] for row in eligible_rows),
    'mean_true_minus_rigid_abs_corr': float(np.mean([
        row['true_minus_rigid_abs_corr'] for row in eligible_rows])) if eligible_rows else None,
    'mean_true_abs_corr': float(np.mean([row['true_abs_corr'] for row in eligible_rows]))
        if eligible_rows else None,
    'mean_rigid_abs_corr': float(np.mean([row['rigid_abs_corr'] for row in eligible_rows]))
        if eligible_rows else None,
    'mean_target_rigid_shift_mm': float(np.mean([row['target_rigid_shift_mm'] for row in eligible_rows]))
        if eligible_rows else None,
    'base_mean_margins': by_base,
    'positive_margin_bases': sum(value > 0 for value in by_base.values())}
gate = ('inconclusive' if len(eligible_rows) < 48 else
        'pass' if summary['true_above_rigid_cases'] >= 48 and
        summary['positive_margin_bases'] >= 6 and
        summary['mean_true_minus_rigid_abs_corr'] >= .01 and
        summary['true_above_shuffled_p95_cases'] >= 38 else 'fail')
completed = {'candidate': 288, 'summary': summary, 'information_gate': gate,
    'elapsed_seconds': time.perf_counter() - started,
    'source_sha256': config['source_sha256'],
    'files_sha256': {name: digest(RUN / name) for name in ('config.json', 'rows.jsonl')},
    'result_role': config['source_role'], 'calibrated': False}
(RUN / 'completed.json').write_text(json.dumps(completed, indent=2, allow_nan=False), encoding='utf8')
print(json.dumps({'event': 'completed', 'gate': gate, 'summary': summary}), flush=True)
