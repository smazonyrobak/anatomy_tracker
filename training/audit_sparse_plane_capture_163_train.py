"""TRAIN-only: distinguish sparse-slice beam failure from missing candidates."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_one_shot_slide_artifacts_v4 import sample_one_shot_slide_artifacts_v4
from training.arbitrary_plane_panel_geometry_train_151 import sample_panel_geometry_train_151
from training.arbitrary_plane_streaming_synthetic_v7_64 import load_streaming_synthetic_v7_64
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


source = root / 'runs/joint_feedback_161_train_preflight_001'
parent = root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt'
out = root / 'reports/sparse_capture_163_train_001'
assert not out.exists()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rows = [json.loads(line) for line in (source / 'rows.jsonl').open()]
completed = json.loads((source / 'completed.json').read_text())
with (source / 'rows.jsonl').open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == completed['output_sha256']['rows.jsonl']
with parent.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == completed['checkpoint_sha256']
selected = [row for row in rows if row['eligible'] and row['valid_fraction'] < .08]
assert len(selected) == 26
context = load_streaming_synthetic_v7_64(device='cuda')
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(parent, map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)
del saved
flags = torch.arange(2, device='cuda')[None, None].expand(1, model.modes, 2)
results = []
with torch.inference_mode():
    for row in selected:
        original = sample_panel_geometry_train_151(context, [row['base_index']], row['draw_seed'], 256)
        sample = sample_one_shot_slide_artifacts_v4(context, [row['base_index']],
            row['draw_seed'], 256, source_sample=original)
        assert sample['provenance'][0]['physical_section_id'] == row['physical_section_id']
        valid = sample['valid_mask'][0]
        assert int(valid.sum()) == row['valid_pixels']
        pixel = valid.nonzero(as_tuple=False)
        chart = torch.stack((pixel[:, 1], pixel[:, 0]), -1).float() / 256
        prediction = model.predict(sample['inputs'])
        states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
        truth = rigid_points_090(sample['state'], sample['reflection'], chart[None])
        proposed = rigid_points_090(states, flags, chart[None, None, None])
        cost = (proposed - truth[:, None, None]).norm(dim=-1).mean(-1).flatten() / 1000
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
        beam = torch.as_tensor(row['beam_branch_ids'], device='cuda')
        assert bool(torch.allclose(cost[beam], torch.as_tensor(
            row['beam_rigid_mm'], device='cuda'), atol=1e-4))
        results.append({'physical_section_id': row['physical_section_id'],
            'valid_fraction': row['valid_fraction'],
            'nearest_cardinal_angle_deg': row['nearest_cardinal_angle_deg'],
            'top1_mm': float(cost[prior.argmax()]),
            'blind16_best_mm': float(cost[beam].min()),
            'all160_best_mm': float(cost.min()),
            'all160_near_count': int((cost <= 1.5).sum()),
            'best160_prior_rank': int((prior > prior[cost.argmin()]).sum()) + 1})

summary = {'draws_under_8pct': sum(row['valid_fraction'] < .08 for row in rows),
    'eligible_under_8pct': len(results),
    'near_in_blind16': sum(r['blind16_best_mm'] <= 1.5 for r in results),
    'near_in_all160': sum(r['all160_best_mm'] <= 1.5 for r in results),
    'median_best160_mm': float(np.median([r['all160_best_mm'] for r in results])),
    'median_blind16_mm': float(np.median([r['blind16_best_mm'] for r in results])),
    'source_run': str(source), 'parent_checkpoint_sha256': completed['checkpoint_sha256'],
    'scope': 'TRAIN synthetic sparse-cut diagnostic only; no independent animal or DEV result'}
out.mkdir(parents=True, exist_ok=False)
(out / 'rows.json').write_text(json.dumps(results, indent=2, allow_nan=False))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
print(json.dumps(summary))
