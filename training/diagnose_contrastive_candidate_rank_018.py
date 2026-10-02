"""Frozen candidate-score versus physical-error diagnostic for 018 batch 8,000."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
checkpoint = root / 'runs/one_shot_contrastive_fit_018/joint_step_08000.pt'
out = root / 'runs/one_shot_contrastive_fit_018_candidate_rank_diagnostic'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
assert sum(row['eligible'] for row in records) == 185
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                               vector_refinement=True, candidate_ranking=True,
                               fitted_ranking=True).cuda().eval()
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert saved['step'] == 8000 and not saved['calibrated']
model.load_state_dict(saved['model'], strict=True)
del saved
out.mkdir(parents=True, exist_ok=False)
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        if not record['eligible']:
            continue
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][0, :, None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit'][0]),
            F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
        choice = prior.topk(8).indices[None]
        selected = {**prediction,
                    'state': prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12)),
                    'log_mass': prediction['log_mass'].gather(1, choice // 2),
                    'reflection_logit': prediction['reflection_logit'].gather(1, choice // 2)}
        index = torch.arange(8, device='cuda')[None]
        first = model.map(selected, offsets, index, choice % 2, (64, 64), atlas, weights,
                          return_refinement_feature=True, feature_side=64,
                          source_shape=(256, 256))
        refined_state, delta, _ = model.refine(first['refinement_feature'], selected['state'])
        refined = {**selected, 'state': refined_state}
        mapped = model.map(refined, offsets, index, choice % 2, (96, 96), atlas, weights,
                           feature_side=96, source_shape=(256, 256))
        score = model.score_fitted_candidates(image, refined, mapped, atlas, weights) + delta
        physical = []
        target = reference[valid]
        for start in range(0, 8, 2):
            high = model.map(refined, offsets, index[:, start:start + 2],
                             (choice % 2)[:, start:start + 2], (256, 256), atlas, weights)
            physical.extend((high['centre_surface_ccf_ap_dv_ml_um'][0, :, valid]
                             - target[None]).norm(dim=-1).mean(-1).tolist())
        row = {key: record[key] for key in ('animal_id', 'specimen_id', 'experiment_id',
                                           'section_id', 'sha256')}
        row.update(choice=choice[0].tolist(), prior=prior[choice[0]].tolist(),
                   fitted_score=score[0].tolist(), physical_mapped_um=physical,
                   fit_energy=mapped['fit_energy'][0].tolist(),
                   local_rms_um=mapped['local_displacement_um'][0].square().mean((1, 2, 3)).sqrt().tolist(),
                   reliability=mapped['correspondence_logit'][0].sigmoid().mean((1, 2, 3)).tolist(),
                   valid_fraction=float(valid.float().mean()))
        stream.write(json.dumps(row) + '\n')
(out / 'completed.json').write_text(json.dumps({
    'rows': 185, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rows_sha256': hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest(),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'cases': 185}), flush=True)
