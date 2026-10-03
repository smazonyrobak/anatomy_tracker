"""Availability of true atlas keys in the frozen 041 coarse match lists."""
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
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.pose_feedback_global_041 import PoseFeedbackGlobal041

panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
run = root / 'runs/pose_feedback_global_041_pilot'
out = root / 'runs/pose_coarse_topk_044_diagnostic'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
protocol = Path(__file__).parents[1] / 'docs/publication/POSE_COARSE_TOPK_044_PROTOCOL_20261003.md'
head_path = run / 'joint_step_01500.pt'
side, beam = 256, 8
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

panel_rows = [json.loads(line) for line in (panel / 'records.jsonl').open()]
synthetic = [row for row in panel_rows if row['eligible']]
assert len(synthetic) == 177
assert len({row['synthetic_subject_plan_id'] for row in synthetic}) == 8
assert json.loads((run / 'completed.json').read_text())['updates'] == 2000
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                              vector_refinement=True, candidate_ranking=True,
                              fitted_ranking=True).cuda().eval()
model.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
head = PoseFeedbackGlobal041().cuda().eval()
checkpoint = torch.load(head_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 1500 and not checkpoint['calibrated']
head.load_state_dict(checkpoint['head'])
del checkpoint

out.mkdir(parents=True, exist_ok=False)
config = {'step': 1500, 'beam': beam, 'truth_threshold_um': 1500,
          'key_support_threshold': .1, 'query_grid': 'pixel centres 8:256:16',
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
          'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
          'head_sha256': hashlib.sha256(head_path.read_bytes()).hexdigest(),
          'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
          'train_completed_sha256': hashlib.sha256((run / 'completed.json').read_bytes()).hexdigest(),
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(synthetic, 1):
        path = panel / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        ids = valid.flatten().nonzero().flatten()
        chart = torch.stack((ids.remainder(side),
            ids.div(side, rounding_mode='floor')), -1).float()[None] / side
        reference = target.reshape(-1, 3)[ids]
        truth = target[8::16, 8::16].reshape(256, 3)
        visible = valid[8::16, 8::16].reshape(256)
        prediction = model.predict(image)
        prior_score = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = prior_score.topk(beam, -1).indices
        state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        reflection = choice % 2
        centre, frame, basis = full_frame_state_to_components(state)
        chart = chart[:, None].expand(-1, beam, -1, -1).clone()
        chart[..., 0] = torch.where(reflection[..., None].bool(),
                                     255 / 256 - chart[..., 0], chart[..., 0])
        plane = centre[..., None, :] + torch.einsum('bkij,bkqj->bkqi',
            frame[..., :, :2] @ basis, chart - .5)
        prior_error = (plane - reference[None, None]).norm(dim=-1).mean(-1)[0]
        best = int(prior_error.argmin())
        result = head(prediction, image, state, reflection, atlas, offsets, weights)
        distance = torch.cdist(truth[None].expand(beam, -1, -1), result['key_world'][0])
        supported = result['key_support'][0] > .1
        available = ((distance <= 1500) & supported[:, None]).any(-1)
        top = result['logits'][0].topk(64, -1).indices
        hit = ((distance.gather(-1, top) <= 1500) &
               supported[:, None].expand(-1, 256, -1).gather(-1, top))
        row = {key: record[key] for key in ('animal_id', 'specimen_id',
               'experiment_id', 'section_id', 'synthetic_subject_plan_id',
               'appearance_mode', 'sha256')}
        row['valid_cells'] = int(visible.sum())
        row['prior_top1_um'] = float(prior_error[0])
        row['prior_best8_um'] = float(prior_error[best])
        row['prior_best8_index'] = best
        for label, candidate in (('top1', 0), ('best8', best)):
            row[label + '_available_cells'] = int((available[candidate] & visible).sum())
            for k in (8, 32, 64):
                found = hit[candidate, :, :k].any(-1) & visible
                row[f'{label}_top{k}_hits'] = int(found.sum())
                if k == 64:
                    quadrants = found.reshape(16, 16)
                    row[label + '_top64_quadrants'] = sum(int(quadrants[y:y+8, x:x+8].any())
                        for y in (0, 8) for x in (0, 8))
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number in (64, 128, 177):
            stream.flush()
            print(json.dumps({'synthetic_sections': number}), flush=True)

summary = {'synthetic_eligible_sections': len(rows), 'synthetic_subjects': 8,
           'mean_valid_cells': float(np.mean([row['valid_cells'] for row in rows]))}
identities = sorted({row['synthetic_subject_plan_id'] for row in rows})
for label in ('top1', 'best8'):
    summary[label] = {}
    for name in ('availability', 'top8_absolute', 'top32_absolute', 'top64_absolute',
                 'top8_conditional', 'top32_conditional', 'top64_conditional',
                 'top64_three_quadrant_fraction'):
        group_means = []
        for identity in identities:
            cases = [row for row in rows if row['synthetic_subject_plan_id'] == identity]
            if name == 'availability':
                values = [row[label + '_available_cells'] / row['valid_cells'] for row in cases]
            elif name == 'top64_three_quadrant_fraction':
                values = [float(row[label + '_top64_quadrants'] >= 3) for row in cases]
            else:
                k, denominator = name.split('_')
                denominator_field = ('valid_cells' if denominator == 'absolute'
                                     else label + '_available_cells')
                values = [row[f'{label}_{k}_hits'] / row[denominator_field]
                          for row in cases if row[denominator_field] > 0]
            if values:
                group_means.append(float(np.mean(values)))
        summary[label][name] = float(np.mean(group_means))
    summary[label]['sections_with_available_cells'] = sum(
        row[label + '_available_cells'] > 0 for row in rows)
summary['fine_reranker_justified'] = (
    summary['best8']['top64_conditional'] >= .8 and
    summary['top1']['top64_conditional'] >= .6 and
    summary['best8']['top64_three_quadrant_fraction'] >= .8)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    name + '_sha256': hashlib.sha256((out / (name + ('.jsonl' if name == 'rows'
        else '.json'))).read_bytes()).hexdigest() for name in ('config', 'rows', 'summary')
} | {'rows': len(rows), 'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'fine_reranker_justified': summary['fine_reranker_justified']}), flush=True)
