"""Truth-blind top-16 anatomy-fit ranking on preselected synthetic DEV sections."""

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

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_156 import JointAnatomyModel156


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


train = root / 'runs/joint_feedback_158_train151_001'
panel = root / 'data/fresh_v4_pose_dev_panel_132'
out = root / 'runs/joint_feedback_159_blind_fit_dev_001'
assert out.drive.upper() == 'I:' and not out.exists()
done = json.loads((train / 'completed.json').read_text())
panel_done = json.loads((panel / 'completed.json').read_text())
assert sha(panel / 'records.jsonl') == panel_done['records_sha256']
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
plans = sorted({record['synthetic_subject_plan_id'] for record in records})
assert len(plans) == 8 and len(records) == 256
selected = []
for plan in plans:
    eligible = [record for record in records if record['synthetic_subject_plan_id'] == plan
                and record['eligible']]
    selected += sorted(eligible, key=lambda record:
        hashlib.sha256(record['section_id'].encode()).hexdigest())[:8]
assert len(selected) == 64

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas_array, _ = _decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
atlas_zero = atlas.clone()
atlas_zero[0].zero_()
model = JointAnatomyModel156().cuda().eval().requires_grad_(False)
checkpoints = [('baseline', 'feedback', 0), ('feedback_1000', 'feedback', 1000),
               ('control_1000', 'detached_control', 1000)]
config = {'training_completion_sha256': sha(train / 'completed.json'),
          'panel_completion_sha256': sha(panel / 'completed.json'),
          'section_ids': [record['section_id'] for record in selected],
          'selection': 'eight eligible sections per synthetic plan, smallest SHA-256(section_id); no model outputs used',
          'checkpoints': {}, 'source_sha256': sha(__file__),
          'ranking': 'unaltered model score = direct log mass - fitted energy; no DEV-tuned weight',
          'atlas_zero': 'out-of-distribution intensity-zero diagnostic, support retained; not a matched support-only model',
          'scope': 'reused v4 synthetic DEV plans from one atlas, not independent biological animals',
          'calibrated': False, 'final_animals_used': False, 'public_benchmark_used': False}
out.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for label, arm, step in checkpoints:
        checkpoint = train / arm / f'joint_feedback_step_{step:05d}.pt'
        assert sha(checkpoint) == done['arms'][arm]['checkpoint_sha256'][str(step)]
        config['checkpoints'][label] = sha(checkpoint)
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'], strict=True)
        del saved
        for record in selected:
            assert sha(panel / record['file']) == record['sha256']
            with np.load(panel / record['file'], allow_pickle=False) as sample:
                image = torch.from_numpy(sample['inputs'][None].copy()).cuda()
                state = torch.from_numpy(sample['target_state'][None].copy()).cuda()
                reflection = torch.as_tensor(sample['reflection'], device='cuda').reshape(1)
                offsets = torch.from_numpy(sample['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(sample['weights'][None].copy()).cuda()
                valid = torch.from_numpy(sample['valid_mask'].copy()).cuda().bool()
            pixels = valid.flatten().nonzero()[:, 0]
            pixels = pixels[torch.linspace(0, len(pixels) - 1, min(1024, len(pixels)),
                device='cuda').long()]
            chart = torch.stack((pixels.remainder(256),
                pixels.div(256, rounding_mode='floor')), -1).float() / 256
            prediction = model.pose.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
            branch = prior.topk(16).indices
            mode = (branch // 2)[None]
            flags = (branch % 2)[None]
            truth = rigid_points_090(state, reflection, chart)
            proposal = rigid_points_090(prediction['state'][:, mode[0]], flags, chart[None])
            original_error = ((proposal - truth[:, None]).norm(dim=-1).mean(-1)[0] / 1000)
            scores, zero_scores, corrected_error = [], [], []
            for first in range(0, 16, 4):
                choice = mode[:, first:first + 4]
                mirror = flags[:, first:first + 4]
                fitted = model(image, atlas, offsets, weights, choice, mirror)
                scores.append(fitted['score_uncalibrated'][0])
                corrected = rigid_points_090(fitted['corrected_state'], mirror, chart[None])
                corrected_error.append((corrected - truth[:, None]).norm(dim=-1).mean(-1)[0] / 1000)
                zeroed = model(image, atlas_zero, offsets, weights, choice, mirror)
                zero_scores.append(zeroed['score_uncalibrated'][0])
            score = torch.cat(scores)
            score_zero = torch.cat(zero_scores)
            postfit = torch.cat(corrected_error)
            fit_choice = int(score.argmax())
            zero_choice = int(score_zero.argmax())
            row = {'checkpoint': label, 'section_id': record['section_id'],
                'physical_section_id': record['panel_physical_section_id'],
                'synthetic_plan_id': record['synthetic_subject_plan_id'],
                'appearance_mode': record['appearance_mode'],
                'valid_fraction': float(valid.float().mean()),
                'obliquity_deg': float(np.degrees(np.arccos(
                    np.max(np.abs(record['plane_normal_ap_dv_ml']))))),
                'branch_ids': branch.tolist(), 'prior_log_mass': prior[branch].tolist(),
                'original_error_mm': original_error.tolist(),
                'corrected_error_mm': postfit.tolist(),
                'fit_score_uncalibrated': score.tolist(),
                'zero_intensity_score_uncalibrated': score_zero.tolist(),
                'direct_selected_mm': float(original_error[0]),
                'fit_selected_original_mm': float(original_error[fit_choice]),
                'fit_selected_corrected_mm': float(postfit[fit_choice]),
                'zero_intensity_selected_original_mm': float(original_error[zero_choice]),
                'top16_oracle_mm': float(original_error.min()),
                'direct_selected_near': bool(original_error[0] <= 1.5),
                'fit_selected_near': bool(original_error[fit_choice] <= 1.5),
                'top16_oracle_near': bool(original_error.min() <= 1.5)}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'checkpoint': label, 'sections_done': len(selected)}), flush=True)

summary = {}
for label, _, _ in checkpoints:
    group = [row for row in rows if row['checkpoint'] == label]
    result = {'sections': len(group), 'synthetic_plans': len(plans)}
    for metric in ('direct_selected_mm', 'fit_selected_original_mm',
                   'fit_selected_corrected_mm', 'zero_intensity_selected_original_mm',
                   'top16_oracle_mm', 'direct_selected_near', 'fit_selected_near',
                   'top16_oracle_near'):
        result[metric] = float(np.mean([np.mean([row[metric] for row in group
            if row['synthetic_plan_id'] == plan]) for plan in plans]))
    result['fit_selection_gain_mm'] = result['direct_selected_mm'] - result['fit_selected_original_mm']
    summary[label] = result
(out / 'config.json').write_text(json.dumps(config, indent=2))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'final_animals_used': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'blind_fit_complete', 'summary': summary}), flush=True)
