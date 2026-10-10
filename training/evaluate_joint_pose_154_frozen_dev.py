"""Compare frozen 153 and selected 154 pose proposals on existing synthetic DEV planes."""

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

from training.global_atlas_contrast_090 import rigid_points_090
from training.joint_anatomy_model_153 import JointAnatomyModel153


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


train153 = root / 'runs/joint_anatomy_model_153_train151_002'
train154 = root / 'runs/joint_pose_curriculum_154_train151_001'
panels = {'v3': root / 'data/joint_in_path_correspondence_128_dev_panel',
          'v4': root / 'data/fresh_v4_pose_dev_panel_132'}
out = root / 'runs/joint_pose_154_frozen_dev_eval_001'
assert out.drive.upper() == 'I:' and not out.exists()
done153 = json.loads((train153 / 'completed.json').read_text())
done154 = json.loads((train154 / 'completed.json').read_text())
selected = done154['selected_step']
checkpoints = {'153_parent': train153 / 'joint_step_01300.pt',
               '154_selected': train154 / f'joint_pose_step_{selected:05d}.pt'}
assert done153['selected_step'] == 1300
assert sha(checkpoints['153_parent']) == done153['checkpoint_sha256']['1300']
assert sha(checkpoints['154_selected']) == done154['checkpoint_sha256'][str(selected)]
assert done154['parent_checkpoint_sha256'] == sha(checkpoints['153_parent'])
panel_records = {}
panel_sha = {}
for cohort, panel in panels.items():
    receipt = json.loads((panel / 'completed.json').read_text())
    assert sha(panel / 'records.jsonl') == receipt['records_sha256']
    records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
    assert len(records) == receipt['physical_sections'] == 256
    assert all(row['provenance']['split'] == 'development' for row in records)
    assert all(sha(panel / row['file']) == row['sha256'] for row in records)
    panel_records[cohort] = [row for row in records if row['eligible']]
    assert len(panel_records[cohort]) == receipt['eligible']
    panel_sha[cohort] = sha(panel / 'completed.json')

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = JointAnatomyModel153().cuda().eval().requires_grad_(False)
reflection_options = torch.arange(2, device='cuda')[None, None]
out.mkdir(parents=True, exist_ok=False)
config = {'selected_154_step': selected, 'checkpoint_sha256': {key: sha(path)
    for key, path in checkpoints.items()}, 'panel_receipt_sha256': panel_sha,
    'evaluator_sha256': sha(__file__),
    'model_source_sha256': sha(Path(__file__).with_name('joint_anatomy_model_153.py')),
    'train153_completion_sha256': sha(train153 / 'completed.json'),
    'train154_completion_sha256': sha(train154 / 'completed.json'),
    'metric': 'mean CCF distance over up to 1024 evenly indexed valid native-256 pixels; all 80 modes x two reflections',
    'case_selection': 'reused eligible synthetic DEV sections; not used for 154 checkpoint selection',
    'scope': 'synthetic deformation plans of one Allen template, not new biological animals or expert truth',
    'calibrated': False, 'public_benchmark_used': False, 'final_animals_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for arm, checkpoint in checkpoints.items():
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['step'] == (1300 if arm == '153_parent' else selected)
        model.load_state_dict(state['model'], strict=True)
        del state
        for cohort, panel in panels.items():
            for case in panel_records[cohort]:
                with np.load(panel / case['file']) as sample:
                    image = torch.from_numpy(sample['inputs'][None]).cuda()
                    target = torch.from_numpy(sample['target_state'][None]).cuda()
                    reflection = torch.as_tensor(sample['reflection'], device='cuda').reshape(1)
                    valid = torch.from_numpy(sample['valid_mask']).cuda().bool()
                pixels = valid.flatten().nonzero()[:, 0]
                pixels = pixels[torch.linspace(0, len(pixels) - 1,
                    min(1024, len(pixels)), device='cuda').long()]
                chart = torch.stack((pixels.remainder(256),
                    pixels.div(256, rounding_mode='floor')), -1).float() / 256
                prediction = model.pose.predict(image)
                log_prior = (prediction['log_mass'][..., None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit']),
                    F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
                truth = rigid_points_090(target, reflection, chart)
                errors = []
                for first in range(0, model.pose.modes, 8):
                    states = prediction['state'][:, first:first + 8, None].expand(-1, -1, 2, -1)
                    flags = reflection_options.expand(1, 8, 2)
                    points = rigid_points_090(states, flags, chart[None, None])
                    errors.append(((points - truth[:, None, None]).norm(dim=-1)
                                   .mean(-1) / 1000).flatten())
                error = torch.cat(errors)
                order = log_prior.argsort(descending=True)
                normal = np.abs(case['plane_normal_ap_dv_ml'])
                row = {'arm': arm, 'cohort': cohort, 'section_id': case['section_id'],
                    'physical_section_id': case['panel_physical_section_id'],
                    'synthetic_plan_id': case['synthetic_subject_plan_id'],
                    'appearance_mode': case['appearance_mode'],
                    'valid_fraction': float(valid.float().mean()),
                    'nearest_axis': ('AP', 'DV', 'ML')[int(np.argmax(normal))],
                    'obliquity_deg': float(np.degrees(np.arccos(float(np.max(normal))))),
                    'error_mm_all160': error.cpu().tolist(),
                    'log_prior_all160_uncalibrated': log_prior.cpu().tolist(),
                    'top1_mm': float(error[order[0]]),
                    'top2_best_mm': float(error[order[:2]].min()),
                    'top16_best_mm': float(error[order[:16]].min()),
                    'oracle_best_mm': float(error.min())}
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        print(json.dumps({'arm_complete': arm, 'cases': sum(r['arm'] == arm for r in rows)}), flush=True)

summary = {}
for arm in checkpoints:
    for cohort in panels:
        group = [row for row in rows if row['arm'] == arm and row['cohort'] == cohort]
        plans = sorted({row['synthetic_plan_id'] for row in group})
        summary[f'{arm}/{cohort}'] = {'n': len(group), 'synthetic_plans': len(plans)}
        for metric in ('top1_mm', 'top2_best_mm', 'top16_best_mm', 'oracle_best_mm'):
            summary[f'{arm}/{cohort}'][metric] = float(np.mean([
                np.mean([row[metric] for row in group if row['synthetic_plan_id'] == plan])
                for plan in plans]))
            summary[f'{arm}/{cohort}'][metric.replace('_mm', '_within1p5')] = float(np.mean([
                np.mean([row[metric] <= 1.5 for row in group
                         if row['synthetic_plan_id'] == plan]) for plan in plans]))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'config_sha256': sha(out / 'config.json'),
    'rows_sha256': sha(out / 'rows.jsonl'), 'summary_sha256': sha(out / 'summary.json'),
    'case_rows': len(rows), 'calibrated': False, 'public_benchmark_used': False,
    'final_animals_used': False}, indent=2))
print(json.dumps({'event': 'evaluation_complete', 'summary': summary}), flush=True)
