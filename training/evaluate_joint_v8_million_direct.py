"""One frozen, synthetic-subject-held-out direct-pose readout after million-plane training."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_joint_model_v8 import JointSliceFeedbackModel

RUN = ROOT / 'runs/joint_v8_million_direct_tail_001'
DATA = ROOT / 'data/joint_v7_synthetic_dev192_001'
OUTPUT = ROOT / 'runs/joint_v8_million_direct_dev192_001'
SIDE = 192
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def five_points(state, reflection):
    centre, frame, basis = full_frame_state_to_components(state)
    end = (SIDE - 1) / SIDE
    uv = state.new_tensor([[0., 0.], [end, 0.], [0., end], [end, end], [end / 2, end / 2]]) - .5
    uv = uv.expand(*state.shape[:-1], 5, 2).clone()
    uv[..., 0] = torch.where(torch.as_tensor(reflection, device=state.device)[..., None].bool(),
                             end - 1 - uv[..., 0], uv[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi', frame[..., :2] @ basis, uv)


training_done = json.loads((RUN / 'completed.json').read_text())
assert training_done['stage_updates'] == 125000 and training_done['unique_eligible_optimizer_synthetic_planes'] == 1000000
assert training_done['replayed_tail_updates'] == 5000
assert training_done['max_reference_loss_gap'] < .01
data_done = json.loads((DATA / 'completed.json').read_text())
assert data_done['physical_sections'] == 64 and data_done['observations'] == 192
assert sha(DATA / 'protocol.json') == data_done['protocol_sha256']
assert sha(DATA / 'records.jsonl') == data_done['records_sha256']
records = [json.loads(row) for row in (DATA / 'records.jsonl').read_text().splitlines()]
assert len(records) == 64 and len({r['animal_id'] for r in records}) == 4
checkpoint_path = RUN / 'joint_step_161000.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert checkpoint['step'] == 161000 and checkpoint['stage_step'] == 125000 and not checkpoint['calibrated']
config = checkpoint['config']
assert config['resolution'] == [192, 192] and config['trainable_modules'] == ['encoder', 'pose']
assert not {r['animal_id'] for r in records} & {
    r['lineage']['animal_id'] for r in config['synthetic_provenance']['base_subjects']}
repository = Path(__file__).resolve().parents[1]
for name, expected in config['source_sha256'].items():
    assert sha(repository / 'training' / name) == expected, name
model = JointSliceFeedbackModel().cuda().eval()
model.load_state_dict(checkpoint['model_state'], strict=True)
del checkpoint
OUTPUT.mkdir(parents=True, exist_ok=False)
protocol = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha(checkpoint_path),
    'training_completed_sha256': sha(RUN / 'completed.json'),
    'development_completed_sha256': sha(DATA / 'completed.json'),
    'physical_sections': 64, 'observations': 192, 'synthetic_subjects': 4,
    'selection': 'largest direct joint mode/reflection log probability; fixed before evaluation',
    'metric': 'reflection-aware five full-canvas physical points, and antipodal plane-normal angle',
    'aggregation': 'eligible observations averaged within synthetic subject and background mode, then four subjects equally weighted',
    'scope': 'synthetic development only; not biological-animal validation, calibration, public benchmark, or deployment',
    'evaluator_sha256': sha(Path(__file__)),
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()}
(OUTPUT / 'evaluation.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
rows = []
for section_index, record in enumerate(records):
    path = DATA / record['file']
    assert sha(path) == record['sha256']
    with np.load(path) as arrays:
        images = arrays['inputs'].copy()
        truth = torch.from_numpy(arrays['target_state'].copy())
        target_reflection = bool(arrays['reflection'])
        eligible = arrays['eligible'].copy()
    target_points = five_points(truth, target_reflection)
    target_normal = full_frame_state_to_components(truth)[1][:, 2]
    for mode_index, mode in enumerate(record['modes']):
        with torch.inference_mode():
            prediction = model.predict(torch.from_numpy(images[mode_index:mode_index + 1]).cuda())
        state = prediction['state'][0].cpu()
        log_mass = prediction['log_mass'][0].cpu()
        reflection_logit = prediction['reflection_logit'][0].cpu()
        prior = log_mass[:, None] + torch.stack((F.logsigmoid(-reflection_logit),
                                                F.logsigmoid(reflection_logit)), -1)
        chosen_mode, chosen_reflection = divmod(int(prior.flatten().argmax()), 2)
        reflection = torch.arange(2).expand(model.modes, 2)
        error = (five_points(state[:, None].expand(-1, 2, -1), reflection) - target_points).norm(dim=-1).mean(-1)
        normal = full_frame_state_to_components(state)[1][..., :, 2]
        angle = torch.rad2deg((normal * target_normal).sum(-1).abs().clamp(0, 1).acos())
        save = OUTPUT / f'prediction_{section_index:02d}_{mode}.npz'
        np.savez_compressed(save, state=state.numpy(), log_mass=log_mass.numpy(),
            reflection_logit=reflection_logit.numpy(), std=prediction['std'][0].cpu().numpy(),
            concentration=prediction['concentration'][0].cpu().numpy(),
            prior=prior.numpy(), five_point_error_um=error.numpy(), normal_angle_deg=angle.numpy())
        rows.append({'section_id': record['section_id'], 'animal_id': record['animal_id'],
            'subject_id': record['subject_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'], 'mode': mode,
            'eligible': bool(eligible[mode_index]), 'prediction_file': save.name,
            'prediction_sha256': sha(save), 'selected_component': [chosen_mode, chosen_reflection],
            'selected_five_point_um': float(error[chosen_mode, chosen_reflection]),
            'oracle_five_point_um': float(error.min()),
            'selected_normal_deg': float(angle[chosen_mode])})
    print(json.dumps({'event': 'direct_dev192_section', 'sections': section_index + 1,
                      'total_sections': len(records)}), flush=True)
(OUTPUT / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
summary = {}
for mode in records[0]['modes']:
    subjects = {}
    for animal in sorted({r['animal_id'] for r in rows}):
        sample = [r for r in rows if r['animal_id'] == animal and r['mode'] == mode and r['eligible']]
        assert sample
        subjects[animal] = {'eligible_observations': len(sample), **{
            key: float(np.mean([r[key] for r in sample])) for key in
            ('selected_five_point_um', 'oracle_five_point_um', 'selected_normal_deg')}}
    summary[mode] = {'synthetic_subjects': subjects, 'subject_equal': {
        key: float(np.mean([r[key] for r in subjects.values()])) for key in
        ('selected_five_point_um', 'oracle_five_point_um', 'selected_normal_deg')}}
(OUTPUT / 'completed.json').write_text(json.dumps({'sections': len(records), 'observations': len(rows),
    'rows_sha256': sha(OUTPUT / 'rows.jsonl'), 'evaluation_sha256': sha(OUTPUT / 'evaluation.json'),
    'summary': summary, 'calibrated': False, 'scope': protocol['scope']}, indent=2), encoding='utf8')
print(json.dumps({mode: value['subject_equal'] for mode, value in summary.items()}, indent=2), flush=True)
