"""Frozen v7 DEV image-only pose comparison: new stage 249 versus 247."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
BASELINE = ROOT / 'runs/one_shot_v7_heldout_dev_panel_248_readout_001'
TRAIN = ROOT / 'runs/pose_stage1_signal_249_train_001'
OUT = ROOT / 'runs/pose_stage1_signal_249_readout_001'
DOC = Path('docs/publication/POSE_STAGE1_SIGNAL_249_READOUT_PROTOCOL_20261012.md')
SIDE, STEP = 192, 3000
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                (SIDE - 1) / SIDE - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('...ij,...pj->...pi',
                                               frame[..., :2] @ basis, chart - .5)


assert (PANEL / 'completed.json').is_file() and (BASELINE / 'completed.json').is_file()
assert (TRAIN / 'completed.json').is_file(), '249 training has not completed'
panel_done = json.loads((PANEL / 'completed.json').read_text())
base_done = json.loads((BASELINE / 'completed.json').read_text())
train_done = json.loads((TRAIN / 'completed.json').read_text())
assert sha(PANEL / 'records.jsonl') == panel_done['records_sha256']
assert sha(BASELINE / 'rows.jsonl') == base_done['rows_sha256']
assert sha(TRAIN / 'training.jsonl') == train_done['training_sha256']
assert sha(TRAIN / 'config.json') == train_done['config_sha256']
assert sha(TRAIN / f'joint_step_{STEP:05d}.pt') == train_done['checkpoint_sha256'][str(STEP)]
config = json.loads((TRAIN / 'config.json').read_text())
assert config['protocol_sha256'] == sha(Path('docs/publication/POSE_STAGE1_SIGNAL_249_PROTOCOL_20261012.md'))
assert config['source_sha256']['arbitrary_plane_one_shot_model.py'] == sha(Path(__file__).parent / 'arbitrary_plane_one_shot_model.py')
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
base_rows = {row['physical_section_id']: row for row in
             (json.loads(line) for line in (BASELINE / 'rows.jsonl').open(encoding='utf8'))}
eligible = [row for row in records if row['eligible']]
assert len(eligible) == panel_done['eligible'] == len(base_rows)
assert {row['provenance']['physical_section_id'] for row in eligible} == set(base_rows)
assert all(sha(PANEL / row['file']) == row['sha256'] for row in records)
assert torch.cuda.is_available()

model = OneShotJointSliceModel(
    modes=0, normal_anchor_count=64, shared_anchor_pose=True,
    atlas_conditioning=True, vector_refinement=True, fitted_ranking=True,
).cuda().eval()
checkpoint = torch.load(TRAIN / f'joint_step_{STEP:05d}.pt', map_location='cpu', weights_only=True)
assert checkpoint['step'] == STEP and not checkpoint['calibrated']
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
full_chart = torch.stack(torch.meshgrid(torch.arange(SIDE, device='cuda'),
                                       torch.arange(SIDE, device='cuda'), indexing='ij'), -1)[..., [1, 0]].float() / SIDE

OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in eligible:
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        assert int(valid.sum()) == record['valid_pixels']
        prediction = model.predict(image)
        modes = prediction['log_mass'][0].topk(4).indices
        mode_indices = modes.repeat_interleave(2)
        reflections = torch.tensor([[0, 1] * 4], device='cuda')
        prior = (prediction['log_mass'][0, mode_indices]
                 + torch.where(reflections[0].bool(),
                               F.logsigmoid(prediction['reflection_logit'][0, mode_indices]),
                               F.logsigmoid(-prediction['reflection_logit'][0, mode_indices])))
        selected = int(prior.argmax())
        state = prediction['state'][:, mode_indices[selected:selected + 1]]
        reflection = reflections[:, selected:selected + 1]
        chart = full_chart[valid]
        target = target[valid]
        rigid = (points(state, reflection, chart) - target).norm(dim=-1).mean()
        axis = full_frame_state_to_components(state)[1][0, 0, :, 2]
        true_axis = full_frame_state_to_components(truth)[1][0, :, 2]
        axis_error = torch.rad2deg(torch.acos((axis * true_axis).sum().abs().clamp_max(1)))
        cut_axis = torch.tensor(record['provenance']['virtual_unit_normal'], device='cuda')
        cut_angle = torch.rad2deg(torch.acos(cut_axis.abs().max().clamp_max(1)))
        old = base_rows[record['provenance']['physical_section_id']]
        assert old['panel_file_sha256'] == record['sha256']
        row = {'physical_section_id': record['provenance']['physical_section_id'],
               'animal_id': record['animal_id'], 'section_id': record['section_id'],
               'panel_file_sha256': record['sha256'], 'appearance_mode': record['appearance_mode'],
               'valid_fraction': record['valid_pixels'] / SIDE**2,
               'cut_nearest_principal_axis_deg': float(cut_angle),
               'stage1_selected_rigid_um': float(rigid),
               'stage1_selected_normal_axis_error_deg': float(axis_error),
               'baseline247_selected_rigid_um': old['prior_top1_rigid_um'],
               'baseline247_selected_normal_axis_error_deg': old['prior_top1_normal_axis_error_deg'],
               'stage1_selected_mode': int(mode_indices[selected]),
               'stage1_selected_reflection': int(reflection[0, 0])}
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'evaluated': len(rows), 'eligible': len(eligible)}), flush=True)

keys = ('stage1_selected_rigid_um', 'stage1_selected_normal_axis_error_deg',
        'baseline247_selected_rigid_um', 'baseline247_selected_normal_axis_error_deg')
identities = sorted({row['animal_id'] for row in rows})
assert len(identities) == panel_done['synthetic_subjects'] == 8
subject_means = {identity: {key: float(np.mean([row[key] for row in rows
                                               if row['animal_id'] == identity])) for key in keys}
                 for identity in identities}
equal = {key: float(np.mean([subject_means[identity][key] for identity in identities])) for key in keys}
last = [json.loads(line)['normal_ce'] for line in
        (TRAIN / 'training.jsonl').open(encoding='utf8')]
assert len(last) == train_done['batches'] == STEP
last500 = float(np.mean(last[-500:]))
normal_gain = equal['baseline247_selected_normal_axis_error_deg'] - equal['stage1_selected_normal_axis_error_deg']
rigid_gain_mm = (equal['baseline247_selected_rigid_um'] - equal['stage1_selected_rigid_um']) / 1000


def group_mean(group):
    return {'cases': len(group),
            'case_mean': {key: float(np.mean([row[key] for row in group])) for key in keys} if group else {}}


summary = {'status': 'development mechanism readout, not a deployable or calibrated model',
           'cases': len(rows), 'synthetic_subjects': len(identities),
           'subject_equal_mean': equal, 'subject_means': subject_means,
           'train_normal_ce_last500': last500, 'uniform_normal_ce': math.log(64),
           'normal_axis_gain_deg': normal_gain, 'rigid_gain_mm': rigid_gain_mm,
           'stage1_continuation_gate': bool(last500 <= math.log(64) - .5
                                             and normal_gain >= 10 and rigid_gain_mm >= .5),
           'physical_cut_groups': {f'{low}-{high}': group_mean([row for row in rows
                                    if low <= row['cut_nearest_principal_axis_deg'] < high])
                                   for low, high in ((0, 15), (15, 30), (30, 45), (45, 60))},
           'valid_tissue_fraction_groups': {f'{low:.1f}-{high:.1f}': group_mean([row for row in rows
                                    if low <= row['valid_fraction'] < high])
                                   for low, high in ((0, .1), (.1, .3), (.3, .5), (.5, 1.01))}}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'cases': len(rows), 'checkpoint_step': STEP,
    'checkpoint_sha256': sha(TRAIN / f'joint_step_{STEP:05d}.pt'),
    'training_completed_sha256': sha(TRAIN / 'completed.json'),
    'baseline_completed_sha256': sha(BASELINE / 'completed.json'),
    'panel_completed_sha256': sha(PANEL / 'completed.json'),
    'readout_protocol_sha256': sha(DOC),
    'source_sha256': {name: sha(Path(__file__).parent / name) for name in
                      (Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
                       'arbitrary_plane_full_frame_primitives.py')},
    'rows_sha256': sha(OUT / 'rows.jsonl'), 'summary_sha256': sha(OUT / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'cases': len(rows),
                  'stage1_continuation_gate': summary['stage1_continuation_gate']}), flush=True)
