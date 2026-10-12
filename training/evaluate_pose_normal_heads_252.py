"""Same-case v7 DEV normal-axis readout of frozen 247/249/250/251 heads."""
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

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

PANEL = ROOT / 'data/one_shot_v7_heldout_dev_panel_248_004'
OUT = ROOT / 'runs/pose_normal_heads_252_readout_001'
DOC = Path('docs/publication/POSE_NORMAL_HEADS_252_READOUT_PROTOCOL_20261012.md')
RUNS = {
    '247': (ROOT / 'runs/anchor_only_blind_feedback_247_train_001', 6000, True),
    '249': (ROOT / 'runs/pose_stage1_signal_249_train_001', 3000, True),
    '250': (ROOT / 'runs/pose_ce_only_250_train_001', 3000, True),
    '251': (ROOT / 'runs/pose_direct_normal_251_train_001', 3000, False),
}
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert (PANEL / 'completed.json').is_file(), '248 panel must be frozen first'
panel_done = json.loads((PANEL / 'completed.json').read_text())
assert sha(PANEL / 'records.jsonl') == panel_done['records_sha256']
assert sha(PANEL / 'protocol.json') == panel_done['protocol_sha256']
records = [json.loads(line) for line in (PANEL / 'records.jsonl').open(encoding='utf8')]
assert len(records) == panel_done['physical_sections'] == 256
assert len({r['provenance']['physical_section_id'] for r in records}) == 256
assert all(sha(PANEL / r['file']) == r['sha256'] for r in records)
eligible = [r for r in records if r['eligible'] and r['valid_pixels'] > 0]
assert len(eligible) == panel_done['eligible']

models, checkpoint_sha, training_sha, train_ce = {}, {}, {}, {}
for name, (run, step, shared) in RUNS.items():
    done = json.loads((run / 'completed.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    checkpoint = run / f'joint_step_{step:05d}.pt'
    assert sha(run / 'config.json') == done['config_sha256']
    assert sha(run / 'training.jsonl') == done['training_sha256']
    assert sha(checkpoint) == done['checkpoint_sha256'][str(step)]
    assert sha(Path('training/arbitrary_plane_one_shot_model.py')) == config['source_sha256']['arbitrary_plane_one_shot_model.py']
    checkpoint_sha[name], training_sha[name] = sha(checkpoint), sha(run / 'completed.json')
    if name != '247':
        protocol = {'249': 'POSE_STAGE1_SIGNAL_249_PROTOCOL_20261012.md',
                    '250': 'POSE_CE_ONLY_250_PROTOCOL_20261012.md',
                    '251': 'POSE_DIRECT_NORMAL_251_PROTOCOL_20261012.md'}[name]
        assert sha(Path('docs/publication') / protocol) == config['protocol_sha256']
        values = [json.loads(line)['normal_ce'] for line in
                  (run / 'training.jsonl').open(encoding='utf8')]
        assert len(values) == done['batches'] == step
        train_ce[name] = float(np.mean(values[-500:]))
    model = OneShotJointSliceModel(
        modes=0, normal_anchor_count=64, shared_anchor_pose=shared,
        atlas_conditioning=True, vector_refinement=True, fitted_ranking=True,
    ).cuda().eval()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['step'] == step and not saved['calibrated']
    model.load_state_dict(saved['model'], strict=True)
    models[name] = model
    del saved
assert torch.cuda.is_available()

OUT.mkdir(parents=True, exist_ok=False)
rows = []
with torch.inference_mode(), (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for record in eligible:
        with np.load(PANEL / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
        truth = full_frame_state_to_components(state)[1][0, :, 2]
        row = {'physical_section_id': record['provenance']['physical_section_id'],
               'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
               'experiment_id': record['experiment_id'], 'section_id': record['section_id'],
               'panel_file_sha256': record['sha256'], 'appearance_mode': record['appearance_mode'],
               'valid_fraction': record['valid_pixels'] / 192**2,
               'physical_cut_nearest_principal_deg': float(torch.rad2deg(torch.acos(
                   torch.tensor(record['provenance']['virtual_unit_normal'], device='cuda').abs().max().clamp_max(1)))),
               'axis_error_deg': {}, 'top_anchor': {}}
        for name, model in models.items():
            prediction = model.predict(image)
            index = int(prediction['log_mass'][0].argmax())
            axis = full_frame_state_to_components(prediction['state'][:, index:index + 1])[1][0, 0, :, 2]
            error = torch.rad2deg(torch.acos((axis * truth).sum().abs().clamp_max(1)))
            row['axis_error_deg'][name] = float(error)
            row['top_anchor'][name] = index
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 32 == 0:
            stream.flush()
            print(json.dumps({'evaluated': len(rows), 'eligible': len(eligible)}), flush=True)

animals = sorted({row['animal_id'] for row in rows})
assert len(animals) == panel_done['synthetic_subjects'] == 8
subject_mean = {animal: {name: float(np.mean([row['axis_error_deg'][name] for row in rows
                                              if row['animal_id'] == animal])) for name in RUNS}
                for animal in animals}
equal = {name: float(np.mean([subject_mean[a][name] for a in animals])) for name in RUNS}
groups = {}
for label, subset in {
    **{f'cut_{low}_{high}': [r for r in rows if low <= r['physical_cut_nearest_principal_deg'] < high]
       for low, high in ((0, 15), (15, 30), (30, 45), (45, 60))},
    **{f'valid_{low}_{high}': [r for r in rows if low <= r['valid_fraction'] < high]
       for low, high in ((0, .1), (.1, .3), (.3, .5), (.5, 1.01))},
}.items():
    groups[label] = {'cases': len(subset), 'case_mean_deg':
                     {name: float(np.mean([r['axis_error_deg'][name] for r in subset]))
                      for name in RUNS} if subset else {}}
summary = {
    'status': 'synthetic DEV mechanism comparison only; no deployable or calibrated model',
    'eligible_cases': len(rows), 'synthetic_subjects': len(animals),
    'subject_equal_axis_error_deg': equal, 'subject_axis_error_deg': subject_mean,
    'train_normal_ce_last500': train_ce, 'uniform_normal_ce': math.log(64),
    'gain_251_over_249_deg': equal['249'] - equal['251'],
    'gain_251_over_250_deg': equal['250'] - equal['251'],
    'gate_250': bool(train_ce['250'] <= math.log(64) - .5
                     and equal['249'] - equal['250'] >= 10),
    'gate_251': bool(train_ce['251'] <= math.log(64) - .5
                     and equal['249'] - equal['251'] >= 10
                     and equal['250'] - equal['251'] >= 10),
    'strata': groups,
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(OUT / 'completed.json').write_text(json.dumps({
    'eligible_cases': len(rows), 'panel_completed_sha256': sha(PANEL / 'completed.json'),
    'readout_protocol_sha256': sha(DOC), 'checkpoint_sha256': checkpoint_sha,
    'training_completed_sha256': training_sha,
    'source_sha256': {name: sha(Path('training') / name) for name in
                      (Path(__file__).name, 'arbitrary_plane_one_shot_model.py',
                       'arbitrary_plane_full_frame_primitives.py')},
    'rows_sha256': sha(OUT / 'rows.jsonl'), 'summary_sha256': sha(OUT / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
}, indent=2), encoding='utf8')
print(json.dumps({'event': 'complete', 'cases': len(rows),
                  'gate_250': summary['gate_250'], 'gate_251': summary['gate_251']}), flush=True)
