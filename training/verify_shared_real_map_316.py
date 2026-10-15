"""Read-only receipt and raw-row audit of the frozen 316 pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('I:/AnatomyTracker')
SOURCE = Path(__file__).resolve()
RUN = ROOT / 'runs/shared_real_map_316_001'
OUT = RUN / 'readout'
PROTOCOL = SOURCE.parents[1] / 'docs/publication/SHARED_REAL_MAP_316_PROTOCOL_20261015.md'
assert SOURCE.drive.upper() == RUN.drive.upper() == 'I:'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
done = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
readout = json.loads((OUT / 'config.json').read_text(encoding='utf8'))
readout_done = json.loads((OUT / 'completed.json').read_text(encoding='utf8'))
summary = json.loads((OUT / 'summary.json').read_text(encoding='utf8'))
assert config['gradient_animals'] == ['CC2B', 'CC4B', 'CC4A']
assert config['guard_animal'] == 'AL1A' and config['batches_per_arm'] == 600
assert config['protocol_sha256'] == done['protocol_sha256'] == sha(PROTOCOL)
assert done['config_sha256'] == sha(RUN / 'config.json')
assert config['source_sha256']['train_shared_real_map_316.py'] == sha(
    SOURCE.with_name('train_shared_real_map_316.py'))
assert readout['script_sha256'] == readout_done['script_sha256'] == sha(
    SOURCE.with_name('readout_shared_real_map_316.py'))
assert readout_done['protocol_sha256'] == sha(PROTOCOL)
for name, digest in config['source_sha256'].items():
    assert digest == sha(SOURCE.with_name(name))
inputs = {'protocol': PROTOCOL, 'split': SOURCE.with_name('lupori_animal_split_185.json'),
    'panel': ROOT / 'runs/first_choice_exposure_309_001/panels.pt',
    'panel_draws': ROOT / 'runs/first_choice_exposure_309_001/panel_draws.jsonl',
    'panel_config': ROOT / 'runs/first_choice_exposure_309_001/config.json',
    'parent_314_rows': ROOT / 'reports/acquired_map_314_AL1A/rows.jsonl',
    'run_config': RUN / 'config.json', 'run_completed': RUN / 'completed.json',
    'parent_309_checkpoint': ROOT /
        'runs/first_choice_exposure_309_001/single/pose_local_20000.pt'}
for arm in ('map_only', 'shared_encoder'):
    for name in ('draws.jsonl', 'training.jsonl', 'completed.json'):
        inputs[f'{arm}_{name}'] = RUN / arm / name
    inputs[f'{arm}_checkpoint'] = RUN / arm / 'map_step_00600.pt'
assert set(inputs) == set(readout['input_sha256'])
for name, path in inputs.items():
    assert readout['input_sha256'][name] == sha(path), (name, path)
for name, digest in readout_done['output_sha256'].items():
    assert digest == sha(OUT / name)

draws = []
for arm in ('map_only', 'shared_encoder'):
    arm_path = RUN / arm
    receipt = json.loads((arm_path / 'completed.json').read_text(encoding='utf8'))
    assert done['arm_completion_sha256'][arm] == sha(arm_path / 'completed.json')
    assert receipt['config_sha256'] == sha(RUN / 'config.json')
    assert receipt['draws_sha256'] == sha(arm_path / 'draws.jsonl')
    assert receipt['training_sha256'] == sha(arm_path / 'training.jsonl')
    for step, digest in receipt['checkpoint_sha256'].items():
        assert digest == sha(arm_path / f'map_step_{int(step):05d}.pt')
    rows = [json.loads(line) for line in (arm_path / 'draws.jsonl').read_text(
        encoding='utf8').splitlines()]
    training = [json.loads(line) for line in (arm_path / 'training.jsonl').read_text(
        encoding='utf8').splitlines()]
    assert len(rows) == len(training) == 600
    assert [row['step'] for row in training] == list(range(1, 601))
    assert [row['step'] for row in rows] == list(range(1, 601))
    assert all(row['real']['animal_id'] in config['gradient_animals']
               and row['synthetic_virtual'] // 64 < 56 for row in rows)
    assert all(np.isfinite([row['real_map_mm'], row['synthetic_map_mm'],
                            row['loss'], row['gradient_norm']]).all() for row in training)
    draws.append(sha(arm_path / 'draws.jsonl'))
assert draws[0] == draws[1] and done['paired_draws_identical']

real = [json.loads(line) for line in (OUT / 'al1a_rows.jsonl').read_text(
    encoding='utf8').splitlines()]
panel = [json.loads(line) for line in (OUT / 'panel_rows.jsonl').read_text(
    encoding='utf8').splitlines()]
old = {row['section_id']: row for row in (json.loads(line) for line in
    (ROOT / 'reports/acquired_map_314_AL1A/rows.jsonl').read_text(
        encoding='utf8').splitlines())}
assert len(real) == 52 and {row['section_id'] for row in real} == set(old)
assert len(panel) == 64 and [row['panel_slot'] for row in panel] == list(range(64))
assert [row['base'] for row in panel] == [56 + index % 8 for index in range(64)]
for row in real:
    assert row['animal_id'] == 'AL1A' and row['specimen_id'] == 'AL1A'
    for name in ('zero', 'parent_309'):
        assert abs(row['rms_mm'][name] - old[row['section_id']]['rms_mm'][name]) < 1e-5
for name in ('zero', 'parent_309', 'map_only', 'shared_encoder'):
    values = np.asarray([row['rms_mm'][name] for row in real])
    assert abs(values.mean() - summary['al1a'][name]['map_rms_mean_mm']) < 1e-10
    if name == 'zero':
        continue
    pose = np.asarray([row['blind_first_five_site_gap_mm'][name] for row in real])
    exact = np.asarray([row['exact_pose_map_mm'][name] for row in panel])
    blind = np.asarray([row['blind_first_rigid_mm'][name] for row in panel])
    permuted = np.asarray([row['permuted_image_rms_mm'][name] for row in real])
    assert abs(pose.mean() - summary['al1a'][name][
        'blind_first_five_site_gap_mean_mm']) < 1e-10
    assert abs(exact.mean() - summary['panel'][name]['exact_pose_map_mean_mm']) < 1e-10
    assert abs(blind.mean() - summary['panel'][name]['blind_first_rigid_mean_mm']) < 1e-10
    assert int((blind > 5).sum()) == summary['panel'][name]['blind_first_gt5mm']
    assert abs((permuted.mean() - values.mean()) - summary['image_dependence'][name][
        'permuted_minus_original_mean_mm']) < 1e-10
    assert int((values < permuted).sum()) == summary['image_dependence'][name][
        'original_better_sections']
assert all(not summary['predeclared_gate'][arm]['promising']
           for arm in ('map_only', 'shared_encoder'))
assert not summary['predeclared_gate']['shared_encoder']['shared_encoder_attribution']
print(json.dumps({'verified': True, 'paired_draws_identical': True,
    'al1a_sections': len(real), 'panel_cases': len(panel),
    'map_rms_mean_mm': {name: summary['al1a'][name]['map_rms_mean_mm']
                        for name in ('zero', 'parent_309', 'map_only', 'shared_encoder')},
    'promotion_gate_passed': False}))
