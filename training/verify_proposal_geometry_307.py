"""Independent post-exit receipt, identity and gate check for 307."""

import hashlib
import json
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
RUN = ROOT / 'runs/proposal_geometry_307_001'
SOURCE = Path(__file__).with_name('train_proposal_geometry_307.py')
PROTOCOL = SOURCE.parents[1] / 'docs/publication/PROPOSAL_GEOMETRY_307_PROTOCOL_20261014.md'
PARENT = ROOT / 'runs/joint_spatial_305_acquired_rehearsal_001/intervention/pose_local_00400.pt'
completed = json.loads((RUN / 'completed.json').read_text(encoding='utf8'))
config = json.loads((RUN / 'config.json').read_text(encoding='utf8'))
summary = json.loads((RUN / 'summary.json').read_text(encoding='utf8'))

assert completed['intervention_recovered_after_pretraining_device_error']
assert completed['control_source_sha256'] == config['source_sha256']
assert completed['source_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
assert completed['protocol_sha256'] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
with PARENT.open('rb') as stream:
    assert config['parent_sha256'] == hashlib.file_digest(stream, 'sha256').hexdigest()
for name, expected in completed['output_sha256'].items():
    with (RUN / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
assert completed['config_sha256'] == completed['output_sha256']['config.json']
assert completed['summary'] == summary

ids = set()
counts = {}
for name in ('panel_draws.jsonl', 'control/draws.jsonl', 'intervention/draws.jsonl'):
    rows = [json.loads(line) for line in (RUN / name).open(encoding='utf8')]
    counts[name] = sum(row['accepted'] for row in rows)
    for row in rows:
        section_id = row['provenance']['physical_section_id']
        assert section_id not in ids
        ids.add(section_id)
assert counts == {'panel_draws.jsonl': 192,
                  'control/draws.jsonl': 1600,
                  'intervention/draws.jsonl': 1600}

for arm in ('control', 'intervention'):
    rows = [json.loads(line) for line in (RUN / arm / 'readouts.jsonl').open(encoding='utf8')]
    assert len(rows) == 3 * 2 * 96
    for step in (0, 400, 800):
        for version in ('v9', 'v10'):
            for partition, expected_n in (('natural', 64), ('sparse', 16), ('full', 16)):
                group = [row for row in rows if row['step'] == step
                    and row['version'] == version and row['partition'] == partition]
                assert len(group) == expected_n
                metric = summary['readouts'][arm][str(step)][version][partition]
                assert metric['cases'] == expected_n
                assert metric['all_capture_le1p5mm'] == sum(
                    row['all_best_site_mm'] <= 1.5 for row in group)
                assert metric['top16_capture_le1p5mm'] == sum(
                    row['top16_best_site_mm'] <= 1.5 for row in group)
                assert abs(metric['first_mean_site_mm'] - sum(
                    row['first_site_mm'] for row in group) / len(group)) < 1e-10

c = summary['readouts']['control']['800']
i = summary['readouts']['intervention']['800']
initial = summary['readouts']['intervention']['0']
gate = {'v10_natural_gain_vs_control_ge6': i['v10']['natural']['all_capture_le1p5mm'] - c['v10']['natural']['all_capture_le1p5mm'] >= 6,
        'v10_natural_gain_vs_initial_ge4': i['v10']['natural']['all_capture_le1p5mm'] - initial['v10']['natural']['all_capture_le1p5mm'] >= 4,
        'v9_natural_loss_vs_control_le2': i['v9']['natural']['all_capture_le1p5mm'] >= c['v9']['natural']['all_capture_le1p5mm'] - 2,
        'v10_sparse_gain_ge1': i['v10']['sparse']['all_capture_le1p5mm'] >= c['v10']['sparse']['all_capture_le1p5mm'] + 1,
        'v10_full_gain_ge1': i['v10']['full']['all_capture_le1p5mm'] >= c['v10']['full']['all_capture_le1p5mm'] + 1,
        'v9_first_nonregression': i['v9']['natural']['first_mean_site_mm'] <= c['v9']['natural']['first_mean_site_mm'] + .2,
        'v10_first_nonregression': i['v10']['natural']['first_mean_site_mm'] <= c['v10']['natural']['first_mean_site_mm'] + .2}
assert summary['gate'] == gate and summary['pass'] is False and not all(gate.values())
print(json.dumps({'verified': True, 'accepted': counts, 'gate': gate, 'pass': False}))
