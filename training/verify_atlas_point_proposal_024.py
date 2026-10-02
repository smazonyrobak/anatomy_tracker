"""Independent provenance and frozen-result verifier for point-proposal pilot 024."""
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from scipy.spatial import cKDTree

root = Path('I:/AnatomyTracker')
run = root / 'runs/atlas_point_proposal_024'
evaluation = root / 'runs/atlas_point_proposal_024_development_eval'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((run / 'config.json').read_text())
completed = json.loads((run / 'completed.json').read_text())
assert config['updates'] == completed['updates'] == 3000
assert config['sections_per_batch'] == 2 and config['points_per_section'] == 32
assert completed['accepted_synthetic_sections'] == 6000 and completed['point_pairs'] == 192000
assert config['random_initialization'] and not config['prior_model_weights_features_pseudolabels']
assert config['real_training_images'] == 0
assert not completed['calibrated'] and not completed['public_benchmark_used']
for name in ('config', 'draws', 'training'):
    path = run / ('config.json' if name == 'config' else name + '.jsonl')
    assert sha(path) == completed[name + '_sha256']
for name, expected in config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == expected
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
used = [row for row in draws if row['used']]
assert len(used) == 6000
assert {(row['step'], row['slot']) for row in used} == {
    (step, slot) for step in range(1, 3001) for slot in range(2)}
development_animals = {json.loads(line)['animal_id'] for line in (panel / 'records.jsonl').open()}
assert all(row['split'] == 'train' and row['base_lineage']['split'] == 'train' for row in used)
assert not development_animals & {row['base_lineage']['animal_id'] for row in used}
training = [json.loads(line) for line in (run / 'training.jsonl').open()]
assert [row['step'] for row in training] == list(range(1, 3001))
assert np.isfinite([row['loss'] for row in training]).all()
assert np.isfinite([row['gradient_norm'] for row in training]).all()
assert training[0]['gradient_norm'] > 0
checkpoints = [torch.load(run / f'point_step_{step:05d}.pt', map_location='cpu',
                          weights_only=True) for step in (0, 1000, 2000, 3000)]
assert [checkpoint['step'] for checkpoint in checkpoints] == [0, 1000, 2000, 3000]
assert all(not checkpoint['calibrated'] and 'optimizer' in checkpoint and
           'subjects_rng' in checkpoint and 'draw_seed' in checkpoint and
           'torch_rng' in checkpoint and 'cuda_rng' in checkpoint for checkpoint in checkpoints)
assert any(not torch.equal(value, checkpoints[-1]['model'][name])
           for name, value in checkpoints[0]['model'].items())
evaluation_config = json.loads((evaluation / 'config.json').read_text())
evaluation_completed = json.loads((evaluation / 'completed.json').read_text())
assert sha(run / 'completed.json') == evaluation_config['run_completed_sha256']
assert sha(run / 'config.json') == evaluation_config['run_config_sha256']
assert sha(panel / 'completed.json') == evaluation_config['panel_completed_sha256']
assert sha(panel / 'records.jsonl') == evaluation_config['panel_records_sha256']
assert evaluation_config['bank_seed'] == 2026102401
assert evaluation_config['bank_positions'] == 100000
assert evaluation_config['query_points_per_section'] == 32
assert not evaluation_config['calibrated'] and not evaluation_config['public_benchmark_used']
for name, expected in evaluation_config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == expected
for step, expected in evaluation_config['checkpoints_sha256'].items():
    assert sha(run / f'point_step_{int(step):05d}.pt') == expected
for name in ('config', 'bank', 'rows', 'summary'):
    path = (evaluation / {'config': 'config.json', 'bank': 'bank_ccf_ap_dv_ml_um.npy',
                          'rows': 'rows.jsonl', 'summary': 'summary.json'}[name])
    assert sha(path) == evaluation_completed[name + '_sha256']
bank = np.load(evaluation / 'bank_ccf_ap_dv_ml_um.npy', allow_pickle=False)
assert bank.shape == (100000, 3) and bank.dtype == np.float32
assert np.unique(bank, axis=0).shape[0] == len(bank)
tree = cKDTree(bank)
rows = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
summary = json.loads((evaluation / 'summary.json').read_text())
assert len(rows) == evaluation_completed['rows'] == summary['rows'] == 740
assert sorted({row['step'] for row in rows}) == [0, 1000, 2000, 3000]
assert len({row['section_id'] for row in rows}) == 185
assert not summary['calibrated'] and not summary['public_benchmark_used']
for row in rows:
    assert row['animal_id'] in development_animals
    truth = np.array(row['true_ccf_um'])
    xy = np.array(row['query_pixel_yx'])
    choice = np.array(row['top16_bank_indices'])
    assert truth.shape == (32, 3) and xy.shape == (32, 2) and choice.shape == (32, 16)
    assert np.all((choice >= 0) & (choice < len(bank)))
    distance = np.linalg.norm(bank[choice] - truth[:, None], axis=-1).min(-1)
    assert np.max(np.abs(distance - row['min_top16_um'])) < .1
    nearest = tree.query(truth, k=1)[0]
    assert np.max(np.abs(nearest - row['nearest_bank_um'])) < .1
    good = np.flatnonzero(distance < 500)
    triple = any(all(np.linalg.norm(truth[a] - truth[b]) >= 1000 and
                         np.linalg.norm(xy[a] - xy[b]) >= 16 for a, b in
                         itertools.combinations(indices, 2))
                 for indices in itertools.combinations(good, 3))
    assert bool(triple) == row['three_separated']
for group in summary['checkpoints']:
    selected = [row for row in rows if row['step'] == group['step']]
    subjects = sorted({row['synthetic_subject_plan_id'] for row in selected})
    assert len(selected) == group['sections'] == 185
    assert len(subjects) == group['synthetic_subjects'] == 8
    for threshold in (250, 500, 1000):
        for field, value_name in (('min_top16_um', f'recall16_{threshold}um'),
                                  ('nearest_bank_um', f'bank_coverage_{threshold}um')):
            mean = np.mean([np.mean([np.mean(np.array(row[field]) < threshold)
                                    for row in selected if row['synthetic_subject_plan_id'] == subject])
                            for subject in subjects])
            assert abs(mean - group[value_name]) < 1e-9
    fraction = np.mean([np.mean([row['three_separated'] for row in selected
                                 if row['synthetic_subject_plan_id'] == subject])
                        for subject in subjects])
    assert abs(fraction - group['three_separated_fraction']) < 1e-9
assert summary['promising_gate'] == any(row['recall16_500um'] >= .20 and
                                       row['three_separated_fraction'] >= .50
                                       for row in summary['checkpoints'][1:])
print(json.dumps({'status': 'passed', 'training_batches': len(training),
                  'synthetic_sections': len(used), 'development_rows': len(rows),
                  'promising_gate': summary['promising_gate']}))
