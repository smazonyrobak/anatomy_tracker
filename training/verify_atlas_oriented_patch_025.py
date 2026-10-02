"""Independent training lineage and oracle-plane patch-result verifier."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

root = Path('I:/AnatomyTracker')
run = root / 'runs/atlas_oriented_patch_025'
evaluation = root / 'runs/atlas_oriented_patch_025_development_eval'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


config = json.loads((run / 'config.json').read_text())
completed = json.loads((run / 'completed.json').read_text())
assert config['updates'] == completed['updates'] == 3000
assert config['sections_per_batch'] == 2 and config['positive_points_per_section'] == 32
assert config['atlas_negatives_per_section'] == 32
assert completed['accepted_synthetic_sections'] == 6000 and completed['positive_pairs'] == 192000
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
panel_records = {row['sha256']: row for row in map(json.loads, (panel / 'records.jsonl').open())}
development_animals = {row['animal_id'] for row in panel_records.values()}
assert all(row['split'] == 'train' and row['base_lineage']['split'] == 'train' for row in used)
assert not development_animals & {row['base_lineage']['animal_id'] for row in used}
training = [json.loads(line) for line in (run / 'training.jsonl').open()]
assert [row['step'] for row in training] == list(range(1, 3001))
assert np.isfinite([row['loss'] for row in training]).all()
assert np.isfinite([row['gradient_norm'] for row in training]).all()
assert training[0]['gradient_norm'] > 0
checkpoints = [torch.load(run / f'patch_step_{step:05d}.pt', map_location='cpu',
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
assert evaluation_config['atlas_candidates_per_section'] == 288
assert evaluation_config['query_points_per_section'] == 32
assert not evaluation_config['calibrated'] and not evaluation_config['public_benchmark_used']
for name, expected in evaluation_config['source_sha256'].items():
    assert sha(Path(__file__).parent / name) == expected
for step, expected in evaluation_config['checkpoints_sha256'].items():
    assert sha(run / f'patch_step_{int(step):05d}.pt') == expected
for name in ('config', 'rows', 'summary'):
    assert sha(evaluation / (name + ('.jsonl' if name == 'rows' else '.json'))) == \
        evaluation_completed[name + '_sha256']
rows = [json.loads(line) for line in (evaluation / 'rows.jsonl').open()]
summary = json.loads((evaluation / 'summary.json').read_text())
assert len(rows) == evaluation_completed['rows'] == summary['rows'] == 740
assert sorted({row['step'] for row in rows}) == [0, 1000, 2000, 3000]
assert len({row['section_id'] for row in rows}) == 185
assert not summary['calibrated'] and not summary['public_benchmark_used']
for row in rows:
    panel_record = panel_records[row['sha256']]
    assert panel_record['animal_id'] == row['animal_id']
    assert panel_record['appearance_mode'] == row['appearance_mode']
    truth = np.asarray(row['true_ccf_um'])
    xy = np.asarray(row['query_pixel_yx'])
    candidate = np.asarray(row['candidate_ccf_um'])
    candidate_xy = np.asarray(row['candidate_pixel_yx'])
    top = np.asarray(row['top16_indices'])
    assert truth.shape == (32, 3) and xy.shape == (32, 2)
    assert candidate.shape == (288, 3) and candidate_xy.shape == (288, 2)
    assert top.shape == (32, 16) and np.all((top >= 0) & (top < 288))
    assert np.array_equal(candidate_xy[256:], xy) and np.allclose(candidate[256:], truth)
    distances = np.linalg.norm(candidate[top] - truth[:, None], axis=-1)
    assert np.max(np.abs(distances.min(-1) - row['min_top16_um'])) < .1
    assert np.max(np.abs(distances[:, 0] - row['best_um'])) < .1
    assert np.all((np.asarray(row['exact_positive_rank']) >= 1) &
                  (np.asarray(row['exact_positive_rank']) <= 288))
for group in summary['checkpoints']:
    selected = [row for row in rows if row['step'] == group['step']]
    subjects = sorted({row['synthetic_subject_plan_id'] for row in selected})
    assert len(selected) == group['sections'] == 185
    assert len(subjects) == group['synthetic_subjects'] == 8
    for threshold in (250, 500, 1000):
        for field, name in (('best_um', 'recall1'), ('min_top16_um', 'recall16')):
            mean = np.mean([np.mean([np.mean(np.asarray(row[field]) < threshold)
                                    for row in selected if row['synthetic_subject_plan_id'] == subject])
                            for subject in subjects])
            assert abs(mean - group[f'{name}_{threshold}um']) < 1e-9
    exact = np.mean([np.mean([np.mean(np.asarray(row['exact_positive_rank']) == 1)
                              for row in selected if row['synthetic_subject_plan_id'] == subject])
                     for subject in subjects])
    assert abs(exact - group['exact_positive_top1']) < 1e-9
    for mode, claimed in group['mode_recall16_500um'].items():
        subset = [row for row in selected if row['appearance_mode'] == mode]
        mode_subjects = sorted({row['synthetic_subject_plan_id'] for row in subset})
        value = np.mean([np.mean([np.mean(np.asarray(row['min_top16_um']) < 500)
                                  for row in subset if row['synthetic_subject_plan_id'] == subject])
                         for subject in mode_subjects])
        assert abs(value - claimed) < 1e-9
assert summary['necessary_gate'] == any(row['recall1_500um'] >= .25 and
    row['recall16_500um'] >= .70 and
    min(row['mode_recall16_500um'].values()) >= .5 * row['recall16_500um']
    for row in summary['checkpoints'][1:])
print(json.dumps({'status': 'passed', 'training_batches': len(training),
                  'synthetic_sections': len(used), 'development_rows': len(rows),
                  'necessary_gate': summary['necessary_gate']}))
