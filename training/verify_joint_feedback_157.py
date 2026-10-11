"""Read-only hash, paired-draw and metric audit after both 157 arms finish."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker')
repo = Path(__file__).resolve().parents[1]
run = root / 'runs/joint_feedback_157_train151_001'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
protocol = json.loads((run / 'protocol.json').read_text())
draws = [json.loads(line) for line in (run / 'draws.jsonl').open()]
inner_draws = [json.loads(line) for line in (run / 'inner_draws.jsonl').open()]
accepted = [row for row in draws if row['used']]
assert sha(run / 'protocol.json') == done['protocol_sha256'] == config['protocol_sha256']
assert sha(run / 'config.json') == done['config_sha256']
assert sha(run / 'draws.jsonl') == done['draws_sha256']
assert sha(run / 'inner_draws.jsonl') == done['inner_draws_sha256']
assert config['protocol'] == protocol
assert set(protocol['arms']) == set(done['arms']) == {'feedback', 'detached_control'}
assert len(accepted) == done['accepted_physical_sections_per_arm'] == \
    done['updates_per_arm'] == protocol['updates_per_arm'] == 3000
assert len([row for row in inner_draws if row['used']]) == 48
assert {row['base_index'] for row in accepted} <= set(protocol['gradient_bases'])
assert {row['base_index'] for row in inner_draws} <= set(protocol['inner_bases'])
assert all(sha(repo / 'training' / name) == digest for name, digest in
           config['source_sha256'].items())
assert done['source_sha256'] == config['source_sha256']
train = root / 'runs/joint_fitter_155_train151_001'
train_done = json.loads((train / 'completed.json').read_text())
step = train_done['selected_step']
assert sha(train / f'joint_fitter_step_{step:05d}.pt') == \
    done['parent_checkpoint_sha256'] == protocol['parent_checkpoint_sha256']

case_ids = []
baseline = []
for arm in protocol['arms']:
    directory = run / arm
    result = done['arms'][arm]
    assert sha(directory / 'training.jsonl') == result['training_sha256']
    assert sha(directory / 'inner_scores.jsonl') == result['inner_scores_sha256']
    for number, digest in result['checkpoint_sha256'].items():
        assert int(number) in protocol['checkpoints']
        assert sha(directory / f'joint_feedback_step_{int(number):05d}.pt') == digest
    training = [json.loads(line) for line in (directory / 'training.jsonl').open()]
    checkpoints = [json.loads(line) for line in (directory / 'inner_scores.jsonl').open()]
    assert len(training) == 3000
    assert [row['update'] for row in training] == list(range(1, 3001))
    ids = [(row['base_index'], row['draw_seed'], row['appearance_version'],
            row['physical_section_id']) for row in training]
    case_ids.append(ids)
    assert ids == [(row['base_index'], row['draw_seed'], row['appearance_version'],
                    row['provenance']['physical_section_id']) for row in accepted]
    assert [row['step'] for row in checkpoints] == protocol['checkpoints']
    scores = {}
    for checkpoint in checkpoints:
        metrics = checkpoint['metrics']
        rows = checkpoint['rows']
        assert len(rows) == metrics['n'] == 48
        assert math.isclose(np.mean([row['direct_top1_mm'] for row in rows]),
                            metrics['direct_top1_mm'], abs_tol=1e-8)
        assert math.isclose(np.mean([row['oracle_best_mm'] for row in rows]),
                            metrics['oracle_best_mm'], abs_tol=1e-8)
        assert math.isclose(np.mean([row['top16_near'] for row in rows]),
                            metrics['top16_near_fraction'], abs_tol=1e-8)
        near = [row for row in rows if row['top16_near']]
        assert near
        assert math.isclose(np.mean([row['near_prefit_mm'] for row in near]),
                            metrics['near_prefit_mm'], abs_tol=1e-8)
        assert math.isclose(np.mean([row['near_postfit_mm'] for row in near]),
                            metrics['near_postfit_mm'], abs_tol=1e-8)
        score = (metrics['direct_top1_mm'] + .5 * metrics['oracle_best_mm']
                 + 1 - metrics['top16_near_fraction'])
        assert math.isclose(score, metrics['selection_score'], abs_tol=1e-8)
        scores[checkpoint['step']] = score
    assert result['selected_step'] == min(scores, key=scores.get)
    assert result['selected_inner_metrics'] == next(row['metrics'] for row in checkpoints
                                                    if row['step'] == result['selected_step'])
    baseline.append(checkpoints[0])
assert case_ids[0] == case_ids[1]
assert baseline[0] == baseline[1]
assert not any(done[key] or protocol[key] for key in
    ('calibrated', 'pretrained_weights_used', 'real_images_used',
     'dev_or_public_benchmark_used', 'final_animals_used'))
print(json.dumps({'verified': True, 'paired_sections': len(accepted),
                  'selected': {arm: done['arms'][arm]['selected_inner_metrics']
                               for arm in protocol['arms']}}, indent=2))
