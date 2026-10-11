"""Independent read-only audit of the frozen 190 ranking assay."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

root = Path('I:/AnatomyTracker')
source = Path(__file__).resolve().parent
run = root / 'runs/lupori_acquired_blind_fit_ranking_190_train_001'
previous = root / 'runs/lupori_acquired_atlas_fit_direction_189_train_001'
done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
rows = [json.loads(line) for line in (run / 'rows.jsonl').read_text().splitlines()]
previous_rows = [json.loads(line) for line in (previous / 'cases.jsonl').read_text().splitlines()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert len(rows) == done['sections'] == summary['sections'] == 64
assert config['source_sha256'] == done['source_sha256'] == sha(
    source / 'assay_lupori_acquired_blind_fit_ranking_190.py')
assert config['protocol_sha256'] == sha(source.parent /
    'docs/publication/LUPORI_ACQUIRED_BLIND_FIT_RANKING_190_PROTOCOL_20261011.md')
assert config['checkpoint_sha256'] == done['checkpoint_sha256'] == sha(
    root / 'runs/lupori_fluorescence_pose_188_all_train/joint_step_01200.pt')
assert config['previous_189_completion_sha256'] == sha(previous / 'completed.json')
assert config['previous_189_cases_sha256'] == sha(previous / 'cases.jsonl')
assert {name: sha(run / name) for name in done['output_sha256']} == done['output_sha256']
assert not done['trained'] and not done['calibrated'] and not done['public_benchmark_used']
assert Counter(row['animal_id'] for row in rows) == {animal: 16 for animal in config['animals']}
assert len({(row['animal_id'], row['section_id']) for row in rows}) == 64
with ZipFile(root / 'data/lupori_joint_185/rawData.zip') as archive:
    for row, earlier in zip(rows, previous_rows):
        assert (row['animal_id'], row['section_id'], row['source_thumbnail']) == (
            earlier['animal_id'], earlier['section_id'], earlier['source_thumbnail'])
        assert hashlib.sha256(archive.read(row['source_thumbnail'])).hexdigest() == (
            row['source_thumbnail_sha256'])
        assert row['candidate_ids'] == earlier['candidate_ids']
        assert max(abs(a - b) for a, b in zip(row['candidate_five_point_mm'],
            earlier['candidate_five_point_mm'])) < 1e-4
        assert len(row['individual_fit_score']) == len(row['individual_supported_sites']) == 16
        valid = [(index, value) for index, value in enumerate(row['individual_fit_score'])
                 if value is not None]
        assert all((value is not None) == (sites >= 100) for value, sites in zip(
            row['individual_fit_score'], row['individual_supported_sites']))
        best = max(valid, key=lambda item: item[1])[0] if valid else None
        assert row['blind_fit_slot'] == best
        assert row['prior_selected_mm'] == row['candidate_five_point_mm'][0]
        assert row['blind_fit_selected_mm'] == (row['candidate_five_point_mm'][best]
            if best is not None else None)
        pair = row['pair']
        near = [i for i, error in enumerate(row['candidate_five_point_mm']) if error <= 1.5]
        far = [i for i, error in enumerate(row['candidate_five_point_mm']) if error >= 3.]
        if not near or not far:
            assert not pair['eligible'] and pair['reason'] == 'missing_near_or_far'
            continue
        expected_near = min(near, key=lambda i: row['candidate_five_point_mm'][i])
        assert (pair['near_slot'], pair['far_slot']) == (expected_near, far[0])
        if not pair['eligible']:
            assert pair['reason'] == 'insufficient_common_support' and pair['common_sites'] < 100
            continue
        assert pair['common_sites'] >= 100
        assert pair['real_near_wins'] == (pair['real_near_far'][0] > pair['real_near_far'][1])
        assert pair['null_near_wins'] == (pair['null_near_far'][0] > pair['null_near_far'][1])
        assert pair['prior_near_wins'] == (expected_near < far[0])

paired = [row for row in rows if row['pair']['eligible']]
pair = {'eligible': len(paired),
    'real_near_wins': sum(row['pair']['real_near_wins'] for row in paired),
    'null_near_wins': sum(row['pair']['null_near_wins'] for row in paired),
    'prior_near_wins': sum(row['pair']['prior_near_wins'] for row in paired)}
assert summary['pair'] == pair
animal = {}
for identity in config['animals']:
    group = [row for row in rows if row['animal_id'] == identity]
    paired = [row for row in group if row['pair']['eligible']]
    animal[identity] = {'pair_eligible': len(paired),
        'real_near_wins': sum(row['pair']['real_near_wins'] for row in paired),
        'null_near_wins': sum(row['pair']['null_near_wins'] for row in paired),
        'prior_selected_mean_mm': float(np.mean([row['prior_selected_mm'] for row in group])),
        'blind_fit_selected_mean_mm': float(np.mean([row['blind_fit_selected_mm']
            if row['blind_fit_selected_mm'] is not None else row['prior_selected_mm']
            for row in group]))}
assert summary['animal'] == animal
missing = sum(row['blind_fit_slot'] is None for row in rows)
prior = float(np.mean([animal[a]['prior_selected_mean_mm'] for a in config['animals']]))
blind = float(np.mean([animal[a]['blind_fit_selected_mean_mm'] for a in config['animals']]))
assert summary['individual_score_missing'] == missing
assert summary['prior_equal_animal_mean_mm'] == prior
assert summary['blind_fit_equal_animal_mean_mm'] == blind
pair_gate = pair['eligible'] >= 32 and pair['real_near_wins'] / max(1, pair['eligible']) >= .70 and (
    pair['real_near_wins'] - pair['null_near_wins']) / max(1, pair['eligible']) >= .10 and sum(
    animal[a]['pair_eligible'] > 0 and animal[a]['real_near_wins'] >
    .5 * animal[a]['pair_eligible'] for a in config['animals']) >= 3
blind_gate = prior - blind >= .15 and missing <= 4 and all(
    animal[a]['blind_fit_selected_mean_mm'] - animal[a]['prior_selected_mean_mm'] <= .2
    for a in config['animals'])
assert summary['predeclared_pair_gate'] == pair_gate
assert summary['predeclared_blind_gate'] == blind_gate
print(json.dumps({'verified': True, 'pair': pair, 'prior_equal_animal_mean_mm': prior,
    'blind_fit_equal_animal_mean_mm': blind, 'pair_gate': pair_gate,
    'blind_gate': blind_gate, 'completion_sha256': sha(run / 'completed.json')}))
