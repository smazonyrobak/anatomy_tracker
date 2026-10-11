"""Independent read-only audit of the frozen 189 assay."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

root = Path('I:/AnatomyTracker')
source = Path(__file__).resolve().parent
run = root / 'runs/lupori_acquired_atlas_fit_direction_189_train_001'
done = json.loads((run / 'completed.json').read_text())
config = json.loads((run / 'config.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
rows = [json.loads(line) for line in (run / 'cases.jsonl').read_text().splitlines()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert done['sections'] == len(rows) == 64
assert config['source_sha256'] == sha(source / 'assay_lupori_acquired_fit_direction_189.py')
assert config['protocol_sha256'] == sha(source.parent /
    'docs/publication/LUPORI_ACQUIRED_ATLAS_FIT_DIRECTION_189_PROTOCOL_20261011.md')
assert config['checkpoint_sha256'] == done['checkpoint_sha256'] == sha(
    root / 'runs/lupori_fluorescence_pose_188_all_train/joint_step_01200.pt')
assert config['train_completion_sha256'] == sha(
    root / 'runs/lupori_fluorescence_pose_188_all_train/completed.json')
assert config['split_sha256'] == sha(source / 'lupori_animal_split_185.json')
assert {name: sha(run / name) for name in done['output_sha256']} == done['output_sha256']
assert not done['calibrated'] and not done['public_benchmark_used']
assert Counter(row['animal_id'] for row in rows) == {animal: 16 for animal in config['animals']}
assert len({(row['animal_id'], row['section_id']) for row in rows}) == 64
with ZipFile(root / 'data/lupori_joint_185/rawData.zip') as archive:
    for row in rows:
        assert hashlib.sha256(archive.read(row['source_thumbnail'])).hexdigest() == (
            row['source_thumbnail_sha256'])
        assert row['specimen_id'] == row['animal_id'] and row['experiment_id'] is None
        assert len(row['candidate_ids']) == len(row['prior_log_mass']) == len(
            row['candidate_five_point_mm']) == 16
        assert len(set(row['candidate_ids'])) == 16
        assert all(a >= b for a, b in zip(row['prior_log_mass'], row['prior_log_mass'][1:]))
        assert row['top16_near_le_1p5'] == (min(row['candidate_five_point_mm']) <= 1.5)
        for name in ('prior_top', 'near_diagnostic'):
            branch = row['branches'][name]
            if not branch['eligible']:
                continue
            index = branch['slot']
            assert branch['branch_id'] == row['candidate_ids'][index]
            assert .5 <= row['candidate_five_point_mm'][index] <= 3.
            if name == 'prior_top':
                assert index == 0
            else:
                choices = [error for error in row['candidate_five_point_mm'] if .5 <= error <= 3.]
                assert abs(row['candidate_five_point_mm'][index] - min(choices)) < 1e-5
            physical = branch['physical_mm_toward_center_away']
            assert branch['physical_move_monotonic'] == (physical[0] < physical[1] < physical[2])
            assert abs(physical[1] - row['candidate_five_point_mm'][index]) < 1e-4
            assert branch['fit_direction_positive'] == (
                branch['fit_score_toward_center_away'][0] >
                branch['fit_score_toward_center_away'][2])
            assert branch['null_direction_positive'] == (
                branch['null_score_toward_center_away'][0] >
                branch['null_score_toward_center_away'][2])
            assert branch['fit_derivative_positive'] == (branch['fit_derivative'] > 0)
            assert branch['null_derivative_positive'] == (branch['null_derivative'] > 0)
            assert branch['shared_sites'] >= 100
            assert all(0 <= value <= 1 for value in branch['support_retention_toward_center_away'])

assert summary['sections'] == 64
assert summary['top16_near_le_1p5'] == sum(row['top16_near_le_1p5'] for row in rows)
for name in ('prior_top', 'near_diagnostic'):
    for animal in ('all', *config['animals']):
        values = [row['branches'][name] for row in rows if animal == 'all' or
                  row['animal_id'] == animal]
        eligible = [branch for branch in values if branch['eligible']]
        expected = {'eligible': len(eligible),
            'fit_forward': sum(branch['fit_direction_positive'] for branch in eligible),
            'null_forward': sum(branch['null_direction_positive'] for branch in eligible),
            'fit_gradient_forward': sum(branch['fit_derivative_positive'] for branch in eligible),
            'monotonic_physical': sum(branch['physical_move_monotonic'] for branch in eligible),
            'minimum_support_retention': min((min(branch['support_retention_toward_center_away'])
                for branch in eligible), default=None)}
        assert summary['groups'][f'{name}/{animal}'] == expected
captured = summary['groups']['near_diagnostic/all']
animal_wins = sum(summary['groups'][f'near_diagnostic/{animal}']['eligible'] > 0 and
    summary['groups'][f'near_diagnostic/{animal}']['fit_forward'] >
    .5 * summary['groups'][f'near_diagnostic/{animal}']['eligible']
    for animal in config['animals'])
gate = (captured['eligible'] >= 32 and captured['fit_forward'] / captured['eligible'] >= .65
    and (captured['fit_forward'] - captured['null_forward']) / captured['eligible'] >= .10
    and captured['fit_gradient_forward'] / captured['eligible'] >= .60 and animal_wins >= 3)
assert summary['predeclared_gate'] == {'passed': gate, 'animals_above_half': animal_wins}
print(json.dumps({'verified': True, 'sections': len(rows), 'top16_near_le_1p5':
    summary['top16_near_le_1p5'], 'prior_top': summary['groups']['prior_top/all'],
    'near_diagnostic': captured, 'predeclared_gate': summary['predeclared_gate'],
    'completion_sha256': sha(run / 'completed.json')}))
