"""Independent frozen lineage, 16-way scores, and gate audit for candidate 300."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch

from training.oriented_slab_ranker_299 import make_ranker_299, score_bank_299

RUN = ROOT / 'runs/exact_point_ranker_300_train_inner_002'
PREVIOUS = ROOT / 'runs/oriented_slab_ranker_299_train_inner_001'
SOURCE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).open(encoding='utf8')]


config = json.loads((RUN / 'config.json').read_text())
receipt = json.loads((RUN / 'completed.json').read_text())
summary = json.loads((RUN / 'summary.json').read_text())
assert config['candidate'] == receipt['candidate'] == 300
assert sha(ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt') == \
    config['parent_checkpoint_sha256']
assert sha(PREVIOUS / 'completed.json') == config['previous_completed_sha256']
assert sha(PREVIOUS / 'inner_panel.pt') == config['previous_panel_sha256']
assert sha(PREVIOUS / 'draws.jsonl') == config['previous_draws_sha256']
assert sha(SOURCE.parent / 'docs/publication/EXACT_POINT_RANKER_300_PROTOCOL_20261012.md') == \
    config['protocol_sha256']
for name, expected in config['source_sha256'].items():
    assert sha(SOURCE / name) == expected, name
for name, expected in receipt['output_sha256'].items():
    assert sha(RUN / name) == expected, name
for name, expected in receipt['checkpoint_sha256'].items():
    assert sha(RUN / f'{name}.pt') == expected, name

panel = torch.load(PREVIOUS / 'inner_panel.pt', map_location='cpu', weights_only=False)
before = lines(PREVIOUS / 'readout.jsonl')
previous_ids = {row['physical_section_id'] for row in lines(PREVIOUS / 'draws.jsonl')
                if row['accepted']}
draws, train, rows = map(lambda x: lines(RUN / x),
    ('draws.jsonl', 'training.jsonl', 'readout.jsonl'))
assert len(panel) == len(before) == len(rows) == 64
assert len(draws) == receipt['attempted'] and sum(row['accepted'] for row in draws) == 1200
assert len(train) == receipt['accepted'] == config['updates'] == 1200
train_ids = [row['physical_section_id'] for row in train]
assert len(set(train_ids)) == 1200 and not set(train_ids) & previous_ids
assert not set(train_ids) & {item['physical_section_id'] for item in panel}
assert all(row['base'] in range(56) for row in train)
assert all(row['step'] == step for step, row in enumerate(train, 1))
assert all(row['provenance']['split'] == 'train' for row in draws if row['accepted'])
assert all(row['provenance']['base_lineage']['subject_id'].endswith(
    f"-{row['base']:08d}") for row in draws if row['accepted'])

torch.set_num_threads(4)
models = {}
for name in ('full', 'support_only'):
    saved = torch.load(RUN / f'{name}.pt', map_location='cpu', weights_only=True)
    assert saved['step'] == 1200 and saved['arm'] == name
    assert saved['parent_sha256'] == config['parent_checkpoint_sha256']
    assert saved['calibrated'] is False
    models[name] = make_ranker_299().cuda().eval()
    models[name].load_state_dict(saved['model'], strict=True)

with torch.inference_mode():
    for item, earlier, row in zip(panel, before, rows):
        assert item['physical_section_id'] == earlier['physical_section_id'] == row['physical_section_id']
        assert row['near_far'] == earlier['near_far']
        assert abs(row['previous_299_full_error_mm'] - earlier['arms']['full']['selected_error_mm']) < 1e-8
        assert np.allclose(row['candidate_error_mm'], item['error_mm'].numpy(), atol=1e-6)
        assert abs(row['prior_error_mm'] - row['candidate_error_mm'][0]) < 1e-6
        for name, model in models.items():
            evidence, _ = score_bank_299(model, item['inputs'][None].cuda(),
                                         item['bank'].cuda(), name == 'support_only')
            arm = row['arms'][name]
            assert np.allclose(evidence.cpu().numpy(), arm['evidence'], atol=2e-6)
            selected = int(np.argmax(np.asarray(row['candidate_prior_log'])
                                     + 4 * np.asarray(arm['evidence'])))
            assert selected == arm['selected']
            assert abs(arm['selected_error_mm'] - row['candidate_error_mm'][selected]) < 1e-6
            if row['near_far'] is not None:
                near, far = row['near_far']
                assert arm['near_far_evidence_win'] == \
                    (arm['evidence'][near] > arm['evidence'][far])

eligible = [row for row in rows if row['near_far'] is not None]
assert summary['matched_pairs'] == len(eligible) == 49
assert abs(summary['prior_mean_mm'] - np.mean([row['prior_error_mm'] for row in rows])) < 1e-9
assert abs(summary['oracle_mean_mm'] - np.mean([row['oracle_error_mm'] for row in rows])) < 1e-9
assert abs(summary['previous_299_full_mean_mm'] - np.mean([
    row['previous_299_full_error_mm'] for row in rows])) < 1e-9
for name in models:
    assert abs(summary[name]['mean_mm'] - np.mean([
        row['arms'][name]['selected_error_mm'] for row in rows])) < 1e-9
    assert summary[name]['near_far_wins'] == sum(
        row['arms'][name]['near_far_evidence_win'] for row in eligible)
assert summary['gate_pass'] is False and sum(summary['gate'].values()) == 2

strata = {}
for label, subset in {
    'brush': [row for row in rows if row['brush_available']],
    'no_brush': [row for row in rows if not row['brush_available']],
    'oblique_45plus': [row for row in rows if row['obliquity_deg'] >= 45],
    'other_angle': [row for row in rows if row['obliquity_deg'] < 45],
}.items():
    strata[label] = {'n': len(subset),
        'prior_mm': float(np.mean([row['prior_error_mm'] for row in subset])) if subset else None,
        'full_mm': float(np.mean([row['arms']['full']['selected_error_mm']
                                  for row in subset])) if subset else None,
        'support_only_mm': float(np.mean([row['arms']['support_only']['selected_error_mm']
                                          for row in subset])) if subset else None}
print(json.dumps({'verified': True, 'completed_sha256': sha(RUN / 'completed.json'),
    'summary': summary, 'strata': strata}, allow_nan=False))
