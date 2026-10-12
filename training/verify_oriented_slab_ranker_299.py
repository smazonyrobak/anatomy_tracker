"""Independent frozen-output and final-model audit of candidate 299."""

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

RUN = ROOT / 'runs/oriented_slab_ranker_299_train_inner_001'
SOURCE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lines(name):
    return [json.loads(line) for line in (RUN / name).open(encoding='utf8')]


config = json.loads((RUN / 'config.json').read_text())
receipt = json.loads((RUN / 'completed.json').read_text())
summary = json.loads((RUN / 'summary.json').read_text())
assert config['candidate'] == receipt['candidate'] == 299
assert sha(ROOT / 'runs/pose_capture_275_v9_continuation_001/pose_step_08000.pt') == \
    config['parent_checkpoint_sha256']
assert sha(SOURCE.parent / 'docs/publication/ORIENTED_SLAB_RANKER_299_PROTOCOL_20261012.md') == \
    config['protocol_sha256']
for name, expected in config['source_sha256'].items():
    assert sha(SOURCE / name) == expected, name
for name, expected in receipt['output_sha256'].items():
    assert sha(RUN / name) == expected, name
for name, expected in receipt['checkpoint_sha256'].items():
    assert sha(RUN / f'{name}.pt') == expected, name

panel = torch.load(RUN / 'inner_panel.pt', map_location='cpu', weights_only=False)
inner_draws, draws, train, readout = map(lines,
    ('inner_draws.jsonl', 'draws.jsonl', 'training.jsonl', 'readout.jsonl'))
assert len(panel) == len(readout) == 64
assert len(inner_draws) == receipt['panel_attempted']
assert sum(row['used'] for row in inner_draws) == 64
assert len(draws) == receipt['attempted'] and sum(row['accepted'] for row in draws) == 1200
assert len(train) == receipt['accepted'] == config['updates'] == 1200
panel_ids = [item['physical_section_id'] for item in panel]
train_ids = [row['physical_section_id'] for row in train]
assert len(set(panel_ids)) == 64 and len(set(train_ids)) == 1200
assert not set(panel_ids) & set(train_ids)
assert [row['physical_section_id'] for row in readout] == panel_ids
assert all(row['base'] in range(56) for row in train)
assert all(item['base'] in range(56, 64) for item in panel)
assert all(item['provenance']['base_lineage']['animal_index'] == item['base'] for item in panel)
assert all(item['provenance']['split'] == 'train' for item in panel)
assert all(row['provenance']['base_lineage']['subject_id'].endswith(
    f"-{row['base']:08d}") for row in draws if row['accepted'])
assert all(row['provenance']['split'] == 'train' for row in draws if row['accepted'])
assert all(row['step'] == step for step, row in enumerate(train, 1))

torch.set_num_threads(4)
rankers = {}
for name in ('full', 'support_only'):
    saved = torch.load(RUN / f'{name}.pt', map_location='cpu', weights_only=True)
    assert saved['step'] == 1200 and saved['arm'] == name
    assert saved['parent_sha256'] == config['parent_checkpoint_sha256']
    assert saved['calibrated'] is False
    rankers[name] = make_ranker_299().cuda().eval()
    rankers[name].load_state_dict(saved['model'], strict=True)

with torch.inference_mode():
    for item, row in zip(panel, readout):
        assert item['physical_section_id'] == row['physical_section_id']
        assert len(row['candidate_error_mm']) == len(row['candidate_prior_log']) == 16
        assert np.allclose(item['error_mm'].numpy(), row['candidate_error_mm'], atol=1e-6)
        assert int(np.argmin(row['candidate_error_mm'])) == int(np.argmin(item['error_mm']))
        assert abs(row['prior_error_mm'] - row['candidate_error_mm'][0]) < 1e-6
        if row['near_far'] is not None:
            near, far = row['near_far']
            assert row['candidate_error_mm'][near] <= 1.5
            assert row['candidate_error_mm'][far] >= max(2., row['candidate_error_mm'][near] + 1.)
            assert abs(row['candidate_support'][near] - row['candidate_support'][far]) <= .100001
        for name, model in rankers.items():
            evidence, _ = score_bank_299(model, item['inputs'][None].cuda(),
                                         item['bank'].cuda(), name == 'support_only')
            observed = row['arms'][name]
            assert np.allclose(evidence.cpu().numpy(), observed['evidence'], atol=2e-6)
            selected = int(np.argmax(np.asarray(row['candidate_prior_log'])
                                     + 4 * np.asarray(observed['evidence'])))
            assert selected == observed['selected']
            assert abs(observed['selected_error_mm'] - row['candidate_error_mm'][selected]) < 1e-6
            if row['near_far'] is not None:
                assert observed['near_far_evidence_win'] == \
                    (observed['evidence'][near] > observed['evidence'][far])

eligible = [row for row in readout if row['near_far'] is not None]
prior_mean = np.mean([row['prior_error_mm'] for row in readout])
oracle_mean = np.mean([row['oracle_error_mm'] for row in readout])
assert abs(summary['prior_mean_mm'] - prior_mean) < 1e-9
assert abs(summary['oracle_mean_mm'] - oracle_mean) < 1e-9
assert summary['matched_pairs'] == len(eligible)
for name in rankers:
    selected = np.mean([row['arms'][name]['selected_error_mm'] for row in readout])
    wins = sum(row['arms'][name]['near_far_evidence_win'] for row in eligible)
    assert abs(summary[name]['mean_mm'] - selected) < 1e-9
    assert summary[name]['near_far_wins'] == wins
    for base in range(56, 64):
        mean = np.mean([row['arms'][name]['selected_error_mm']
                        for row in readout if row['base'] == base])
        assert abs(summary[name]['per_base_mean_mm'][str(base)] - mean) < 1e-9
assert summary['gate_pass'] is False and not all(summary['gate'].values())
assert all(value is False for name, value in summary['gate'].items()
           if name != 'at_least_30_matched_pairs')

strata = {}
for label, subset in {
    'brush': [row for row in readout if row['brush_available']],
    'no_brush': [row for row in readout if not row['brush_available']],
    'oblique_45plus': [row for row in readout if row['obliquity_deg'] >= 45],
    'other_angle': [row for row in readout if row['obliquity_deg'] < 45],
}.items():
    strata[label] = {'n': len(subset),
        'prior_mm': float(np.mean([row['prior_error_mm'] for row in subset])) if subset else None,
        'full_mm': float(np.mean([row['arms']['full']['selected_error_mm']
                                  for row in subset])) if subset else None,
        'support_only_mm': float(np.mean([row['arms']['support_only']['selected_error_mm']
                                          for row in subset])) if subset else None}
print(json.dumps({'verified': True, 'completed_sha256': sha(RUN / 'completed.json'),
    'summary': summary, 'strata': strata}, allow_nan=False))
