"""Summarize saved 107 scores; constant comparison scores have undefined rank."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

root = Path('I:/AnatomyTracker')
panel = root / 'data/v3_pose_capture_confirmation_panel_001'
evaluation = root / 'runs/coarse_pose_updater_106_dev_eval'
training = root / 'runs/coarse_pose_updater_106'
out = root / 'runs/atlas_evidence_diagnostic_107'
assert (out / 'scores.jsonl').is_file() and not (out / 'completed.json').exists()
rows = [json.loads(line) for line in (out / 'scores.jsonl').open() if line.strip()]
sections = [json.loads(line) for line in (evaluation / 'sections.jsonl').open()
    if line.strip()]
sections = [row for row in sections if row['step'] == 4000 and row['arm'] == 'atlas']
assert len(sections) == 64 and len(rows) == 64 * 15
grouped = [[row for row in rows if row['section_id'] == section['section_id']]
    for section in sections]
assert all([row['slot'] for row in group] == list(range(15)) and
    all(row['oracle_true_plane_diagnostic_only'] == (row['slot'] == 14)
        for row in group) for group in grouped)
score_names = [key for key in rows[0] if key.endswith(('_blind', '_true_mask'))]
summary = {'role': 'exploratory, reused synthetic DEV; true plane diagnostic only',
    'sections': 64, 'blind_candidates_per_section': 14, 'scores': {}}
for name in score_names:
    rho = []
    for group in grouped:
        values = np.array([row[name] for row in group[:14]])
        error = np.array([row['input_rigid_error_mm'] for row in group[:14]])
        if np.std(values) > 1e-8 and np.std(error) > 1e-8:
            value = spearmanr(values, -error).statistic
            if np.isfinite(value):
                rho.append(float(value))
    oracle = np.array([group[14][name] for group in grouped])
    prior = np.array([group[section['prior_slot']][name]
        for group, section in zip(grouped, sections)])
    best_blind = np.array([max(row[name] for row in group[:14]) for group in grouped])
    summary['scores'][name] = {'mean_within_beam_rho': float(np.mean(rho)) if rho else None,
        'sections_with_defined_rho': len(rho),
        'oracle_above_prior_fraction': float(np.mean(oracle > prior)),
        'oracle_above_all_blind_fraction': float(np.mean(oracle > best_blind)),
        'oracle_minus_prior_median': float(np.median(oracle - prior)),
        'oracle_minus_best_blind_median': float(np.median(oracle - best_blind))}

(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'generation_source_sha256': hashlib.sha256(
        (Path(__file__).parent / 'diagnose_atlas_evidence_107.py').read_bytes()).hexdigest(),
    'summary_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'panel_completion_sha256': hashlib.sha256((panel / 'completed.json').read_bytes()).hexdigest(),
    'evaluation_completion_sha256': hashlib.sha256(
        (evaluation / 'completed.json').read_bytes()).hexdigest(),
    'training_completion_sha256': hashlib.sha256(
        (training / 'completed.json').read_bytes()).hexdigest(),
    'scores_sha256': hashlib.sha256((out / 'scores.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'no_new_training': True, 'uses_reused_dev': True,
    'true_plane_never_selectable': True}, indent=2))
print(json.dumps(summary), flush=True)
