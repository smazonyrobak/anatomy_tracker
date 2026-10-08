"""Exploratory orientation/visibility attribution on the frozen 099 fresh panel."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

root = Path('I:/AnatomyTracker')
panel = root / 'data/contextual_match_099_fresh_synthetic_dev_panel_001'
evaluation = root / 'runs/contextual_match_099_fresh_development_eval'
out = root / 'runs/arbitrary_plane_100_exploratory_readout'
assert not out.exists()
panel_done = json.loads((panel / 'completed.json').read_text())
evaluation_done = json.loads((evaluation / 'completed.json').read_text())
assert panel_done['physical_sections'] == 256
assert evaluation_done['panel_completed_sha256'] == hashlib.sha256(
    (panel / 'completed.json').read_bytes()).hexdigest()
records = {row['section_id']: row for row in map(json.loads,
    (panel / 'records.jsonl').open()) if row['eligible']}
sections = {(row['section_id'], row['step']): row for row in map(json.loads,
    (evaluation / 'sections.jsonl').open())}
candidates = {}
for row in map(json.loads, (evaluation / 'candidates.jsonl').open()):
    candidates.setdefault((row['section_id'], row['step']), []).append(row)
assert len(records) == evaluation_done['eligible_sections_per_step']
assert len(sections) == 2 * len(records)
assert len(candidates) == 2 * len(records)

rows = []
family_names = ('AP', 'DV', 'ML')
for section_id, record in records.items():
    zero, final = sections[(section_id, 0)], sections[(section_id, 4000)]
    baseline = sorted(candidates[(section_id, 0)], key=lambda row: row['beam_slot'])
    updated = sorted(candidates[(section_id, 4000)], key=lambda row: row['beam_slot'])
    assert [row['beam_slot'] for row in baseline] == list(range(14))
    assert [row['beam_slot'] for row in updated] == list(range(14))
    assert all(a['branch_id'] == b['branch_id'] and
               a['mapped96_error_um'] == b['mapped96_error_um']
               for a, b in zip(baseline, updated))
    with np.load(panel / record['file'], allow_pickle=False) as arrays:
        target_state = torch.from_numpy(arrays['target_state'].copy()).float()
    truth_centre, truth_frame, _ = full_frame_state_to_components(target_state)
    fitted_states = torch.tensor([row['fitted_state'] for row in baseline])
    centres, frames, _ = full_frame_state_to_components(fitted_states)
    normal_errors = torch.rad2deg(torch.acos(
        (frames[:, :, 2] * truth_frame[:, 2]).sum(-1).abs().clamp(0, 1))).numpy()
    centre_errors_mm = ((centres - truth_centre).norm(dim=-1) / 1000).numpy()
    normal = np.asarray(record['plane_normal_ap_dv_ml'])
    family = family_names[int(np.argmax(np.abs(normal)))]
    angle = float(np.degrees(np.arccos(np.abs(normal).max().clip(0, 1))))
    angle_bin = ('0-15' if angle < 15 else '15-30' if angle < 30 else
                 '30-45' if angle < 45 else '45-54.7')
    valid_fraction = record['valid_pixels'] / 65536
    visibility = ('<10%' if valid_fraction < .1 else
                  '10-30%' if valid_fraction < .3 else '>=30%')
    artifact_count = (sum(record['provenance']['one_shot']['events'].values()) +
                      int(record['provenance']['appearance']['damage']))
    burden = 'none' if artifact_count == 0 else 'one' if artifact_count == 1 else '2+'
    best = zero['best_physical_beam_slot']
    selected = zero['parent_selected_beam_slot']
    assert best == final['best_physical_beam_slot']
    assert selected == final['parent_selected_beam_slot']
    rows.append({'section_id': section_id,
        'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
        'nearest_axis_family': family, 'angle_from_nearest_cardinal_deg': angle,
        'angle_bin': angle_bin, 'valid_fraction': valid_fraction,
        'visibility_bin': visibility, 'appearance_mode': record['appearance_mode'],
        'artifact_event_count': artifact_count, 'artifact_burden': burden,
        'best_mapped96_error_mm': zero['best_physical_mapped96_error_um'] / 1000,
        'parent_selected_mapped96_error_mm':
            zero['parent_selected_mapped96_error_um'] / 1000,
        'selection_gap_mm': (zero['parent_selected_mapped96_error_um'] -
                             zero['best_physical_mapped96_error_um']) / 1000,
        'best_normal_error_deg': float(normal_errors[best]),
        'selected_normal_error_deg': float(normal_errors[selected]),
        'best_centre_error_mm': float(centre_errors_mm[best]),
        'selected_centre_error_mm': float(centre_errors_mm[selected]),
        'best_raw_match_step0': baseline[best]['fine_raw_top1_fraction'],
        'best_raw_match_step4000': updated[best]['fine_raw_top1_fraction'],
        'mean_raw_match_step0': zero['mean_fine_raw_top1_fraction'],
        'mean_raw_match_step4000': final['mean_fine_raw_top1_fraction']})

groups = {'all': rows}
for key, categories in (('nearest_axis_family', family_names),
                        ('angle_bin', ('0-15', '15-30', '30-45', '45-54.7')),
                        ('visibility_bin', ('<10%', '10-30%', '>=30%')),
                        ('artifact_burden', ('none', 'one', '2+')),
                        ('appearance_mode', ('raw', 'exact_black', 'imperfect_brush'))):
    for category in categories:
        groups[f'{key}/{category}'] = [row for row in rows if row[key] == category]
for family in family_names:
    for angle_bin in ('0-15', '15-30', '30-45', '45-54.7'):
        groups[f'family_angle/{family}/{angle_bin}'] = [row for row in rows
            if row['nearest_axis_family'] == family and row['angle_bin'] == angle_bin]
metrics = ('best_mapped96_error_mm', 'parent_selected_mapped96_error_mm',
    'selection_gap_mm', 'best_normal_error_deg', 'selected_normal_error_deg',
    'best_centre_error_mm', 'selected_centre_error_mm',
    'best_raw_match_step0', 'best_raw_match_step4000',
    'mean_raw_match_step0', 'mean_raw_match_step4000')
summary = {}
for name, group in groups.items():
    plans = sorted({row['synthetic_subject_plan_id'] for row in group})
    summary[name] = {'sections': len(group), 'synthetic_deformation_plans': len(plans),
        'exploratory_small_cell': len(group) < 20,
        'plan_equal_mean': {metric: (float(np.mean([
            np.mean([row[metric] for row in group
                     if row['synthetic_subject_plan_id'] == plan and row[metric] is not None])
            for plan in plans if any(row['synthetic_subject_plan_id'] == plan and
                                     row[metric] is not None for row in group)]))
            if any(row[metric] is not None for row in group) else None)
            for metric in metrics}}

out.mkdir(parents=True)
(out / 'sections.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'scope': 'exploratory synthetic-only orientation/visibility attribution; not intrinsic ambiguity, biological validation, calibration or benchmark',
    'eligible_sections': len(rows),
    'panel_completed_sha256': hashlib.sha256((panel / 'completed.json').read_bytes()).hexdigest(),
    'evaluation_completed_sha256': hashlib.sha256((evaluation / 'completed.json').read_bytes()).hexdigest(),
    'sections_sha256': hashlib.sha256((out / 'sections.jsonl').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
}, indent=2))
print(json.dumps({'eligible_sections': len(rows), 'all': summary['all']}))
