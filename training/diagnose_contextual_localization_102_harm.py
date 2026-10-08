"""Locate where the frozen 102A decoded offsets help or hurt fitted planes."""
import json
from pathlib import Path

import numpy as np

path = Path('I:/AnatomyTracker/runs/contextual_localization_102_fresh_development_eval/candidates.jsonl')
rows = [row for row in map(json.loads, path.open()) if row['step'] == 4000]
for name, group in (
    ('best/<0.75mm', [r for r in rows if r['physical_best'] and r['mapped96_error_um'] < 750]),
    ('best/0.75-1.5mm', [r for r in rows if r['physical_best'] and 750 <= r['mapped96_error_um'] < 1500]),
    ('best/>=1.5mm', [r for r in rows if r['physical_best'] and r['mapped96_error_um'] >= 1500]),
    ('other/<0.75mm', [r for r in rows if not r['physical_best'] and r['mapped96_error_um'] < 750]),
    ('other/0.75-1.5mm', [r for r in rows if not r['physical_best'] and 750 <= r['mapped96_error_um'] < 1500]),
    ('other/>=1.5mm', [r for r in rows if not r['physical_best'] and r['mapped96_error_um'] >= 1500]),
):
    usable = [r for r in group if r['decoded_error_um'] is not None]
    print(json.dumps({'group': name, 'candidates': len(group), 'usable': len(usable),
        'fraction_match_reduces_error': float(np.mean([r['decoded_error_um'] < r['null_error_um']
                                                       for r in usable])) if usable else None,
        'mean_error_change_mm': float(np.mean([(r['decoded_error_um'] - r['null_error_um']) / 1000
                                               for r in usable])) if usable else None,
        'mean_correct_fraction_change': float(np.mean([
            r['decoded_correct_fraction'] - r['null_correct_fraction'] for r in usable
        ])) if usable else None}))
