"""Read-only decomposition of the completed 094 synthetic DEV candidate table."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

run = Path('I:/AnatomyTracker/runs/spatial_verifier_094_development_eval')
receipt = json.loads((run / 'completed.json').read_text())
digest = hashlib.sha256((run / 'candidates.jsonl').read_bytes()).hexdigest()
assert digest == receipt['candidates_sha256']
rows = pd.read_json(run / 'candidates.jsonl', lines=True)
rows = rows[(rows.step == 20000) & rows.in_fixed_beam & rows.independent_head_readout].copy()
assert rows.section_id.nunique() == 182 and len(rows) == 182 * 14
rows['error_rank'] = rows.groupby('section_id').mapped96_error_um.rank(method='first').astype(int)
measures = ('prior_score', 'fit_energy', 'score', 'atlas_support_fraction',
            'predicted_inlier_fraction', 'actual_correct_fraction', 'quality',
            'coverage', 'pose_correction_cost', 'warp_cost')
correlations = {measure: np.mean([spearmanr(part[measure], -part.mapped96_error_um).statistic
    for _, part in rows.groupby('section_id') if part[measure].nunique() > 1])
    for measure in measures}
best = rows[rows.error_rank == 1]
worst = rows[rows.error_rank == 14]
print(json.dumps({'raw_candidates_sha256': digest,
    'sections': rows.section_id.nunique(), 'candidates': len(rows),
    'within_section_rho_with_lower_error': correlations,
    'mean_best': best[list(measures) + ['mapped96_error_um']].mean().to_dict(),
    'mean_worst': worst[list(measures) + ['mapped96_error_um']].mean().to_dict(),
    'head_site_auc': json.loads((run / 'summary.json').read_text())['20000'][
        'independent_sites']['auroc']}, indent=2))
