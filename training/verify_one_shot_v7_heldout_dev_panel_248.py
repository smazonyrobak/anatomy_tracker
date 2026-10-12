"""Independent read-only audit of the frozen 248 synthetic DEV panel."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

root = Path('I:/AnatomyTracker/data/one_shot_v7_heldout_dev_panel_248_004')
repo = Path(__file__).resolve().parents[1]
done = json.loads((root / 'completed.json').read_text())
protocol = json.loads((root / 'protocol.json').read_text())
rows = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]

assert hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest() == done['protocol_sha256']
assert hashlib.sha256((root / 'records.jsonl').read_bytes()).hexdigest() == done['records_sha256']
assert hashlib.sha256((repo / 'docs/publication/ONE_SHOT_V7_HELDOUT_DEV_PANEL_248_PROTOCOL_20261012.md').read_bytes()).hexdigest() == protocol['protocol_document_sha256']
for name, digest in protocol['source_sha256'].items():
    assert hashlib.sha256((repo / 'training' / name).read_bytes()).hexdigest() == digest
assert len(rows) == done['physical_sections'] == 256
assert len(done['subjects']) == done['synthetic_subjects'] == 8
assert len({r['provenance']['physical_section_id'] for r in rows}) == 256
assert len({r['draw_seed'] for r in rows}) == 256
assert len({r['section_id'] for r in rows}) == 256
assert len({r['synthetic_subject_plan_id'] for r in rows}) == 8
assert len({r['synthetic_subject_realization_id'] for r in rows}) == 8
assert len({r['animal_id'] for r in rows}) == 8
assert Counter(r['appearance_mode'] for r in rows) == done['modes']
assert sum(r['eligible'] for r in rows) == done['eligible'] == 227
assert sum(not r['eligible'] for r in rows) == done['ineligible'] == 29

normal_bins = Counter()
for r in rows:
    path = root / r['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['sha256']
    assert r['split'] == r['provenance']['split'] == 'development'
    assert r['provenance']['base_lineage']['animal_id'] == r['animal_id']
    assert r['provenance']['subject_plan_receipt_sha256'] == r['plan_receipt_sha256']
    assert r['provenance']['inverse_map_sha256'] == r['inverse_map_sha256']
    assert r['appearance_mode'] == r['provenance']['mode']
    n = np.asarray(r['provenance']['virtual_unit_normal'])
    assert np.isclose(np.linalg.norm(n), 1, atol=1e-5)
    angle = np.degrees(np.arccos(np.clip(np.abs(n).max(), 0, 1)))
    normal_bins[int(angle // 15) * 15] += 1
    with np.load(path, allow_pickle=False) as a:
        assert a['inputs'].shape == (5, 192, 192)
        assert a['target_state'].shape == (12,)
        assert a['target_centre_um'].shape == (192, 192, 3)
        assert a['valid_mask'].shape == a['visible_support'].shape == (192, 192)
        assert a['offsets_um'].shape == a['weights'].shape == (9,)
        assert np.isfinite(a['inputs']).all() and np.isfinite(a['target_state']).all()
        assert np.isfinite(a['target_centre_um']).all()
        assert np.isclose(a['weights'].sum(), 1, atol=1e-6)
        assert int(a['valid_mask'].sum()) == r['valid_pixels']
        assert bool(a['eligible']) == r['eligible']
        assert np.isclose(float(a['source_visible_support_mass']), r['source_visible_support_mass'], atol=.01)
        assert r['eligible'] == (r['source_visible_support_mass'] >= 192 * 192 / 144
                                 and r['valid_pixels'] >= 192 * 192 / 144)

for subject in done['subjects']:
    path = root / subject['file']
    directory = path.parent
    index = int(directory.name.removeprefix('subject_'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == subject['sha256']
    assert hashlib.sha256((directory / 'completed.json').read_bytes()).hexdigest() == done['subject_completed_sha256'][str(index)]
    record = json.loads(path.read_text())
    completed = json.loads((directory / 'completed.json').read_text())
    animal_rows = [r for r in rows if r['animal_id'] == record['lineage']['animal_id']]
    assert len(animal_rows) == 32 and completed['records'] == animal_rows
    assert {r['section_index'] for r in animal_rows} == set(range(32))
    assert completed['subject_index_record'] == subject
    assert completed['plan_receipt_sha256'] == subject['plan_receipt_sha256']
    assert completed['local_field_sha256'] == subject['local_field_sha256']
    assert completed['inverse_map_sha256'] == subject['inverse_map_sha256']
    for name, digest in record['plan_artifact_sha256'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    map_path = directory / 'inverse_map.npz'
    assert hashlib.sha256(map_path.read_bytes()).hexdigest() == subject['inverse_map_sha256']
    with np.load(map_path, allow_pickle=False) as a:
        error = np.linalg.norm(a['check_exact_ccf_um'] - a['check_approximate_ccf_um'], axis=1)
        assert len(error) == 4096
        assert error.max() <= 5
        assert np.isclose(error.max(), record['mapping_checks'][-1]['max_error_um'])
        assert tuple(a['displacement_um'].shape[:3]) == tuple(record['mapping_checks'][-1]['shape'])

print(json.dumps({'status': 'verified', 'subjects': 8, 'attempted_sections': 256,
                  'eligible': 227, 'ineligible': 29, 'modes': done['modes'],
                  'nearest_principal_angle_bins_15deg': dict(sorted(normal_bins.items())),
                  'scope': 'same-atlas synthetic DEV only; not biological validation or a benchmark'}))
