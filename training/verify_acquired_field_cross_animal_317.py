"""Read-only provenance and row-arithmetic check of the frozen 317 assay."""
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

ROOT = Path('I:/AnatomyTracker')
SOURCE = Path(__file__).resolve()
OUT = ROOT / 'reports/acquired_field_cross_animal_317'
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
PROTOCOL = SOURCE.parents[1] / 'docs/publication/ACQUIRED_FIELD_CROSS_ANIMAL_317_PROTOCOL_20261015.md'
SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
assert SOURCE.drive.upper() == OUT.drive.upper() == 'I:'
config = json.loads((OUT / 'config.json').read_text(encoding='utf8'))
done = json.loads((OUT / 'completed.json').read_text(encoding='utf8'))
summary = json.loads((OUT / 'summary.json').read_text(encoding='utf8'))
rows = [json.loads(line) for line in (OUT / 'rows.jsonl').read_text(
    encoding='utf8').splitlines()]
split = json.loads(SPLIT.read_text(encoding='utf8'))
assert config['animals'] == ['CC2B', 'CC4B', 'CC4A']
assert config['source_doi'] == split['source_doi']
assert config['archive_md5'] == split['archive_md5']
assert config['archive_bytes'] == ARCHIVE.stat().st_size == split['archive_bytes']
assert config['protocol_sha256'] == done['protocol_sha256'] == hashlib.sha256(
    PROTOCOL.read_bytes()).hexdigest()
assert config['split_sha256'] == hashlib.sha256(SPLIT.read_bytes()).hexdigest()
for name, digest in config['source_sha256'].items():
    assert digest == done['source_sha256'][name] == hashlib.sha256(
        SOURCE.with_name(name).read_bytes()).hexdigest()
for name, digest in done['output_sha256'].items():
    assert digest == hashlib.sha256((OUT / name).read_bytes()).hexdigest()
assert len(rows) == done['sections'] == summary['sections'] == 188
assert len({(row['animal_id'], row['section_id']) for row in rows}) == len(rows)

with ZipFile(ARCHIVE) as source_zip:
    for animal in config['animals']:
        valid = {item.findtext('name') for item in ElementTree.fromstring(
            source_zip.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if item.findtext('valid') == '1'}
        group = [row for row in rows if row['animal_id'] == animal]
        assert {row['section_id'] for row in group} == valid
        for row in group:
            section = row['section_id']
            assert row['specimen_id'] == animal and row['experiment_id'] is None
            members = (row['source_thumbnail'],
                f'{animal}/dispFields/{section}-dispFieldX.csv',
                f'{animal}/dispFields/{section}-dispFieldY.csv',
                f'{animal}/masks/{section}-mask.png')
            keys = ('source_thumbnail_sha256', 'displacement_x_sha256',
                    'displacement_y_sha256', 'mask_sha256')
            for member, key in zip(members, keys):
                assert hashlib.sha256(source_zip.read(member)).hexdigest() == row[key]
            if row['score'] is not None:
                assert row['common_sites_96'] >= 400
                assert all(len(row['score'][metric]) == 4 and
                           np.isfinite(row['score'][metric]).all()
                           for metric in config['scores'])
            else:
                assert row['common_sites_96'] < 400

for animal in config['animals']:
    group = [row for row in rows if row['animal_id'] == animal]
    eligible = [row for row in group if row['score'] is not None]
    result = summary['animals'][animal]
    assert (len(group), len(eligible)) == (result['sections'], result['eligible_sections'])
    passed = len(eligible) >= .9 * len(group)
    for metric in config['scores']:
        values = np.asarray([row['score'][metric] for row in eligible])
        a, b = values[:, 1] - values[:, 0], values[:, 1] - values[:, 2]
        item = result['scores'][metric]
        assert np.allclose(values.mean(0), item['mean'], atol=1e-10)
        assert abs(a.mean() - item['human_minus_zero_mean']) < 1e-10
        assert abs(np.median(a) - item['human_minus_zero_median']) < 1e-10
        assert abs(b.mean() - item['human_minus_negative_mean']) < 1e-10
        assert abs(np.median(b) - item['human_minus_negative_median']) < 1e-10
        assert (int((a > 0).sum()), int((b > 0).sum())) == (
            item['human_over_zero_sections'], item['human_over_negative_sections'])
        passed &= bool((a > 0).sum() >= .6 * len(eligible)
                       and (b > 0).sum() >= .6 * len(eligible)
                       and np.median(a) > 0 and np.median(b) > 0)
    assert passed == result['consistency_passed']
assert summary['consistency_gate_passed'] == all(
    result['consistency_passed'] for result in summary['animals'].values())
assert not summary['consistency_gate_passed']
print(json.dumps({'verified': True, 'sections': len(rows),
    'animal_gate': {animal: summary['animals'][animal]['consistency_passed']
                    for animal in config['animals']}, 'consistency_gate_passed': False}))
