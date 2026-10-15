"""Recover the frozen 319 decision after the runner's JSON-bool write error."""
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

run = Path('I:/AnatomyTracker/runs/red_source_diversity_319_001')
config = json.loads((run / 'config.json').read_text())
assert config['updates_per_arm'] == 2000 and config['readout_steps'] == [0, 1000, 2000]
expected = {'ccf': 128, 'red_182849231': 64, 'red_517337553': 64, 'AL1A': 52}
readouts = {}
for arm in ('control', 'red'):
    for step in config['readout_steps']:
        item = json.loads((run / f'{arm}_{step:04d}_readout.json').read_text())
        assert len(item['rows']) == 308 and set(item['metrics']) == set(expected)
        for cohort, count in expected.items():
            errors = np.array([row['first_mm'] for row in item['rows'] if row['cohort'] == cohort])
            assert len(errors) == count and np.isfinite(errors).all()
            assert abs(errors.mean() - item['metrics'][cohort]['first_mean_mm']) < 1e-12
            assert int((errors > 5).sum()) == item['metrics'][cohort]['first_gt5']
        readouts[(arm, step)] = item
assert readouts['control', 0]['rows'] == readouts['red', 0]['rows']
assert readouts['control', 0]['metrics'] == readouts['red', 0]['metrics']
assert len(json.loads((run / 'schedule.json').read_text())) == 2000
assert all((run / f'{arm}_{step:04d}.pt').stat().st_size > 0
           for arm in ('control', 'red') for step in (1000, 2000))
with ZipFile('I:/AnatomyTracker/data/lupori_joint_185/rawData.zip') as archive:
    valid = {item.findtext('name') for item in ElementTree.fromstring(
        archive.read('AL1A/AL1A-info.xml')).findall('slices')
        if item.findtext('valid') == '1'}
    registration = json.loads(archive.read('AL1A/AL1A-visualign.json'))
    sections = [Path(item['filename']).name.removesuffix('-thumb.png')
                for item in registration['slices']
                if Path(item['filename']).name.removesuffix('-thumb.png') in valid]
assert len(sections) == 52
assert all(row['cohort'] == 'AL1A' for row in readouts['control', 2000]['rows'][256:])
(run / 'AL1A_row_provenance.json').write_text(json.dumps([
    {'row_index': 256 + i, 'dataset': 'Lupori185', 'animal_id': 'AL1A', 'section_id': section}
    for i, section in enumerate(sections)], indent=2) + '\n')

control = readouts['control', 2000]['metrics']
red = readouts['red', 2000]['metrics']
donors = ('red_182849231', 'red_517337553')
control_dev = np.mean([control[donor]['first_mean_mm'] for donor in donors])
red_dev = np.mean([red[donor]['first_mean_mm'] for donor in donors])
gate = {'ccf_gain_ge0p35': bool(control['ccf']['first_mean_mm'] - red['ccf']['first_mean_mm'] >= .35),
        'ccf_gt5_not_worse': bool(red['ccf']['first_gt5'] <= control['ccf']['first_gt5']),
        'dev_red_gain_ge0p20': bool(control_dev - red_dev >= .20),
        'AL1A_regression_le0p10': bool(red['AL1A']['first_mean_mm'] - control['AL1A']['first_mean_mm'] <= .10)}
decision = {'gate': gate, 'pass': all(gate.values()), 'control': control, 'red': red,
    'effects_mm': {'ccf_gain': control['ccf']['first_mean_mm'] - red['ccf']['first_mean_mm'],
                   'dev_red_gain': float(control_dev - red_dev),
                   'AL1A_regression': red['AL1A']['first_mean_mm'] - control['AL1A']['first_mean_mm']},
    'runner_exit': 'exit 1 after all readouts/checkpoints/schedule: NumPy bool JSON serialization',
    'recovery': 'decision recomputed from frozen per-case readouts; no training rerun'}
(run / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
print(json.dumps(decision, indent=2))
