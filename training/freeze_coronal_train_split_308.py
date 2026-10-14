"""Freeze a new donor-disjoint TRAIN-inner split before reading image pixels."""
import hashlib
import json
from pathlib import Path

root = Path('I:/AnatomyTracker')
corpus = root / 'data/joint_v7_reserved_train_images_192_001'
old = root / 'data/joint_v7_allen_fullcanvas_192_001/records.jsonl'
output = root / 'data/coronal_train_split_308_001'
manifest_bytes = (corpus / 'completed.json').read_bytes()
manifest = json.loads(manifest_bytes)
assert manifest['training_role'] == 'train_only' and manifest['training_donors'] == 1885
assert manifest['training_images'] == 263754

with old.open('rb') as stream:
    earlier_lines = [stream.readline() for _ in range(1280)]
earlier = [json.loads(line) for line in earlier_lines]
earlier_keys = {row['animal_partition_key'] for row in earlier}
assert len(earlier_keys) == 58 and all(row['training_split'] == 'train' for row in earlier)

donors = {}
for animal_id, receipt_hash in manifest['donor_receipts'].items():
    directory = corpus / f'donor_{animal_id}'
    receipt_bytes = (directory / 'completed.json').read_bytes()
    assert hashlib.sha256(receipt_bytes).hexdigest() == receipt_hash
    receipt = json.loads(receipt_bytes)
    with (directory / 'records.jsonl').open(encoding='utf8') as stream:
        first = json.loads(stream.readline())
    key = first['animal_partition_key']
    assert first['training_split'] == 'train' and key not in donors
    assert first['animal_id'] == int(animal_id) and receipt['image_count'] > 0
    donors[key] = {'animal_id': int(animal_id), 'image_count': receipt['image_count'],
                   'receipt_sha256': receipt_hash}
assert len(donors) == 1885 and earlier_keys <= donors.keys()
assert sum(row['image_count'] for row in donors.values()) == 263754

available = donors.keys() - earlier_keys
ranked = sorted(available, key=lambda key: hashlib.sha256(
    f'coronal-308-inner-20261014:{key}'.encode()).hexdigest())
heldout = ranked[:100]
gradient = sorted(available - set(heldout))
readout = []
for key in heldout:
    directory = corpus / f"donor_{donors[key]['animal_id']}"
    rows = [json.loads(line) for line in (directory / 'records.jsonl').open(encoding='utf8')]
    assert len(rows) == donors[key]['image_count']
    assert all(row['animal_partition_key'] == key and row['training_split'] == 'train'
               for row in rows)
    row = sorted(rows, key=lambda item: (item['section_number'], item['section_id']))[
        (len(rows) - 1) // 2]
    readout.append({'animal_partition_key': key, 'animal_id': row['animal_id'],
                    'specimen_id': row['specimen_id'], 'experiment_id': row['experiment_id'],
                    'section_id': row['section_id'], 'donor_array_row': row['donor_array_row'],
                    'source_image_sha256': row['source_image_sha256'],
                    'model_pixel_to_ap_dv_ml_um': row['model_pixel_to_ap_dv_ml_um'],
                    'label_role': row['label_role']})

split = {'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
         'earlier_1280_rows_sha256': hashlib.sha256(b''.join(earlier_lines)).hexdigest(),
         'rule': 'first 100 fresh partition keys by SHA256(coronal-308-inner-20261014:<key>); median section by section_number, section_id',
         'prior_pool_excluded_from_308_gradients': sorted(earlier_keys),
         'new_inner_readout': readout, 'gradient_donors': gradient,
         'donor_shards': donors,
         'counts': {'source_donors': len(donors), 'old_excluded': len(earlier_keys),
                    'new_inner': len(heldout), 'gradient_donors': len(gradient)}}
output.mkdir(parents=True, exist_ok=False)
(output / 'split.json').write_text(json.dumps(split, indent=2) + '\n', encoding='utf8')
print(json.dumps({'output': str(output / 'split.json'), **split['counts']}))
