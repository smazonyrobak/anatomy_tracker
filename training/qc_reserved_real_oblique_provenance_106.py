"""Exact TRAIN-donor lineage and bounded QC for virtual real-stack oblique cuts."""
import hashlib
import json
from pathlib import Path

import numpy as np

from training.arbitrary_plane_reserved_real_oblique_stream import sample_reserved_real_oblique_train


def sample_traceable_reserved_real_oblique_train(context, donor_index, seed, side=192):
    donor = context['donors'][int(donor_index)]
    record_path = Path(donor['images']).parent / 'records.jsonl'
    records = [json.loads(line) for line in record_path.open()]
    sections = np.asarray([row['section_number'] for row in records])
    affines = np.asarray([row['model_pixel_to_ap_dv_ml_um'] for row in records])
    steps = np.diff(affines[:, :, 2], axis=0)
    assert all(row['training_split'] == 'train' and str(row['animal_id']) == str(donor['animal_id'])
               for row in records)
    assert np.all(np.diff(sections) == 1)
    assert np.max(np.abs(affines[:, :, :2] - affines[0, :, :2])) < 1e-5
    assert np.max(np.abs(steps - np.median(steps, axis=0))) < 1e-4
    sample = sample_reserved_real_oblique_train(context, donor_index, seed, side=side, device='cpu')
    order = np.argsort(donor['state'][:, 0])
    lo, hi = sample['provenance'][0]['real_oblique']['source_sorted_index_span']
    fields = ('animal_partition_key', 'animal_id_namespace', 'animal_id', 'specimen_id',
              'experiment_id', 'section_id_namespace', 'section_id', 'section_number',
              'donor_array_row', 'training_split', 'source_image_path', 'source_image_sha256')
    sample['provenance'][0]['real_oblique'].update({
        'source_records_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest(),
        'possible_source_sections': [{key: records[int(order[i])][key] for key in fields}
                                     for i in range(lo, hi + 1)],
        'source_span_role': 'all donor sections within sampled slab index span; potential interpolation sources',
        'virtual_slab_thickness_um': float(sample['offsets'].max() - sample['offsets'].min()),
        'source_section_thickness_um': float(records[0]['section_thickness_um']),
        'acquisition_role': 'virtual trilinear reslice of serial coronal TRAIN images, not a physically acquired oblique section',
        'target_role': 'weak Allen affine CCF plus synthetic image deformation, not expert pose/deformation truth',
        'appearance_caveat': 'the current sampler fixes background appearance at zero; donor volume exteriors remain near black',
    })
    return sample


if __name__ == '__main__':
    from PIL import Image, ImageDraw
    import torch

    torch.set_num_threads(2)
    root = Path('I:/AnatomyTracker/data/joint_v7_reserved_train_images_192_001')
    out = Path('I:/AnatomyTracker/runs/reserved_real_oblique_provenance_106_qc')
    donor_ids = (112687682, 112707797, 112714260, 112779863)
    donors = []
    for animal_id in donor_ids:
        directory = root / f'donor_{animal_id}'
        records = [json.loads(line) for line in (directory / 'records.jsonl').open()]
        with np.load(directory / 'geometry.npz') as geometry:
            state = geometry['state'].astype(np.float32)
        donors.append({'animal_id': str(animal_id), 'images': directory / 'images.npy',
                       'state': state, 'identities': [{key: row[key] for key in
                           ('animal_id', 'specimen_id', 'experiment_id', 'section_id')}
                           for row in records]})
    sheet = Image.new('RGB', (384, 428), 'white')
    draw = ImageDraw.Draw(sheet)
    receipt = {'seed_base': 20261008106, 'scope': 'four fixed-seed TRAIN-only CPU samples; no training',
               'acquisition_role': 'virtual oblique views of registered serial coronal images, not physically acquired oblique histology',
               'target_role': 'weak Allen affine CCF and synthetic deformation; no expert ground truth',
               'appearance_caveat': 'the current sampler fixes background appearance at zero; donor volume exteriors remain near black',
               'samples': []}
    for i, animal_id in enumerate(donor_ids):
        seed = 20261008106 + i
        sample = sample_traceable_reserved_real_oblique_train({'donors': donors}, i, seed)
        image = np.uint8(np.clip(sample['inputs'][0, 0].numpy(), 0, 1) * 255)
        x, y = 192 * (i % 2), 214 * (i // 2)
        sheet.paste(Image.fromarray(image).convert('RGB'), (x, y + 22))
        draw.text((x + 4, y + 4), f'donor {animal_id}  seed {seed}', fill='black')
        provenance = sample['provenance'][0]
        receipt['samples'].append({'donor': animal_id, 'seed': seed,
            'valid_fraction': float(sample['valid_mask'][0].float().mean()),
            'source_state': provenance['one_shot']['source_state'],
            'observed_state': provenance['one_shot']['observed_affine_state'],
            'provenance': provenance['real_oblique']})
    out.mkdir(parents=True, exist_ok=True)
    sheet.save(out / 'four_virtual_obliques.png')
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'output': str(out), 'valid_fractions':
        [row['valid_fraction'] for row in receipt['samples']]}))
