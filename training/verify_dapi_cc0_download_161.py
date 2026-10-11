"""Freeze the downloaded CC0 DAPI source bytes and native NIfTI geometry."""

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


source = Path('I:/AnatomyTracker/data/g_node_dapi_cc0_individual_volumes_20261011')
doi = '10.12751/g-node.16wrxa'
tag = f'https://gin.g-node.org/doi/B6_dapi_template/raw/{doi}/'
ids = ('BM03', 'BM04', 'BM06', 'BM10', 'BM11', 'BM12', 'BM13', 'BM14',
       'BM15', 'BM16', 'BM17', 'BM18')
files = [f'{animal}.nii' for animal in ids] + [
    'dapi_template.nii.gz', 'ccf_to_dapi_affine_transformation.txt', 'LICENSE']
assert all((source / name).is_file() for name in files)

records = {}
for name in files:
    path = source / name
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    item = {'bytes': path.stat().st_size, 'sha256': digest,
            'immutable_source_url': (f'{tag}data/mice/reconstructed_brain_volumes/{name}'
                                     if name.endswith('.nii') else tag + name)}
    if name == 'BM03.nii':
        item['download_revision'] = 'master (hash-frozen locally; DOI-tag recovery URL above)'
    if name.endswith(('.nii', '.nii.gz')):
        image = nib.load(path)
        item.update(shape=list(image.shape), dtype=str(image.header.get_data_dtype()),
                    header_voxel_size=[float(value) for value in image.header.get_zooms()[:3]],
                    spatial_unit=image.header.get_xyzt_units()[0],
                    sform_code=int(image.header['sform_code']),
                    selected_affine=image.affine.tolist())
        if name.endswith('.nii'):
            expected = int(image.dataobj.offset) + int(np.prod(image.shape)) * \
                       image.header.get_data_dtype().itemsize
            assert item['bytes'] == expected
    records[name] = item

assert len({records[f'{animal}.nii']['sha256'] for animal in ids}) == len(ids)
manifest = {'dataset_doi': doi, 'source_repository':
    'https://gin.g-node.org/doi/B6_dapi_template/src/10.12751/g-node.16wrxa/',
    'license_scope': 'G-Node repository DAPI intensities, code and affine under CC0; Allen-derived CCF/reference/annotation separately governed by Allen terms',
    'coordinate_status': 'BM native identity sforms conflict with header pixdim; per-animal registration to DAPI average and anatomical QC required before CCF labels',
    'physical_oblique_truth': False, 'animal_ids': ids, 'files': records}
(source / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'animal_volumes': len(ids), 'source_bytes': sum(x['bytes'] for x in records.values()),
                  'manifest': str(source / 'source_manifest.json')}))
