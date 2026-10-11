"""Check Lupori TRAIN VisuAlign planes and image-to-atlas displacement labels."""

import io
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

archive = Path('I:/AnatomyTracker/data/lupori_joint_185/rawData.zip')
split = json.loads(Path(__file__).with_name('lupori_animal_split_185.json').read_text())
assert archive.stat().st_size == split['archive_bytes']
records = []
pairing = []

with ZipFile(archive) as source:
    names = set(source.namelist())
    for animal in split['train']:
        root = ElementTree.fromstring(source.read(f'{animal}/{animal}-info.xml'))
        info = {node.findtext('name'): node for node in root.findall('slices')}
        valid = {name for name, node in info.items() if node.findtext('valid') == '1'}
        visualign = json.loads(source.read(f'{animal}/{animal}-visualign.json'))
        assert visualign['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
        assert visualign['target-resolution'] == [456, 528, 320]
        section_names = [Path(s['filename']).name.removesuffix('-thumb.png') for s in visualign['slices']]
        assert len(section_names) == len(set(section_names))
        pairing.append({'animal_id': animal, 'valid_info_sections': len(valid),
            'visualign_sections': len(section_names),
            'valid_without_visualign': sorted(valid - set(section_names)),
            'visualign_not_valid_info': sorted(set(section_names) - valid)})
        for section in visualign['slices']:
            name = Path(section['filename']).name.removesuffix('-thumb.png')
            thumbnail = f'{animal}/{section["filename"]}'
            mask = f'{animal}/masks/{name}-mask.png'
            field_x = f'{animal}/dispFields/{name}-dispFieldX.csv'
            field_y = f'{animal}/dispFields/{name}-dispFieldY.csv'
            assert {thumbnail, mask, field_x, field_y} <= names, name
            with Image.open(io.BytesIO(source.read(thumbnail))) as image:
                width, height = image.size
            with Image.open(io.BytesIO(source.read(mask))) as image:
                tissue = np.asarray(image, dtype=bool)
            assert (width, height) == (section['width'], section['height']), name
            dx = np.loadtxt(io.BytesIO(source.read(field_x)), delimiter=',')
            dy = np.loadtxt(io.BytesIO(source.read(field_y)), delimiter=',')
            assert dx.shape == dy.shape == (round(height * .2), round(width * .2)), name
            assert np.isfinite(dx).all() and np.isfinite(dy).all(), name
            markers = np.asarray(section['markers'], dtype=float).reshape(-1, 4)
            assert len(markers) > 0, name
            expected = markers[:, :2] - markers[:, 2:]
            coords = [markers[:, 3] * .2 - 1, markers[:, 2] * .2 - 1]
            sampled = np.stack([map_coordinates(field, coords, order=1, mode='nearest')
                                for field in (dx, dy)], axis=1)
            residual = np.linalg.norm(sampled - expected, axis=1)
            reverse_residual = np.linalg.norm(sampled + expected, axis=1)
            jac_xx = 1 + np.gradient(dx, axis=1) * .2
            jac_xy = np.gradient(dx, axis=0) * .2
            jac_yx = np.gradient(dy, axis=1) * .2
            jac_yy = 1 + np.gradient(dy, axis=0) * .2
            determinant = jac_xx * jac_yy - jac_xy * jac_yx
            yy, xx = np.indices(dx.shape)
            tissue_grid = map_coordinates(tissue.astype(np.uint8),
                [(yy + 1) / .2 - 1, (xx + 1) / .2 - 1], order=0, mode='nearest') > 0
            o, u, v = np.asarray(section['anchoring']).reshape(3, 3)
            corners = np.stack([o, o + u, o + v, o + u + v])
            center = o + .5 * u + .5 * v
            normal = np.cross(u, v)
            records.append({
                'animal_id': animal, 'section_id': name, 'valid_info': name in valid,
                'section_number': int(info[name].findtext('number')),
                'width': width, 'height': height,
                'marker_count': len(markers), 'marker_median_error_px': float(np.median(residual)),
                'marker_rms_error_px': float(np.sqrt(np.mean(residual ** 2))),
                'marker_max_error_px': float(np.max(residual)),
                'reverse_marker_rms_error_px': float(np.sqrt(np.mean(reverse_residual ** 2))),
                'field_median_displacement_px': float(np.median(np.hypot(dx, dy))),
                'tissue_median_displacement_px': float(np.median(np.hypot(dx[tissue_grid], dy[tissue_grid]))),
                'field_p99_displacement_px': float(np.percentile(np.hypot(dx, dy), 99)),
                'tissue_p99_displacement_px': float(np.percentile(np.hypot(dx[tissue_grid], dy[tissue_grid]), 99)),
                'field_max_displacement_px': float(np.max(np.hypot(dx, dy))),
                'folded_grid_fraction': float(np.mean(determinant <= 0)),
                'folded_tissue_fraction': float(np.mean(determinant[tissue_grid] <= 0)),
                'tissue_grid_fraction': float(np.mean(tissue_grid)),
                'plane_angle_from_coronal_deg': float(np.degrees(np.arccos(np.clip(
                    abs(normal[1]) / np.linalg.norm(normal), 0, 1)))),
                'plane_center_ml_vox': float(center[0]),
                'plane_center_ap_vox': float(center[1]),
                'plane_center_dv_vox': float(center[2]),
                'atlas_center_in_bounds': bool(np.all((center >= 0) &
                    (center < [456, 528, 320]))),
                'atlas_corners_in_bounds': int(np.sum(np.all((corners >= 0) &
                    (corners < [456, 528, 320]), axis=1))),
                'source_visualign_json': f'{animal}/{animal}-visualign.json',
                'source_thumbnail': thumbnail, 'source_displacement_x': field_x,
                'source_displacement_y': field_y,
            })

output = archive.with_name('train_label_qc.json')
output.write_text(json.dumps({'source_doi': split['source_doi'],
    'archive_md5': split['archive_md5'], 'split': 'train', 'pairing': pairing,
    'sections': records}, indent=2))
print('sections', len(records), 'animals', sorted({r['animal_id'] for r in records}))
print('valid_sections', sum(r['valid_info'] for r in records),
      'invalid_sections', [r['section_id'] for r in records if not r['valid_info']])
print('pairing', pairing)
for animal in split['train']:
    ordered = sorted((r for r in records if r['animal_id'] == animal and r['valid_info']),
                     key=lambda r: r['section_number'])
    numbers = np.array([r['section_number'] for r in ordered])
    ap = np.array([r['plane_center_ap_vox'] for r in ordered])
    print('ap_series', animal, 'first/last voxel', round(ap[0], 1), round(ap[-1], 1),
          'corr_with_section_number', round(np.corrcoef(numbers, ap)[0, 1], 3),
          'nonmonotone_steps', int(np.sum(np.diff(ap) <= 0)))
for field in ('marker_count', 'marker_rms_error_px', 'reverse_marker_rms_error_px',
              'tissue_median_displacement_px', 'tissue_p99_displacement_px',
              'field_max_displacement_px', 'folded_grid_fraction', 'folded_tissue_fraction',
              'plane_angle_from_coronal_deg', 'atlas_corners_in_bounds'):
    values = np.array([r[field] for r in records])
    print(field, 'min/median/p95/max', np.percentile(values, [0, 50, 95, 100]))
print('worst_marker_sections', [(r['section_id'], round(r['marker_rms_error_px'], 2))
    for r in sorted(records, key=lambda r: r['marker_rms_error_px'], reverse=True)[:10]])
print('folded_sections', [(r['section_id'], round(r['folded_grid_fraction'], 3))
    for r in sorted(records, key=lambda r: r['folded_grid_fraction'], reverse=True)[:10]])
print('output', output)
