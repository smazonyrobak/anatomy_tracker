"""No-update cross-animal acquired field image-information assay."""
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(ROOT / 'tmp')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import map_coordinates

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
    render_finite_thickness_coordinate_grid)
from training.arbitrary_plane_geometry import (
    physical_ouv_to_frame, quicknii_to_allen_points, quicknii_to_allen_vectors)
from training.arbitrary_plane_ribbon_v6 import project_surface_affine_out

SOURCE = Path(__file__).resolve()
PROTOCOL = SOURCE.parents[1] / 'docs/publication/ACQUIRED_FIELD_CROSS_ANIMAL_317_PROTOCOL_20261015.md'
SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
OUT = ROOT / 'reports/acquired_field_cross_animal_317'
ANIMALS = ('CC2B', 'CC4B', 'CC4A')
assert SOURCE.drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
split = json.loads(SPLIT.read_text(encoding='utf8'))
assert split['train'] == ['AL1A', *ANIMALS] and ARCHIVE.stat().st_size == split['archive_bytes']
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
side, grid = 192, 96
axis = (torch.arange(grid, device='cuda') + .5) / grid - .5 / side
y, x = torch.meshgrid(axis, axis, indexing='ij')
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
fields = ('zero', 'human_affine_free', 'negative_human_affine_free', 'raw_full_human')
scores = ('abs_pearson_3x3', 'negative_mind8_3px_mse')
source_files = ('assay_acquired_field_cross_animal_317.py',
    'arbitrary_plane_allen_atlas_binding_v6.py',
    'arbitrary_plane_full_frame_primitives.py', 'arbitrary_plane_geometry.py',
    'arbitrary_plane_ribbon_v6.py')
config = {'animals': ANIMALS, 'role': 'three TRAIN gradient animals, no-update information audit',
    'source_doi': split['source_doi'], 'archive_md5': split['archive_md5'],
    'archive_bytes': split['archive_bytes'],
    'split_sha256': hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'source_sha256': {name: hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()
                      for name in source_files},
    'input_side': side, 'map_side': grid, 'offsets_um': offsets.tolist(),
    'weights_normalized_by_renderer': weights.tolist(), 'fields': fields, 'scores': scores,
    'common_mask': 'source tissue and all four atlas supports >0.5, eroded 7x7; >=400 sites',
    'calibrated': False, 'public_benchmark_used': False, 'model_checkpoint_used': False}
OUT.mkdir(parents=True)
(OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

rows = []
with ZipFile(ARCHIVE) as source_zip, (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    for animal in ANIMALS:
        valid = {item.findtext('name') for item in ElementTree.fromstring(
            source_zip.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if item.findtext('valid') == '1'}
        registration = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
        assert registration['target'] == 'ABA_Mouse_CCFv3_2017_25um.cutlas'
        seen = set()
        for metadata in registration['slices']:
            section = Path(metadata['filename']).name.removesuffix('-thumb.png')
            if section not in valid:
                continue
            seen.add(section)
            image_name = f'{animal}/{metadata["filename"]}'
            image_bytes = source_zip.read(image_name)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            width, height = image.size
            rgb = np.asarray(image.resize((side, side), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
            inputs = torch.zeros(1, 5, side, side, device='cuda')
            inputs[0, 0] = torch.from_numpy(gray).cuda()
            q = torch.tensor(metadata['anchoring'], device='cuda', dtype=torch.float32).reshape(3, 3)
            ouv = torch.stack((quicknii_to_allen_points(q[0]),
                quicknii_to_allen_vectors(q[1]), quicknii_to_allen_vectors(q[2]))) * 25
            state = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
            frame = full_frame_state_to_components(state)[1]
            dx_bytes = source_zip.read(f'{animal}/dispFields/{section}-dispFieldX.csv')
            dy_bytes = source_zip.read(f'{animal}/dispFields/{section}-dispFieldY.csv')
            mask_bytes = source_zip.read(f'{animal}/masks/{section}-mask.png')
            dx = np.loadtxt(BytesIO(dx_bytes), delimiter=',')
            dy = np.loadtxt(BytesIO(dy_bytes), delimiter=',')
            query = [(y * height * .2 - 1).cpu().numpy(),
                     (x * width * .2 - 1).cpu().numpy()]
            sampled_x = torch.from_numpy(map_coordinates(dx, query, order=1, mode='nearest')).cuda().float()
            sampled_y = torch.from_numpy(map_coordinates(dy, query, order=1, mode='nearest')).cuda().float()
            displacement = sampled_x[..., None] * ouv[1] / width + sampled_y[..., None] * ouv[2] / height
            local = torch.einsum('ij,hwj->ihw', frame.T, displacement)
            human = project_surface_affine_out(local[None])[0][0]
            mask = torch.from_numpy(np.asarray(Image.open(BytesIO(mask_bytes)).convert('L').resize(
                (grid, grid), Image.Resampling.NEAREST)) > 127).cuda()

            with torch.inference_mode():
                candidate_fields = torch.stack((torch.zeros_like(human), human, -human, local))
                plane = ouv[0] + x[..., None] * ouv[1] + y[..., None] * ouv[2]
                surface = plane[None] + torch.einsum('ij,bjhw->bhwi', frame, candidate_fields)
                world = surface[:, None] + offsets[None, :, None, None, None] * frame[:, 2]
                rendered = render_finite_thickness_coordinate_grid(
                    atlas, world, (0., 0., 0.), (25., 25., 25.), weights.expand(4, -1))
                support = rendered[:, 1].clamp(0, 1)
                atlas_images = rendered[:, :1] / support[:, None].clamp_min(1e-4)
                common = mask & (support > .5).all(0)
                common = F.avg_pool2d(common.float()[None, None], 7, 1, 3)[0, 0] > .999
                sites = int(common.sum())
                score = None
                if sites >= 400:
                    observed = F.interpolate(inputs[:, :1], (grid, grid), mode='area')
                    images = torch.cat((observed, atlas_images), 0)
                    smoothed = F.avg_pool2d(images, 3, 1, 1)[:, 0]
                    source = smoothed[0]
                    targets = smoothed[1:]
                    source_centered = (source - source[common].mean()) * common
                    target_centered = (targets - targets[:, common].mean(-1)[:, None, None]) * common
                    pearson = ((source_centered[None] * target_centered).sum((-2, -1)) /
                        (source_centered.square().sum() * target_centered.square().sum((-2, -1)))
                        .sqrt().clamp_min(1e-8)).abs()
                    padded = F.pad(images, (3, 3, 3, 3), mode='replicate')
                    distances = torch.cat([F.avg_pool2d((images - padded[:, :, 3 + dy:99 + dy,
                        3 + dxx:99 + dxx]).square(), 3, 1, 1) for dy, dxx in
                        ((3, 0), (-3, 0), (0, 3), (0, -3), (3, 3), (3, -3), (-3, 3), (-3, -3))], 1)
                    mind = torch.exp(-distances / distances.mean(1, keepdim=True).clamp_min(1e-4))
                    structure = -((mind[1:] - mind[:1]).square().mean(1) * common).sum((-2, -1)) / common.sum()
                    score = {'abs_pearson_3x3': pearson.tolist(),
                             'negative_mind8_3px_mse': structure.tolist()}
            row = {'animal_id': animal, 'specimen_id': animal, 'experiment_id': None,
                'section_id': section, 'source_thumbnail': image_name,
                'source_thumbnail_sha256': hashlib.sha256(image_bytes).hexdigest(),
                'displacement_x_sha256': hashlib.sha256(dx_bytes).hexdigest(),
                'displacement_y_sha256': hashlib.sha256(dy_bytes).hexdigest(),
                'mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
                'common_sites_96': sites, 'score': score}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        assert seen == valid
        stream.flush()
        print(json.dumps({'event': 'animal_completed', 'animal_id': animal,
            'sections': len(seen)}), flush=True)

assert len(rows) == 188 and len({(row['animal_id'], row['section_id']) for row in rows}) == 188
summary = {'sections': len(rows), 'animals': {}, 'consistency_gate_passed': False,
    'calibrated': False, 'public_benchmark_used': False}
for animal in ANIMALS:
    group = [row for row in rows if row['animal_id'] == animal]
    eligible = [row for row in group if row['score'] is not None]
    result = {'sections': len(group), 'eligible_sections': len(eligible), 'scores': {}}
    for metric in scores:
        values = np.asarray([row['score'][metric] for row in eligible])
        human_zero = values[:, 1] - values[:, 0]
        human_negative = values[:, 1] - values[:, 2]
        result['scores'][metric] = {'mean': values.mean(0).tolist(),
            'human_minus_zero_mean': float(human_zero.mean()),
            'human_minus_zero_median': float(np.median(human_zero)),
            'human_over_zero_sections': int((human_zero > 0).sum()),
            'human_minus_negative_mean': float(human_negative.mean()),
            'human_minus_negative_median': float(np.median(human_negative)),
            'human_over_negative_sections': int((human_negative > 0).sum())}
    result['consistency_passed'] = bool(len(eligible) >= .9 * len(group) and all(
        item['human_over_zero_sections'] >= .6 * len(eligible)
        and item['human_over_negative_sections'] >= .6 * len(eligible)
        and item['human_minus_zero_median'] > 0
        and item['human_minus_negative_median'] > 0
        for item in result['scores'].values()))
    summary['animals'][animal] = result
summary['consistency_gate_passed'] = all(
    result['consistency_passed'] for result in summary['animals'].values())
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
hashes = {}
for name in ('config.json', 'rows.jsonl', 'summary.json'):
    with (OUT / name).open('rb') as stream:
        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(OUT / 'completed.json').write_text(json.dumps({'sections': len(rows),
    'source_sha256': config['source_sha256'], 'protocol_sha256': config['protocol_sha256'],
    'output_sha256': hashes, 'calibrated': False,
    'public_benchmark_used': False}, indent=2), encoding='utf8')
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
