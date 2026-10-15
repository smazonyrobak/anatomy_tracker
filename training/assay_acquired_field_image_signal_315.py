"""No-update acquired-image signal check for the AL1A human residual field."""
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
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
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
PROTOCOL = SOURCE.parents[1] / 'docs/publication/ACQUIRED_FIELD_IMAGE_SIGNAL_315_PROTOCOL_20261015.md'
SPLIT = SOURCE.with_name('lupori_animal_split_185.json')
ARCHIVE = ROOT / 'data/lupori_joint_185/rawData.zip'
PARENT_ROWS = ROOT / 'reports/acquired_map_314_AL1A/rows.jsonl'
OUT = ROOT / 'reports/acquired_field_image_signal_315_AL1A'
assert SOURCE.drive.upper() == OUT.drive.upper() == 'I:' and not OUT.exists()
split = json.loads(SPLIT.read_text(encoding='utf8'))
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
assert ARCHIVE.stat().st_size == split['archive_bytes']
previous = [json.loads(line) for line in PARENT_ROWS.read_text(encoding='utf8').splitlines()]
assert len(previous) == 52 and {row['animal_id'] for row in previous} == {'AL1A'}
expected = {row['section_id']: row for row in previous}
ranked = sorted(previous, key=lambda row: (row['rms_mm']['zero'], row['section_id']))
qc = [ranked[index]['section_id'] for index in (6, 19, 32, 45)]

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
side, grid = 192, 96
axis = (torch.arange(grid, device='cuda') + .5) / grid - .5 / side
y, x = torch.meshgrid(axis, axis, indexing='ij')
offsets = torch.tensor([-50., -25., 0., 25., 50.], device='cuda')
weights = torch.tensor([1., 2., 4., 2., 1.], device='cuda')
config = {'animal_id': 'AL1A', 'role': 'TRAIN-inner diagnostic, not animal-independent',
    'source_doi': split['source_doi'], 'archive_md5': split['archive_md5'],
    'archive_bytes': split['archive_bytes'], 'split_sha256': hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'parent_314_rows_sha256': hashlib.sha256(PARENT_ROWS.read_bytes()).hexdigest(),
    'input_side': side, 'map_side': grid, 'offsets_um': offsets.tolist(),
    'weights_normalized_by_renderer': weights.tolist(),
    'fields': ['zero', 'human_affine_free', 'negative_human_affine_free', 'raw_full_human'],
    'score_mask': 'source tissue AND all four atlas supports >0.5, eroded 7x7; min 400',
    'scores': ['abs_pearson_3x3', 'negative_mind8_3px_mse'],
    'qc_section_ids': qc, 'calibrated': False, 'public_benchmark_used': False}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')

rows, qc_panels = [], {}
with ZipFile(ARCHIVE) as source_zip, (OUT / 'rows.jsonl').open('w', encoding='utf8') as stream:
    info = ElementTree.fromstring(source_zip.read('AL1A/AL1A-info.xml'))
    valid = {item.findtext('name') for item in info.findall('slices')
             if item.findtext('valid') == '1'}
    registration = json.loads(source_zip.read('AL1A/AL1A-visualign.json'))
    for metadata in registration['slices']:
        section = Path(metadata['filename']).name.removesuffix('-thumb.png')
        if section not in valid:
            continue
        image_name = f'AL1A/{metadata["filename"]}'
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
        dx_bytes = source_zip.read(f'AL1A/dispFields/{section}-dispFieldX.csv')
        dy_bytes = source_zip.read(f'AL1A/dispFields/{section}-dispFieldY.csv')
        mask_bytes = source_zip.read(f'AL1A/masks/{section}-mask.png')
        dx = np.loadtxt(BytesIO(dx_bytes), delimiter=',')
        dy = np.loadtxt(BytesIO(dy_bytes), delimiter=',')
        query = [(y * height * .2 - 1).cpu().numpy(), (x * width * .2 - 1).cpu().numpy()]
        sampled_x = torch.from_numpy(map_coordinates(dx, query, order=1, mode='nearest')).cuda().float()
        sampled_y = torch.from_numpy(map_coordinates(dy, query, order=1, mode='nearest')).cuda().float()
        displacement = sampled_x[..., None] * ouv[1] / width + sampled_y[..., None] * ouv[2] / height
        local = torch.einsum('ij,hwj->ihw', frame.T, displacement)
        human = project_surface_affine_out(local[None])[0][0]
        mask = torch.from_numpy(np.asarray(Image.open(BytesIO(mask_bytes)).convert('L').resize(
            (grid, grid), Image.Resampling.NEAREST)) > 127).cuda()

        with torch.inference_mode():
            fields = torch.stack((torch.zeros_like(human), human, -human, local))
            plane = ouv[0] + x[..., None] * ouv[1] + y[..., None] * ouv[2]
            surface = plane[None] + torch.einsum('ij,bjhw->bhwi', frame, fields)
            world = surface[:, None] + offsets[None, :, None, None, None] * frame[:, 2]
            rendered = render_finite_thickness_coordinate_grid(atlas, world,
                (0., 0., 0.), (25., 25., 25.), weights.expand(4, -1))
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
                count = common.sum()
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
                structure = -((mind[1:] - mind[:1]).square().mean(1) * common).sum((-2, -1)) / count
                score = {'abs_pearson_3x3': pearson.tolist(),
                         'negative_mind8_3px_mse': structure.tolist()}
            if section in qc:
                qc_panels[section] = (observed[0, 0].cpu().numpy() if sites >= 400 else
                    F.interpolate(inputs[:, :1], (grid, grid), mode='area')[0, 0].cpu().numpy(),
                    atlas_images[:, 0].cpu().numpy(), support.cpu().numpy(), score)

        original = expected[section]
        assert hashlib.sha256(image_bytes).hexdigest() == original['source_thumbnail_sha256']
        assert hashlib.sha256(dx_bytes).hexdigest() == original['displacement_x_sha256']
        assert hashlib.sha256(dy_bytes).hexdigest() == original['displacement_y_sha256']
        assert hashlib.sha256(mask_bytes).hexdigest() == original['mask_sha256']
        row = {'animal_id': 'AL1A', 'specimen_id': 'AL1A', 'experiment_id': None,
               'section_id': section, 'source_thumbnail': image_name,
               'source_thumbnail_sha256': original['source_thumbnail_sha256'],
               'displacement_x_sha256': original['displacement_x_sha256'],
               'displacement_y_sha256': original['displacement_y_sha256'],
               'mask_sha256': original['mask_sha256'], 'common_sites_96': sites,
               'score': score}
        rows.append(row)
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        if len(rows) % 26 == 0:
            stream.flush()
            print(json.dumps({'event': 'animal_milestone', 'sections': len(rows)}), flush=True)

assert len(rows) == 52 and set(expected) == {row['section_id'] for row in rows}
eligible = [row for row in rows if row['score'] is not None]
summary = {'sections': len(rows), 'eligible_sections': len(eligible), 'scores': {},
           'descriptive_gate_passed': False}
for metric in config['scores']:
    values = np.asarray([row['score'][metric] for row in eligible])
    human_zero = values[:, 1] - values[:, 0]
    human_negative = values[:, 1] - values[:, 2]
    full_zero = values[:, 3] - values[:, 0]
    summary['scores'][metric] = {'mean': values.mean(0).tolist(),
        'human_minus_zero_mean': float(human_zero.mean()),
        'human_minus_zero_median': float(np.median(human_zero)),
        'human_over_zero_sections': int((human_zero > 0).sum()),
        'human_minus_negative_mean': float(human_negative.mean()),
        'human_minus_negative_median': float(np.median(human_negative)),
        'human_over_negative_sections': int((human_negative > 0).sum()),
        'raw_full_minus_zero_mean': float(full_zero.mean()),
        'raw_full_minus_zero_median': float(np.median(full_zero)),
        'raw_full_over_zero_sections': int((full_zero > 0).sum())}
summary['descriptive_gate_passed'] = bool(len(eligible) >= 45 and all(
    item['human_over_zero_sections'] >= 35 and item['human_over_negative_sections'] >= 35
    and item['human_minus_zero_median'] > 0 and item['human_minus_negative_median'] > 0
    for item in summary['scores'].values()))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')

figure, axes = plt.subplots(4, 5, figsize=(12, 10))
titles = ['Acquired', 'Atlas: zero', 'Atlas: local', 'Atlas: negative', 'Atlas: full field']
for row_index, section in enumerate(qc):
    source, targets, supports, scores = qc_panels[section]
    for column, ax in enumerate(axes[row_index]):
        view = source if column == 0 else targets[column - 1]
        if column:
            view = np.where(supports[column - 1] > .5, view, 0)
        ax.imshow(np.clip(view, 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_index == 0:
            ax.set_title(titles[column])
        if column == 0:
            ax.set_ylabel(section, fontsize=8)
figure.tight_layout()
figure.savefig(OUT / 'four_section_qc.png', dpi=150)
plt.close(figure)

hashes = {}
for name in ('config.json', 'rows.jsonl', 'summary.json', 'four_section_qc.png'):
    with (OUT / name).open('rb') as stream:
        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(OUT / 'completed.json').write_text(json.dumps({'sections': len(rows),
    'script_sha256': config['script_sha256'], 'protocol_sha256': config['protocol_sha256'],
    'output_sha256': hashes, 'calibrated': False, 'public_benchmark_used': False},
    indent=2), encoding='utf8')
print(json.dumps({'event': 'completed', 'summary': summary}), flush=True)
