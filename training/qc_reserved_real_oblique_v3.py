"""TRAIN-only virtual-oblique v3 examples and frozen 111 pose predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.arbitrary_plane_reserved_real_oblique_v3 import sample_reserved_real_oblique_v3
from training.arbitrary_plane_reserved_real_stream_v8 import load_reserved_real_train
from training.global_atlas_contrast_090 import rigid_points_090


root = Path('I:/AnatomyTracker')
out = root / 'reports/real_oblique_v3_qc_20261009_001'
checkpoint_path = root / 'runs/v3_mixed_real_pose_coronal_risk_111/joint_step_01959.pt'
torch.set_num_threads(4)
context = load_reserved_real_train()
pool, attempts = [], []
with torch.inference_mode():
    for index in range(24):
        sample = sample_reserved_real_oblique_v3(context, index,
            202610090001 + index, side=256, device='cuda', return_source=True)
        record = sample['provenance'][0]
        eligible = bool(sample['eligible'][0])
        attempts.append({'donor_index': index, 'seed': 202610090001 + index,
            'virtual_section_id': record['section_id'], 'mode': record['mode'],
            'eligible': eligible,
            'valid_fraction': float(sample['valid_mask'][0].float().mean())})
        if eligible:
            chart = torch.tensor([[0., 0.], [255 / 256, 0.],
                [0., 255 / 256], [255 / 256, 255 / 256],
                [127 / 256, 127 / 256]], device='cuda')
            analytic = rigid_points_090(sample['source_state'], sample['reflection'],
                chart, source_shape=(256, 256))
            indices = torch.tensor([0, 255, 256 * 255, 256 * 256 - 1,
                256 * 127 + 127], device='cuda')
            source_points = sample['source_centre'].reshape(1, -1, 3)[:, indices]
            source_error_um = float((analytic - source_points).norm(dim=-1).max())
            mapped = F.grid_sample(sample['source_centre'].permute(0, 3, 1, 2),
                sample['observed_to_source_grid'], align_corners=True
                ).permute(0, 2, 3, 1)
            valid = sample['valid_mask']
            dense_error_um = float((mapped - sample['centre'])[valid].norm(dim=-1).max())
            assert torch.equal(sample['state'], sample['source_state'])
            assert source_error_um < .1 and dense_error_um < .1, (
                source_error_um, dense_error_um)
            pool.append({'sample': {key: value.cpu() if isinstance(value, torch.Tensor)
                else value for key, value in sample.items()},
                'source_error_um': source_error_um,
                'dense_error_um': dense_error_um})

selected = []
for mode in ('raw', 'exact_black', 'imperfect_brush'):
    match = next((i for i, row in enumerate(pool)
        if row['sample']['provenance'][0]['mode'] == mode and i not in selected), None)
    if match is not None:
        selected.append(match)
selected += [i for i in range(len(pool)) if i not in selected]
selected = selected[:6]
atlas_array, annotation = allen._decode_and_preprocess_allen_v6()
del annotation
atlas = torch.from_numpy(atlas_array).cuda()
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
grid = Image.new('RGB', (2 * 4 * 192 + 3 * 12, 3 * 238 + 4 * 12),
    (236, 236, 236))
draw = ImageDraw.Draw(grid)
rows = []
with torch.inference_mode():
    axis = torch.arange(192, device='cuda') / 192
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    chart = torch.stack((xx, yy), -1).reshape(-1, 2)
    five = torch.tensor([[0., 0.], [255 / 256, 0.],
        [0., 255 / 256], [255 / 256, 255 / 256],
        [127.5 / 256, 127.5 / 256]], device='cuda')
    for position, pool_index in enumerate(selected):
        entry = pool[pool_index]
        sample = entry['sample']
        image = sample['inputs'].cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = int(prior[0].argmax())
        state = torch.cat((sample['state'].cuda(),
            prediction['state'][:, branch // 2]), 0)
        reflection = torch.tensor([int(sample['reflection'][0]), branch % 2],
            device='cuda')
        plane = rigid_points_090(state, reflection, chart,
            source_shape=(256, 256)).reshape(2, 192, 192, 3)
        normal = full_frame_state_to_components(state)[1][..., :, 2]
        world = plane[:, None] + sample['offsets'].cuda().expand(2, -1)[:, :, None,
            None, None] * normal[:, None, None, None]
        rendered = render_finite_thickness_coordinate_grid(atlas, world,
            (0., 0., 0.), (25., 25., 25.), sample['weights'].cuda().expand(2, -1))
        atlas_image = (rendered[:, 0] / rendered[:, 1].clamp_min(1e-4)).clamp(0, 1)
        atlas_image = torch.where(rendered[:, 1] > .05, atlas_image, 0)
        five_points = rigid_points_090(state, reflection, five,
            source_shape=(256, 256))
        weak_error_mm = float((five_points[0] - five_points[1]).norm(dim=-1).mean() / 1000)
        source = F.interpolate(sample['source_image'].cuda()[:, None], (192, 192),
            mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        observed = F.interpolate(image[:, :1], (192, 192),
            mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        images = [atlas_image[0].cpu().numpy(), source, observed,
            atlas_image[1].cpu().numpy()]
        labels = ('atlas at weak pose', 'donor virtual cut',
            'after artifacts', '111 guessed atlas')
        x0 = 12 + (position % 2) * (4 * 192 + 12)
        y0 = 12 + (position // 2) * 238
        for column, (array, label) in enumerate(zip(images, labels)):
            x = x0 + column * 192
            draw.text((x + 3, y0 + 3), label, fill=(15, 15, 15))
            grey = Image.fromarray(np.uint8(np.rint(np.clip(array, 0, 1) * 255)), 'L')
            grid.paste(grey.convert('RGB'), (x, y0 + 42))
        record = sample['provenance'][0]
        draw.text((x0 + 3, y0 + 23),
            f"{position + 1} donor {record['animal_id']} {record['mode']}"
            f" | weak pose gap {weak_error_mm:.2f} mm", fill=(15, 15, 15))
        rows.append({'donor_index': record['real_oblique_v3']['donor_index'],
            'seed': record['real_oblique_v3']['seed'],
            'animal_id': record['animal_id'], 'specimen_id': record['specimen_id'],
            'experiment_id': record['experiment_id'],
            'virtual_section_id': record['section_id'], 'mode': record['mode'],
            'valid_fraction': float(sample['valid_mask'][0].float().mean()),
            'analytic_pose_error_um': entry['source_error_um'],
            'dense_map_error_um': entry['dense_error_um'],
            'prior_selected_branch': branch, 'weak_pose_gap_mm': weak_error_mm,
            'provenance': record})

out.mkdir(parents=True, exist_ok=False)
grid_path = out / 'atlas_virtual_artifact_111_grid.png'
grid.save(grid_path)
with checkpoint_path.open('rb') as stream:
    checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
receipt = {'role': 'TRAIN-only illustrative QC, no development or test images',
    'atlas_pose_role': 'weak Allen affine of donor serial stack',
    'virtual_cut_role': 'resliced acquired coronal series, not acquired oblique slide',
    'guessed_plane_role': 'frozen 111 direct pose, uncalibrated and not trained on these virtual cuts',
    'selection': 'first eligible example per appearance mode, then earliest eligible attempts',
    'attempts': attempts, 'rows': rows,
    'source_manifest_sha256': context['manifest_sha256'],
    'checkpoint_sha256': checkpoint_sha256,
    'sampler_sha256': hashlib.sha256(Path(sample_reserved_real_oblique_v3.__code__.co_filename
        ).read_bytes()).hexdigest(),
    'grid_sha256': hashlib.sha256(grid_path.read_bytes()).hexdigest()}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({'grid': str(grid_path), 'attempted': len(attempts),
    'eligible': len(pool), 'selected': len(rows),
    'mode_counts': {mode: sum(row['mode'] == mode for row in attempts)
        for mode in ('raw', 'exact_black', 'imperfect_brush')},
    'weak_pose_gap_mm': [row['weak_pose_gap_mm'] for row in rows]}))
