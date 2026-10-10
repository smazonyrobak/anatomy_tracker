"""Four frozen v4 synthetic sections: true atlas, observed input, blind top-one atlas."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_to_components, render_finite_thickness_coordinate_grid,
)
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


panel = root / 'data/fresh_v4_pose_dev_panel_132'
run = root / 'runs/v4_pose_adaptation_132'
evaluated = root / 'runs/v4_pose_adaptation_132_dev_eval'
output = root / 'reports/actual_v4_pose_examples_147'
checkpoint = run / 'joint_step_02000.pt'
records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
frozen = {row['section_id']: row for row in
          (json.loads(line) for line in (evaluated / 'synthetic_rows.jsonl').open())
          if row['arm'] == '2000' and row['cohort'] == 'v4'}
selection = []
for low, high in ((0, 15), (15, 30), (30, 45), (45, 56)):
    selection.append(next(record for record in records if record['eligible'] and
        low <= frozen[record['section_id']]['nearest_cardinal_angle_deg'] < high and
        record['synthetic_subject_plan_id'] not in
        {used['synthetic_subject_plan_id'] for used in selection}))

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(torch.load(checkpoint, map_location='cpu',
                                 weights_only=True)['model'], strict=True)
axis = torch.arange(256, device='cuda') / 256
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1).reshape(-1, 2)
sheet = Image.new('RGB', (3 * 256 + 4 * 12, 4 * (256 + 52) + 12), (235, 235, 235))
draw = ImageDraw.Draw(sheet)
shown = []

with torch.inference_mode():
    for index, record in enumerate(selection):
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            truth = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
            truth_reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        top = int(prior.argmax())
        assert top == frozen[record['section_id']]['top1_branch_id']
        atlas_images = []
        for state, reflected in ((truth, truth_reflection),
                                 (prediction['state'][:, top // 2],
                                  torch.tensor([top % 2], device='cuda'))):
            plane = rigid_points_090(state, reflected, chart).reshape(1, 256, 256, 3)
            normal = full_frame_state_to_components(state)[1][..., :, 2]
            world = plane[:, None] + offsets[:, :, None, None, None] * normal[:, None, None, None]
            rendered = render_finite_thickness_coordinate_grid(
                atlas, world, (0., 0., 0.), (25., 25., 25.), weights)
            intensity = rendered[:, 0] / rendered[:, 1].clamp_min(1e-4)
            atlas_images.append(torch.where(rendered[:, 1] > .05, intensity, 0)
                                .clamp(0, 1)[0].cpu().numpy())
        observed = image[0, 0].clamp(0, 1).cpu().numpy()
        angle = frozen[record['section_id']]['nearest_cardinal_angle_deg']
        events = record['provenance']['one_shot_slide_artifacts_v3']['events']
        active = ', '.join(name for name, enabled in events.items() if enabled) or 'none'
        exposure = record['provenance']['one_shot_slide_artifacts_v4']['exposure']
        y = 12 + index * (256 + 52)
        draw.text((12, y), f'{angle:.1f} deg off closest cardinal | {record["appearance_mode"]} | '
                  f'artifacts: {active} | exposure {exposure:.2f}', fill=(0, 0, 0))
        for column, (label, array) in enumerate((('atlas at true plane', atlas_images[0]),
                                                 ('observed synthetic input', observed),
                                                 ('atlas at model top-one', atlas_images[1]))):
            x = 12 + column * (256 + 12)
            draw.text((x, y + 17), label, fill=(0, 0, 0))
            tile = Image.fromarray(np.uint8(np.rint(array * 255)), 'L').convert('RGB')
            sheet.paste(tile, (x, y + 52))
        shown.append({'section_id': record['section_id'], 'panel_file_sha256': record['sha256'],
            'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
            'nearest_cardinal_angle_deg': angle, 'appearance_mode': record['appearance_mode'],
            'events': events, 'exposure': exposure, 'top1_branch_id': top,
            'top1_rigid_tissue_error_mm': frozen[record['section_id']]['top1_rigid_mm']})

output.mkdir(parents=True, exist_ok=False)
path = output / 'grid.png'
sheet.save(path)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt = {'description': 'actual frozen v4 generator images; atlas at true global plane and scratch-trained 132 top-one plane',
    'not_training_or_animal_validation': True, 'selection': 'first eligible section in each nearest-cardinal-angle bin, requiring distinct synthetic deformation plans; no error-based selection',
    'atlas_pair': 'finite-thickness Allen atlas with each section PSF, unwarped; not the virtual-subject clean image',
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'checkpoint_sha256': sha(checkpoint), 'evaluator_completed_sha256': sha(evaluated / 'completed.json'),
    'source_sha256': sha(__file__), 'grid_sha256': sha(path), 'cases': shown}
(output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({'grid': str(path), 'cases': len(shown),
                  'top1_rigid_error_mm': [case['top1_rigid_tissue_error_mm'] for case in shown]}))
