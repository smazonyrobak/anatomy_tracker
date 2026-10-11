"""Check whether 158 weak-real disagreement is a coordinate or model-domain failure."""

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
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.joint_anatomy_model_156 import JointAnatomyModel156

data = root / 'data/joint_v7_allen_fullcanvas_192_001'
records = [json.loads(line) for line in (data / 'records.jsonl').open()]
records = [record for record in records if record['training_split'] == 'development']
images = np.load(data / 'images.npy', mmap_mode='r')
with np.load(data / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()

model = JointAnatomyModel156().cuda().eval().requires_grad_(False)
checkpoint = root / 'runs/joint_feedback_158_train151_001/feedback/joint_feedback_step_00000.pt'
with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
model.load_state_dict(saved['model'], strict=True)

rows = []
with torch.inference_mode():
    for record in records:
        index = record['array_row_index']
        native = np.concatenate((np.asarray(images[index], dtype=np.float32),
                                 np.zeros((4, 192, 192), dtype=np.float32)))[None]
        image = F.interpolate(torch.from_numpy(native).cuda(), (256, 256),
                              mode='bilinear', align_corners=False)
        prediction = model.pose.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten()
        mode = int(prior.argmax()) // 2
        centre = full_frame_state_to_components(prediction['state'])[0][0, mode]
        affine = affines[index]
        reference = affine[:, 2] + 95.5 * affine[:, 0] + 95.5 * affine[:, 1]
        rows.append({'donor': record['animal_id'], 'section_id': record['section_id'],
                     'predicted_centre_um': centre.tolist(), 'weak_reference_centre_um': reference.tolist(),
                     'centre_distance_mm': float(torch.linalg.vector_norm(
                         centre - torch.from_numpy(reference).to(centre)).item() / 1000)})

predicted = np.array([row['predicted_centre_um'] for row in rows])
reference = np.array([row['weak_reference_centre_um'] for row in rows])
print(json.dumps({'sections': len(rows), 'donors': len({row['donor'] for row in rows}),
                  'predicted_centre_median_um': np.median(predicted, axis=0).tolist(),
                  'weak_reference_centre_median_um': np.median(reference, axis=0).tolist(),
                  'predicted_centre_range_um': [predicted.min(0).tolist(), predicted.max(0).tolist()],
                  'weak_reference_centre_range_um': [reference.min(0).tolist(), reference.max(0).tolist()],
                  'median_centre_distance_mm': float(np.median([row['centre_distance_mm'] for row in rows])),
                  'first_three': rows[:3]}, allow_nan=False))
