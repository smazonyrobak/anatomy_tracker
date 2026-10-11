"""One reserved TRAIN animal, exact fluorescence input, parent and frozen milestones."""

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.arbitrary_plane_geometry import quicknii_to_allen_points
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher

root = Path('I:/AnatomyTracker')
archive = root / 'data/lupori_joint_185/rawData.zip'
split_path = Path(__file__).with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text())
development = os.environ.get('ANATOMY_LUPORI_DEV_EVAL') == '1'
run = root / 'runs' / ('lupori_fluorescence_pose_188_all_train' if development
                       else 'lupori_fluorescence_pose_187_inner_AL1A')
parent = root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt'
checkpoints = {'132_parent': parent}
finished = (run / 'completed.json').exists()
if finished:
    checkpoints.update({f'187_step_{step:04d}': run / f'joint_step_{step:05d}.pt'
                        for step in (0, 400, 800, 1200)})

assert archive.stat().st_size == split['archive_bytes']
torch.set_num_threads(4)
animal = split['development'][0] if development else 'AL1A'
assert animal == ('CC1B' if development else 'AL1A')
chart = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                      [127.5, 127.5]], device='cuda') / 256
records, images, targets = [], [], []
with ZipFile(archive) as source:
    info = ElementTree.fromstring(source.read(f'{animal}/{animal}-info.xml'))
    valid = {row.findtext('name') for row in info.findall('slices')
             if row.findtext('valid') == '1'}
    visualign = json.loads(source.read(f'{animal}/{animal}-visualign.json'))
    for row in visualign['slices']:
        section_id = Path(row['filename']).name.removesuffix('-thumb.png')
        if section_id not in valid:
            continue
        image = Image.open(BytesIO(source.read(f'{animal}/{row["filename"]}'))).convert('RGB')
        rgb = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR),
                         dtype=np.float32)
        gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
        gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
        o, u, v = torch.tensor(row['anchoring'], dtype=torch.float32).reshape(3, 3)
        quicknii = o + chart.cpu()[:, :1] * u + chart.cpu()[:, 1:] * v
        targets.append(quicknii_to_allen_points(quicknii).numpy() * 25)
        images.append(gray)
        records.append({'animal_id': animal, 'source_section_id': section_id})

assert len(records) > 0 and (development or len(records) == 52)
targets = torch.from_numpy(np.stack(targets)).cuda()
images = np.stack(images)
flags = torch.arange(2, device='cuda')[None, None]
results = []
for checkpoint_id, path in checkpoints.items():
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
        atlas_conditioning=True, fit_quality=True, vector_refinement=True,
        candidate_ranking=True, fitted_ranking=True).cuda().eval()
    attach_global_plane_matcher(model, enabled=True)
    model.load_state_dict(saved['model'], strict=True)
    checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    del saved
    for start in range(0, len(records), 8):
        batch = images[start:start + 8]
        inputs = torch.zeros(len(batch), 5, 256, 256, device='cuda')
        inputs[:, 0] = torch.from_numpy(batch).cuda()
        with torch.inference_mode():
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(batch), model.modes, 2)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            mapped = rigid_points_090(states, reflections, chart)
            errors = (mapped - targets[start:start + len(batch), None, None]
                     ).norm(dim=-1).mean(-1).flatten(1) / 1000
            chosen = prior.argmax(-1)
            top16 = prior.topk(16, -1).indices
            for j in range(len(batch)):
                mapped_selected = mapped.flatten(1, 2)[j, chosen[j]]
                target_selected = targets[start + j]
                mapped_normal = torch.linalg.cross(
                    mapped_selected[1] - mapped_selected[0],
                    mapped_selected[2] - mapped_selected[0])
                target_normal = torch.linalg.cross(
                    target_selected[1] - target_selected[0],
                    target_selected[2] - target_selected[0])
                cosine = F.cosine_similarity(mapped_normal[None],
                    target_normal[None]).abs().clamp_max(1)
                results.append({**records[start + j], 'checkpoint': checkpoint_id,
                    'checkpoint_sha256': checkpoint_sha256,
                    'selected_five_point_mm': float(errors[j, chosen[j]]),
                    'selected_center_mm': float((mapped_selected[4] - target_selected[4]).norm() / 1000),
                    'selected_plane_normal_angle_deg': float(torch.rad2deg(torch.acos(cosine))[0]),
                    'oracle_top16_five_point_mm': float(errors[j, top16[j]].min()),
                    'oracle_all160_five_point_mm': float(errors[j].min())})
    del model

output = root / 'data/lupori_joint_185' / (
    'development_pose_188.json' if development else
    'inner_guard_candidate_187.json' if finished else 'inner_guard_parent_187.json')
output.write_text(json.dumps({'source_doi': split['source_doi'],
    'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
    'archive_md5': split['archive_md5'],
    'input': 'whole RGB thumbnail resized 256, max(red,green)/255, image-wide p99 normalized, other channels zero',
    'label_role': ('official DEV animal human VisuAlign weak reference' if development
                   else 'reserved TRAIN animal human VisuAlign weak reference'),
    'rows': results}, indent=2))
for checkpoint_id in checkpoints:
    rows = [row for row in results if row['checkpoint'] == checkpoint_id]
    print(checkpoint_id, len(rows),
          'selected_median_mm', float(np.median([row['selected_five_point_mm'] for row in rows])),
          'all160_le_1p5_mm', sum(row['oracle_all160_five_point_mm'] <= 1.5 for row in rows),
          'top16_median_mm', float(np.median([row['oracle_top16_five_point_mm'] for row in rows])))
print(output)
