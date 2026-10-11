"""TRAIN-only blind-pose transfer check on acquired Lupori sections."""

import hashlib
import json
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
assert archive.stat().st_size == split['archive_bytes']
torch.set_num_threads(4)
checkpoints = {
    '128': root / 'runs/joint_in_path_correspondence_128_treatment/joint_step_06000.pt',
    '132': root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt',
}
records, images, targets = [], [], []
chart = np.array([[0, 0], [255 / 256, 0], [0, 255 / 256],
                  [255 / 256, 255 / 256], [127.5 / 256, 127.5 / 256]], dtype=np.float32)
with ZipFile(archive) as source:
    for animal in split['train']:
        info = ElementTree.fromstring(source.read(f'{animal}/{animal}-info.xml'))
        valid = {row.findtext('name') for row in info.findall('slices')
                 if row.findtext('valid') == '1'}
        visualign = json.loads(source.read(f'{animal}/{animal}-visualign.json'))
        for row in visualign['slices']:
            section_id = Path(row['filename']).name.removesuffix('-thumb.png')
            if section_id not in valid:
                continue
            source_name = f'{animal}/{row["filename"]}'
            image = Image.open(BytesIO(source.read(source_name))).convert('RGB')
            image = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(image[..., 0], image[..., 1]) / 255
            o, u, v = np.asarray(row['anchoring'], dtype=np.float32).reshape(3, 3)
            quicknii = o[None] + chart[:, :1] * u + chart[:, 1:] * v
            target = quicknii_to_allen_points(torch.from_numpy(quicknii)).numpy() * 25
            records.append({'animal_id': animal, 'section_id': section_id,
                            'source_thumbnail': source_name})
            images.append(gray)
            targets.append(target)

assert len(records) == 240 and len({(r['animal_id'], r['section_id']) for r in records}) == 240
images = np.stack(images)
targets = torch.from_numpy(np.stack(targets)).cuda()
chart = torch.tensor(chart, device='cuda')
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
    for appearance in ('raw_max_rgb', 'tissue_p99_max_rgb'):
        for start in range(0, len(records), 8):
            batch = images[start:start + 8].copy()
            if appearance == 'tissue_p99_max_rgb':
                scale = np.percentile(batch, 99, axis=(1, 2))[:, None, None]
                batch = np.clip(batch / np.maximum(scale, .01), 0, 1)
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
                    results.append({**records[start + j], 'checkpoint': checkpoint_id,
                        'checkpoint_sha256': checkpoint_sha256, 'appearance': appearance,
                        'selected_five_point_mm': float(errors[j, chosen[j]]),
                        'oracle_top16_five_point_mm': float(errors[j, top16[j]].min()),
                        'oracle_all160_five_point_mm': float(errors[j].min())})
    del model

output = root / 'data/lupori_joint_185/train_pose_baseline_186.json'
output.write_text(json.dumps({'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
    'label_role': 'TRAIN-only human VisuAlign weak reference; no held-out qualification',
    'rows': results}, indent=2))
for checkpoint_id in checkpoints:
    for appearance in ('raw_max_rgb', 'tissue_p99_max_rgb'):
        rows = [r for r in results if r['checkpoint'] == checkpoint_id and r['appearance'] == appearance]
        print(checkpoint_id, appearance, len(rows), {key: round(float(np.median([r[key] for r in rows])), 3)
            for key in ('selected_five_point_mm', 'oracle_top16_five_point_mm',
                        'oracle_all160_five_point_mm')})
print(output)
