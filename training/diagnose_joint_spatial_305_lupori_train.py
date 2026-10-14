"""Zero-shot 305 pose on four acquired Lupori TRAIN animals; human labels are weak."""
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.arbitrary_plane_geometry import quicknii_to_allen_points
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.global_atlas_contrast_090 import rigid_points_090

archive = root / 'data/lupori_joint_185/rawData.zip'
split_path = Path(__file__).with_name('lupori_animal_split_185.json')
split = json.loads(split_path.read_text(encoding='utf8'))
checkpoint = root / 'runs/joint_spatial_feedback_305_pose_002/pose_step_26000.pt'
out = root / 'runs/joint_spatial_feedback_305_lupori_train_001'
source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/JOINT_SPATIAL_305_LUPORI_TRAIN_DIAGNOSTIC_PROTOCOL_20261014.md'
assert source.drive.upper() == archive.drive.upper() == checkpoint.drive.upper() == out.drive.upper() == 'I:'
assert not out.exists() and archive.stat().st_size == split['archive_bytes']
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
pose_receipt = json.loads((checkpoint.parent / 'completed.json').read_text(encoding='utf8'))
assert pose_receipt['summary']['selected_step'] == 26000
assert pose_receipt['output_sha256'][checkpoint.name] == checkpoint_sha
model, _ = load_one_shot_checkpoint(checkpoint, device='cuda')
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
chart = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                      [95.5, 95.5]], device='cuda') / 192
chart_cpu = chart.cpu()
flags = torch.arange(2, device='cuda')[None, None]
config = {'source_doi': split['source_doi'],
    'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
    'checkpoint_sha256': checkpoint_sha, 'archive_bytes': archive.stat().st_size,
    'animals': split['train'], 'input': 'max(red,green), p99, four zero channels, 192 pixels',
    'label_role': 'post-inference approximate human VisuAlign reference; TRAIN only'}
rows = []
with ZipFile(archive) as source_zip, torch.inference_mode():
    for animal in split['train']:
        info_path = f'{animal}/{animal}-info.xml'
        visualign_path = f'{animal}/{animal}-visualign.json'
        valid = {row.findtext('name') for row in ElementTree.fromstring(
            source_zip.read(info_path)).findall('slices') if row.findtext('valid') == '1'}
        visualign = json.loads(source_zip.read(visualign_path))
        images, targets, records = [], [], []
        for item in visualign['slices']:
            section_id = Path(item['filename']).name.removesuffix('-thumb.png')
            if section_id not in valid:
                continue
            member = f'{animal}/{item["filename"]}'
            image_bytes = source_zip.read(member)
            rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert('RGB').resize(
                (192, 192), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            gray = np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1)
            o, u, v = torch.tensor(item['anchoring'], dtype=torch.float32).reshape(3, 3)
            site = o + chart_cpu[:, :1] * u + chart_cpu[:, 1:] * v
            targets.append(quicknii_to_allen_points(site).numpy() * 25)
            images.append(gray)
            records.append({'animal_id': animal, 'source_section_id': section_id,
                'archive_member': member,
                'source_image_sha256': hashlib.sha256(image_bytes).hexdigest()})
        assert images
        targets = torch.from_numpy(np.stack(targets)).cuda()
        for start in range(0, len(records), 8):
            batch = images[start:start + 8]
            inputs = torch.zeros(len(batch), 5, 192, 192, device='cuda')
            inputs[:, 0] = torch.from_numpy(np.stack(batch)).cuda()
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(batch), model.modes, 2)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            mapped = rigid_points_090(states, reflections, chart, (192, 192))
            errors = (mapped - targets[start:start + len(batch), None, None]).norm(
                dim=-1).mean(-1).flatten(1) / 1000
            chosen = prior.argmax(-1)
            top16 = prior.topk(16, -1).indices
            for j in range(len(batch)):
                predicted = mapped.flatten(1, 2)[j, chosen[j]]
                human = targets[start + j]
                predicted_normal = torch.linalg.cross(predicted[1] - predicted[0],
                                                       predicted[2] - predicted[0])
                human_normal = torch.linalg.cross(human[1] - human[0], human[2] - human[0])
                cosine = F.cosine_similarity(predicted_normal[None],
                    human_normal[None]).abs().clamp_max(1)
                rows.append({**records[start + j],
                    'first_five_site_gap_mm': float(errors[j, chosen[j]]),
                    'first_centre_gap_mm': float((predicted[4] - human[4]).norm() / 1000),
                    'first_unsigned_normal_gap_deg': float(torch.rad2deg(torch.acos(cosine))[0]),
                    'top16_oracle_five_site_gap_mm': float(errors[j, top16[j]].min())})
summary = {'sections': len(rows), 'animals': {}}
for animal in split['train']:
    group = [row for row in rows if row['animal_id'] == animal]
    summary['animals'][animal] = {'sections': len(group),
        'mean_first_five_site_gap_mm': float(np.mean([row['first_five_site_gap_mm'] for row in group])),
        'median_first_five_site_gap_mm': float(np.median([row['first_five_site_gap_mm'] for row in group])),
        'median_first_unsigned_normal_gap_deg': float(np.median(
            [row['first_unsigned_normal_gap_deg'] for row in group])),
        'top16_oracle_le_1p5mm': sum(row['top16_oracle_five_site_gap_mm'] <= 1.5 for row in group)}
summary['equal_animal_mean_first_five_site_gap_mm'] = float(np.mean([
    value['mean_first_five_site_gap_mm'] for value in summary['animals'].values()]))
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
(out / 'cases.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf8')
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(out / 'completed.json').write_text(json.dumps({
    'cases_sha256': hashlib.sha256((out / 'cases.jsonl').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((out / 'config.json').read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    'no_dev_cal_final_or_public_benchmark': True, 'calibrated': False}, indent=2), encoding='utf8')
print(json.dumps(summary))
