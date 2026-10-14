"""Read-only 309 before/after inference on all four acquired Lupori TRAIN animals."""
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
checkpoints = {step: root / f'runs/first_choice_exposure_309_001/single/pose_local_{step:05d}.pt'
               for step in (0, 20000)}
out = root / 'runs/first_choice_309_lupori_train_audit_001'
assert all(path.drive.upper() == 'I:' for path in (Path(__file__).resolve(), archive, out,
                                                    *checkpoints.values()))
assert not out.exists() and archive.stat().st_size == split['archive_bytes']
assert split['train'] == ['AL1A', 'CC2B', 'CC4B', 'CC4A']

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
chart = torch.tensor([[0., 0.], [191., 0.], [0., 191.], [191., 191.],
                      [95.5, 95.5]], device='cuda') / 192
chart_cpu = chart.cpu()
images, targets, records = [], [], []
with ZipFile(archive) as source_zip:
    for animal in split['train']:
        valid = {row.findtext('name') for row in ElementTree.fromstring(
            source_zip.read(f'{animal}/{animal}-info.xml')).findall('slices')
            if row.findtext('valid') == '1'}
        visualign = json.loads(source_zip.read(f'{animal}/{animal}-visualign.json'))
        for item in visualign['slices']:
            section_id = Path(item['filename']).name.removesuffix('-thumb.png')
            if section_id not in valid:
                continue
            member = f'{animal}/{item["filename"]}'
            raw = source_zip.read(member)
            rgb = np.asarray(Image.open(BytesIO(raw)).convert('RGB').resize(
                (192, 192), Image.Resampling.BILINEAR), dtype=np.float32)
            gray = np.maximum(rgb[..., 0], rgb[..., 1]) / 255
            images.append(np.clip(gray / max(np.percentile(gray, 99), .01), 0, 1))
            o, u, v = torch.tensor(item['anchoring'], dtype=torch.float32).reshape(3, 3)
            points = o + chart_cpu[:, :1] * u + chart_cpu[:, 1:] * v
            targets.append(quicknii_to_allen_points(points).numpy() * 25)
            records.append({'animal_id': animal, 'source_section_id': section_id,
                            'archive_member': member,
                            'source_image_sha256': hashlib.sha256(raw).hexdigest()})
images = torch.from_numpy(np.stack(images))
targets = torch.from_numpy(np.stack(targets)).cuda()
assert len(records) == 240 and len({(r['animal_id'], r['source_section_id']) for r in records}) == 240

rows = []
for step, checkpoint in checkpoints.items():
    model, _ = load_one_shot_checkpoint(checkpoint, device='cuda')
    flags = torch.arange(2, device='cuda')[None, None]
    with torch.inference_mode():
        for start in range(0, len(records), 8):
            gray = images[start:start + 8].cuda()
            inputs = torch.zeros(len(gray), 5, 192, 192, device='cuda')
            inputs[:, 0] = gray
            prediction = model.predict(inputs)
            states = prediction['state'][:, :, None].expand(-1, -1, 2, -1)
            reflections = flags.expand(len(gray), model.modes, 2)
            mass = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            mapped = rigid_points_090(states, reflections, chart, (192, 192))
            errors = (mapped - targets[start:start + len(gray), None, None]).norm(
                dim=-1).mean(-1).flatten(1) / 1000
            first = mass.argmax(-1)
            top16 = mass.topk(16, -1).indices
            for j in range(len(gray)):
                predicted = mapped.flatten(1, 2)[j, first[j]]
                human = targets[start + j]
                a = torch.linalg.cross(predicted[1] - predicted[0],
                                       predicted[2] - predicted[0])
                b = torch.linalg.cross(human[1] - human[0], human[2] - human[0])
                cosine = F.cosine_similarity(a[None], b[None]).abs().clamp_max(1)
                rows.append({'step': step, **records[start + j],
                             'first_five_site_gap_mm': float(errors[j, first[j]]),
                             'first_centre_gap_mm': float((predicted[4] - human[4]).norm() / 1000),
                             'first_unsigned_normal_gap_deg': float(torch.rad2deg(
                                 torch.acos(cosine))[0]),
                             'top16_oracle_five_site_gap_mm': float(errors[j, top16[j]].min())})
    del model
    torch.cuda.empty_cache()

summary = {}
for step in checkpoints:
    summary[str(step)] = {}
    for animal in split['train']:
        group = [row for row in rows if row['step'] == step and row['animal_id'] == animal]
        summary[str(step)][animal] = {'sections': len(group),
            'mean_first_five_site_gap_mm': float(np.mean([r['first_five_site_gap_mm'] for r in group])),
            'median_first_unsigned_normal_gap_deg': float(np.median([
                r['first_unsigned_normal_gap_deg'] for r in group])),
            'top16_oracle_le1p5mm': sum(r['top16_oracle_five_site_gap_mm'] <= 1.5 for r in group)}
config = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
          'archive_bytes': archive.stat().st_size, 'source_doi': split['source_doi'],
          'checkpoint_sha256': {str(step): hashlib.sha256(path.read_bytes()).hexdigest()
                                for step, path in checkpoints.items()},
          'label_role': 'approximate human VisuAlign reference read after image-only inference',
          'input': 'max(red,green), p99, four zero channels, 192 pixels',
          'data_role': 'four TRAIN animals, three gradient animals and one TRAIN-inner guard; not independent validation'}
out.mkdir(parents=True)
(out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
(out / 'rows.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf8')
(out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
(out / 'completed.json').write_text(json.dumps({name: hashlib.sha256(
    (out / name).read_bytes()).hexdigest() for name in ('config.json', 'rows.jsonl',
                                                       'summary.json')}, indent=2), encoding='utf8')
print(json.dumps(summary), flush=True)
