"""TRAIN-only thumbnail and unwarped VisuAlign atlas-plane preview."""

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import nrrd
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

root = Path('I:/AnatomyTracker/data/lupori_joint_185')
atlas = Path('I:/AnatomyTracker/data/Allen Brain Atlas 25um')
animals = ['AL1A', 'CC2B', 'CC4B', 'CC4A']
template = nrrd.read(str(atlas / 'average_template_25.nrrd'), index_order='F')[0]
annotation = nrrd.read(str(atlas / 'annotation_25.nrrd'), index_order='F')[0]
assert template.shape == annotation.shape == (528, 320, 456)
panel = Image.new('RGB', (2000, 1540), 'white')
draw = ImageDraw.Draw(panel)
with ZipFile(root / 'rawData.zip') as source:
    for i, animal in enumerate(animals):
        record = json.loads(source.read(f'{animal}/{animal}-visualign.json'))
        slices = record['slices']
        row = slices[len(slices) // 2]
        name = f'{animal}/{row["filename"]}'
        observed = Image.open(BytesIO(source.read(name))).convert('RGB')
        width, height = observed.size
        out_width, out_height = 480, round(height * 480 / width)
        x = np.arange(out_width, dtype=np.float32) * width / out_width / width
        y = np.arange(out_height, dtype=np.float32) * height / out_height / height
        xx, yy = np.meshgrid(x, y)
        o, u, v = np.asarray(row['anchoring'], dtype=np.float32).reshape(3, 3)
        coords = o[:, None, None] + u[:, None, None] * xx + v[:, None, None] * yy
        acquired = observed.resize((out_width, out_height))
        left, top = 10, i * 385 + 24
        panel.paste(acquired, (left, top))
        draw.text((left, top - 19), name, fill='black')
        for flip_ap in (False, True):
            ap = 528 - coords[1] if flip_ap else coords[1]
            ap_dv_ml = [ap, 320 - coords[2], coords[0]]
            image = map_coordinates(template, ap_dv_ml, order=1, mode='constant', cval=0)
            support = map_coordinates(annotation, ap_dv_ml, order=0, mode='constant', cval=0) > 0
            scale = np.percentile(image[support], [1, 99]) if support.any() else [0, 1]
            gray = np.uint8(np.clip((image - scale[0]) * 255 / max(scale[1] - scale[0], 1), 0, 255))
            atlas_image = Image.fromarray(gray).convert('RGB')
            atlas_image.putalpha(Image.fromarray(np.uint8(support) * 255))
            panel.paste(atlas_image.convert('RGB'), (left + 500 * (flip_ap + 1), top))
            draw.text((left + 500 * (flip_ap + 1), top - 19), 'source AP' if not flip_ap else 'AP reversed', fill='black')
            if flip_ap:
                overlay = acquired.copy()
                overlay.paste(atlas_image, (0, 0), atlas_image.getchannel('A').point(lambda a: a // 3))
                panel.paste(overlay, (left + 1500, top))
                draw.text((left + 1500, top - 19), 'AP reversed over acquired', fill='black')

output = root / 'train_axis_corrected_atlas_preview.png'
panel.save(output)
print(output)
