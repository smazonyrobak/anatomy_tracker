"""Find source-CUTLAS to bound-NRRD axis directions using TRAIN tissue masks."""

import json
from io import BytesIO
from itertools import product
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import nrrd
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

root = Path('I:/AnatomyTracker/data/lupori_joint_185')
atlas = Path('I:/AnatomyTracker/data/Allen Brain Atlas 25um/annotation_25.nrrd')
annotation = nrrd.read(str(atlas), index_order='F')[0]
assert annotation.shape == (528, 320, 456)
animals = ['AL1A', 'CC2B', 'CC4B', 'CC4A']
directions = list(product((False, True), repeat=3))
results = {}
with ZipFile(root / 'rawData.zip') as source:
    for animal in animals:
        record = json.loads(source.read(f'{animal}/{animal}-visualign.json'))
        info = ElementTree.fromstring(source.read(f'{animal}/{animal}-info.xml'))
        valid = {node.findtext('name') for node in info.findall('slices') if node.findtext('valid') == '1'}
        scores = []
        for row in record['slices']:
            if Path(row['filename']).name.removesuffix('-thumb.png') not in valid:
                continue
            thumb = f'{animal}/{row["filename"]}'
            mask_name = thumb.replace('/thumbnails/', '/masks/').replace('-thumb.png', '-mask.png')
            mask = Image.open(BytesIO(source.read(mask_name))).convert('L')
            width, height = mask.size
            width_small, height_small = 256, round(height * 256 / width)
            observed = np.asarray(mask.resize((width_small, height_small), Image.Resampling.NEAREST)) > 127
            xx, yy = np.meshgrid(np.arange(width_small) / width_small, np.arange(height_small) / height_small)
            o, u, v = np.asarray(row['anchoring']).reshape(3, 3)
            q = o[:, None, None] + u[:, None, None] * xx + v[:, None, None] * yy
            dice_scores = []
            for flip_x, flip_y, flip_z in directions:
                ml = 456 - q[0] if flip_x else q[0]
                ap = 528 - q[1] if flip_y else q[1]
                dv = 320 - q[2] if flip_z else q[2]
                predicted = map_coordinates(annotation, [ap, dv, ml], order=0, mode='constant', cval=0) > 0
                dice = 2 * np.count_nonzero(observed & predicted) / (observed.sum() + predicted.sum())
                dice_scores.append(round(float(dice), 4))
            scores.append({'source_filename': row['filename'], 'dice_by_direction': dice_scores})
        results[animal] = scores
        print(animal, len(scores), [(direction, round(float(np.median([row['dice_by_direction'][i] for row in scores])), 4)) for i, direction in enumerate(directions)])

output = root / 'train_axis_qc.json'
output.write_text(json.dumps({'directions_are_flip_ml_ap_dv': directions, 'animals': results}, indent=2), encoding='utf-8')
print(output)
