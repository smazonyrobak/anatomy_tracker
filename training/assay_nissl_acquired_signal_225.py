"""TRAIN-only paired atlas-intensity signal on fixed acquired weak-affine planes."""

import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['XDG_CACHE_HOME'] = str(root / 'cache')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.allen_nissl_appearance_224 import bind_nissl_appearance_224
from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state, full_frame_state_from_components
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.global_atlas_contrast_090 import render_atlas_planes_090


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score(x, y, mask):
    a, b = x[mask], y[mask]
    if a.numel() < 2:
        return None
    a, b = a - a.mean(), b - b.mean()
    return float(((a * b).mean() / (a.square().mean() * b.square().mean()).sqrt()
                  .clamp_min(1e-8)).abs())


torch.set_num_threads(4)
source = Path(__file__).resolve()
protocol = source.parents[1] / 'docs/publication/ALLEN_NISSL_ACQUIRED_SIGNAL_225_PROTOCOL_20261011.md'
out = root / 'runs/allen_nissl_acquired_signal_225_train_001_retry1'
assert source.drive.upper() == out.drive.upper() == 'I:' and not out.exists()

sources = [
    ('coronal_train', root / 'data/allen_real_training_inputs_20260929',
     'raw_model_input.npy', 'image_geometry.jsonl', 'animal_id', 'model_pixel_to_ap_dv_ml_um', 96),
    ('sagittal_train', root / 'data/allen_sagittal_ish_expansion_002_train_inputs_20261008',
     'model_input.npy', 'geometry.jsonl', 'donor_id', 'model_pixel_to_ccf_ref9_ap_dv_ml_um', 256),
]
cases, source_hashes = [], {}
for cohort, directory, image_name, geometry_name, donor_key, affine_key, side in sources:
    summary = json.loads((directory / 'summary.json').read_text())
    source_hashes[cohort] = {name: sha(directory / name) for name in
                             (image_name, geometry_name, 'summary.json')}
    assert all(source_hashes[cohort][name] == summary['output_sha256'][name]
               for name in (image_name, geometry_name))
    rows = [json.loads(line) for line in (directory / geometry_name).read_text().splitlines()]
    by_donor = {}
    for row in rows:
        assert row['split'] == ('development_train' if cohort == 'coronal_train' else 'train')
        by_donor.setdefault(row[donor_key], []).append(row)
    assert len(by_donor) == (58 if cohort == 'coronal_train' else 40)
    images = np.load(directory / image_name, mmap_mode='r')
    assert images.shape == (len(rows), 1, side, side)
    for donor, records in sorted(by_donor.items()):
        row = sorted(records, key=lambda r: (r['section_number'], r['section_id']))[(len(records)-1)//2]
        affine = np.asarray(row[affine_key], dtype=np.float32)
        cases.append({'cohort': cohort, 'donor_id': str(donor), 'section_id': row['section_id'],
            'image': np.asarray(images[row['array_row_index']]).copy(), 'side': side,
            'ouv_um': np.stack((affine[:, 2], side * affine[:, 0], side * affine[:, 1])),
            'thickness_um': float(row['section_thickness_um'] if cohort == 'coronal_train' else 25.),
            'image_sha256': row['actual_image_sha256' if cohort == 'coronal_train' else 'image_sha256']})
assert len(cases) == 98

atlas_array, _ = allen._decode_and_preprocess_allen_v6()
context = bind_nissl_appearance_224({'atlas': torch.from_numpy(atlas_array).to('cuda')})
atlases = {'average_template': context['atlas'], 'nissl': context['nissl_atlas_224']}
updates = torch.zeros((8, 9), device='cuda')
updates[:4, 5] = torch.tensor([1500., -1500., 3000., -3000.], device='cuda')
updates[4:6, 0] = torch.tensor([np.pi / 12, -np.pi / 12], device='cuda')
updates[6:8, 1] = torch.tensor([np.pi / 12, -np.pi / 12], device='cuda')
names = ('normal_plus_1.5_mm', 'normal_minus_1.5_mm', 'normal_plus_3_mm',
         'normal_minus_3_mm', 'u_plus_15_deg', 'u_minus_15_deg',
         'v_plus_15_deg', 'v_minus_15_deg')

out.mkdir(parents=True, exist_ok=False)
config = {'protocol_sha256': sha(protocol), 'source_sha256': sha(source),
    'source_hashes': source_hashes, 'nissl_binding': context['nissl_binding_224'],
    'average_atlas_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'annotation_sha256': allen.ANNOTATION_RAW_SHA256_V6,
    'weak_affine_reference': True, 'candidate_bank': names,
    'support_rule': {'min_common_cells': 64, 'min_iou': .45, 'max_area_fraction_gap': .15},
    'score': 'absolute Pearson of 3x3-smoothed 96-pixel images on common 5x5-eroded atlas support',
    'split': 'TRAIN only', 'trained_weights': False, 'calibrated': False,
    'dev_cal_final_or_public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

results = []
with torch.inference_mode(), (out / 'pairs.jsonl').open('w') as output:
    for case_index, case in enumerate(cases):
        side = case['side']
        image = torch.as_tensor(case['image'], device='cuda', dtype=torch.float32)[None]
        observed = F.avg_pool2d(F.interpolate(image, (96, 96), mode='area'), 3, 1, 1)[0, 0]
        ouv = torch.as_tensor(case['ouv_um'], device='cuda')[None]
        state = full_frame_state_from_components(*physical_ouv_to_frame(ouv))
        states = torch.cat((state, compose_full_frame_state(state.expand(8, -1), updates)), 0)
        offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * case['thickness_um']
        weights = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], device='cuda')[None] / 16
        rendered = {name: render_atlas_planes_090(atlas, states[None],
            torch.zeros((1, 9), device='cuda', dtype=torch.long), offsets, weights,
            source_shape=(side, side))[0] for name, atlas in atlases.items()}
        assert torch.equal(rendered['average_template'][:, 1], rendered['nissl'][:, 1])
        smooth = {name: F.avg_pool2d(value[:, :1], 3, 1, 1)[:, 0]
                  for name, value in rendered.items()}
        null_images = []
        for candidate in range(9):
            key = f"{case['cohort']}:{case['section_id']}|{candidate}"
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63)
            intensity = rendered['nissl'][candidate, 0].flatten().clone()
            inside = (rendered['nissl'][candidate, 1].flatten() > .5).nonzero().flatten()
            order = torch.randperm(len(inside), device='cuda',
                generator=torch.Generator(device='cuda').manual_seed(seed))
            intensity[inside] = intensity[inside][order]
            null_images.append(intensity.reshape(96, 96))
        null_smooth = F.avg_pool2d(torch.stack(null_images)[:, None], 3, 1, 1)[:, 0]
        for candidate, candidate_name in enumerate(names, 1):
            support = rendered['average_template'][:, 1] > .5
            shared = support[0] & support[candidate]
            common = F.avg_pool2d(shared[None, None].float(), 5, 1, 2)[0, 0] >= 1 - 1e-6
            union = int((support[0] | support[candidate]).sum())
            iou = int(shared.sum()) / union if union else 0.
            area_gap = abs(int(support[0].sum()) - int(support[candidate].sum())) / (96 * 96)
            eligible = int(common.sum()) >= 64 and iou >= .45 and area_gap <= .15
            pair = {key: case[key] for key in ('cohort', 'donor_id', 'section_id', 'image_sha256')}
            pair.update({'case_index': case_index, 'candidate': candidate_name,
                'kind': 'offset' if candidate <= 4 else 'tilt', 'eligible': eligible,
                'common_cells': int(common.sum()), 'support_iou': iou,
                'support_area_fraction_gap': area_gap})
            for name in atlases:
                positive = score(observed, smooth[name][0], common)
                wrong = score(observed, smooth[name][candidate], common)
                pair[name] = {'positive': positive, 'wrong': wrong,
                              'win': bool(positive > wrong) if eligible else None}
            positive, wrong = score(observed, null_smooth[0], common), score(observed, null_smooth[candidate], common)
            pair['nissl_permuted'] = {'positive': positive, 'wrong': wrong,
                                      'win': bool(positive > wrong) if eligible else None}
            output.write(json.dumps(pair, allow_nan=False) + '\n')
            results.append(pair)
        if case_index + 1 in (58, 98):
            print(json.dumps({'event': 'acquired_sections_scored', 'count': case_index + 1}), flush=True)

summary = {'sections': len(cases), 'pairs': len(results), 'cohorts': {}}
for cohort in ('coronal_train', 'sagittal_train'):
    summary['cohorts'][cohort] = {}
    for kind in ('all', 'offset', 'tilt'):
        rows = [row for row in results if row['cohort'] == cohort and row['eligible']
                and (kind == 'all' or row['kind'] == kind)]
        donor_scores = {}
        for row in rows:
            donor_scores.setdefault(row['donor_id'], []).append(row)
        summary['cohorts'][cohort][kind] = {'eligible_pairs': len(rows),
            'eligible_donors': len(donor_scores),
            **{name: float(np.mean([np.mean([item[name]['win'] for item in values])
                                    for values in donor_scores.values()]))
               for name in ('average_template', 'nissl', 'nissl_permuted')}}
sag = summary['cohorts']['sagittal_train']
cor = summary['cohorts']['coronal_train']
summary['gate'] = ('inconclusive_support' if min(sag['tilt']['eligible_donors'],
    cor['all']['eligible_donors']) < 25 else 'advance_to_matched_train' if
    sag['tilt']['nissl'] >= .63 and
    sag['tilt']['nissl'] - sag['tilt']['average_template'] >= .10 and
    sag['tilt']['nissl'] - sag['tilt']['nissl_permuted'] >= .10 and
    cor['all']['nissl'] >= cor['all']['average_template'] - .05 and
    sag['offset']['nissl'] >= sag['offset']['average_template'] - .05 else
    'do_not_presume_nissl_improves_acquired_signal')
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
outputs = {name: sha(out / name) for name in ('config.json', 'pairs.jsonl', 'summary.json')}
(out / 'completed.json').write_text(json.dumps({'output_sha256': outputs, 'source_sha256': sha(source),
    'protocol_sha256': sha(protocol), 'gate': summary['gate']}, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
