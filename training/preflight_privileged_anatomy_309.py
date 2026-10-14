"""Read-only 309 TRAIN-inner upper bound: oracle tissue mask is never an inference input."""
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_fov_balance_v10 import load_fov_balance_v10, sample_fov_balance_v10
from training.arbitrary_plane_one_shot_inference import load_one_shot_checkpoint
from training.arbitrary_plane_streaming_synthetic_v7_64 import VARIANTS
from training.global_atlas_contrast_090 import render_atlas_planes_090, rigid_points_090


def correlation(left, right):
    left, right = left - left.mean(), right - right.mean()
    return float((left * right).sum() /
                 (left.square().sum() * right.square().sum()).sqrt().clamp_min(1e-8))


torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
run = root / 'runs/first_choice_exposure_309_001'
panel = torch.load(run / 'panels.pt', map_location='cpu', weights_only=True)['v10'][:64]
readout = next(json.loads(line) for line in (run / 'single/readouts.jsonl').open(encoding='utf8')
               if json.loads(line)['step'] == 20000)
record = {r['physical_section_id']: r for r in readout['rows'] if r['cohort'] == 'v10'}
context = load_fov_balance_v10('cuda')
model, _ = load_one_shot_checkpoint(run / 'single/pose_local_20000.pt')
model.requires_grad_(False)
rng = torch.Generator(device='cuda').manual_seed(20261014310)
rows = []

with torch.inference_mode():
    for item in panel:
        meta = item['meta']
        sample = sample_fov_balance_v10(context,
            [meta['base'] * VARIANTS + meta['variant']], meta['seed'], 192)
        assert torch.equal(sample['inputs'][0].cpu(), item['inputs'])
        assert sample['provenance'][0]['physical_section_id'] == meta['physical_section_id']
        inputs, chart, truth = (item[key][None].cuda() for key in ('inputs', 'chart', 'truth'))
        prediction = model.predict(inputs)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        branch = prior.topk(16, -1).indices
        state = prediction['state'].gather(1, (branch // 2)[..., None].expand(-1, -1, 12))
        rigid = rigid_points_090(state, branch % 2, chart[:, None], (192, 192))
        error = ((rigid - truth[:, None]).norm(dim=-1).mean(-1) / 1000)[0].cpu().numpy()
        assert abs(float(error[0]) - record[meta['physical_section_id']]['first_rigid_site_gap_mm']) < 1e-5
        assert abs(float(error.min()) - record[meta['physical_section_id']]['top16_oracle_gap_mm']) < 1e-5
        bank = render_atlas_planes_090(context['atlas'], state, branch % 2,
            sample['offsets'], sample['weights'], candidate_chunk=2,
            source_shape=(192, 192))[0]
        intact = sample['valid_mask'] & ~sample['fragment_displaced_mask'] & (sample['visible'] > .95)
        pure = F.avg_pool2d(intact[:, None].float(), 2)[0, 0] > .999
        source = F.avg_pool2d(inputs[:, :1], 2)[0, 0]
        scores = {'anatomy': [], 'support': [], 'shuffled': []}
        cells = []
        for candidate in range(16):
            support = bank[candidate, 1]
            mask = pure & (support > .95)
            count = int(mask.sum())
            cells.append(count)
            if count < 64:
                for arm in scores:
                    scores[arm].append(float('-inf'))
                continue
            image = source[mask]
            atlas = bank[candidate, 0][mask]
            scores['anatomy'].append(abs(correlation(image, atlas)))
            scores['support'].append(float(support[pure].mean()))
            scores['shuffled'].append(abs(correlation(image,
                atlas[torch.randperm(count, device='cuda', generator=rng)])))
        selected = {'prior': 0, **{name: int(np.argmax(values)) if max(values) > -float('inf')
                    else 0 for name, values in scores.items()}}
        rows.append({'id': meta['physical_section_id'], 'base': meta['base'],
                     'visible_fraction': meta['postartifact_visible_fraction'],
                     'obliquity_deg': meta['obliquity_deg'], 'cells': cells,
                     'errors': error.tolist(), 'scores': scores, 'selected': selected})

print('cases:', len(rows), 'mean eligible candidates:',
      round(float(np.mean([sum(c >= 64 for c in r['cells']) for r in rows])), 2),
      'cases with no eligible candidate:', sum(max(r['cells']) < 64 for r in rows))
oracle = float(np.mean([min(r['errors']) for r in rows]))
print('top16 oracle mean mm:', round(oracle, 4))
for arm in ('prior', 'anatomy', 'support', 'shuffled'):
    chosen = np.array([r['errors'][r['selected'][arm]] for r in rows])
    print(arm, 'selected mean mm:', round(float(chosen.mean()), 4),
          '>5mm:', int((chosen > 5).sum()),
          'changes vs prior:', sum(r['selected'][arm] != 0 for r in rows))
for control in ('prior', 'support', 'shuffled'):
    gain = np.array([r['errors'][r['selected'][control]] -
                     r['errors'][r['selected']['anatomy']] for r in rows])
    print('anatomy gain vs', control, 'mm:', round(float(gain.mean()), 4),
          'better/worse/tied:', (int((gain > 1e-8).sum()),
                               int((gain < -1e-8).sum()), int((abs(gain) <= 1e-8).sum())))
screen = all(np.mean([r['errors'][r['selected'][control]] -
                      r['errors'][r['selected']['anatomy']] for r in rows]) >= .25
             for control in ('prior', 'support', 'shuffled')) and sum(
                 r['errors'][r['selected']['anatomy']] > 5 for r in rows) <= sum(
                     r['errors'][0] > 5 for r in rows)
print('predeclared qualitative promotion screen:', bool(screen))
