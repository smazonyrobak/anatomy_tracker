"""Counterfactual 143 energy audit with exactly matched atlas support masks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import compose_full_frame_state
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.coherent_anatomy_field_143 import CoherentAnatomyField143
from training.coherent_anatomy_geometry_143 import render_coherent_atlas_143
from training.global_atlas_contrast_090 import rigid_points_090
from training.global_plane_matcher_120 import attach_global_plane_matcher


root = Path('I:/AnatomyTracker')
torch.set_num_threads(4)
atlas_array, _ = allen._decode_and_preprocess_allen_v6()
atlas = torch.from_numpy(atlas_array).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval().requires_grad_(False)
attach_global_plane_matcher(model, enabled=True)
model.load_state_dict(torch.load(root / 'runs/v4_pose_adaptation_132/joint_step_02000.pt',
                                 map_location='cpu', weights_only=True)['model'])
fields = {}
for arm in ('full', 'support_only'):
    field = CoherentAnatomyField143(radius=6).cuda().eval().requires_grad_(False)
    field.load_state_dict(torch.load(root / 'runs/coherent_anatomy_field_143' /
        f'{arm}_step_10000.pt', map_location='cpu', weights_only=True)['field'])
    fields[arm] = field
axis = (torch.arange(64, device='cuda') + .5) / 64
yy, xx = torch.meshgrid(axis, axis, indexing='ij')
chart = torch.stack((xx, yy), -1)
limits = torch.tensor([.07, .07, .07, 850., 850., 850., .04, .04, .03], device='cuda')
panels = {'v4': root / 'data/fresh_v4_pose_dev_panel_132',
          'v3': root / 'data/joint_in_path_correspondence_128_dev_panel'}
for cohort, panel in panels.items():
    result = []
    records = [record for record in
               (json.loads(line) for line in (panel / 'records.jsonl').open())
               if record['eligible']][:64]
    with torch.inference_mode():
        for record in records:
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                state = torch.from_numpy(arrays['target_state'][None].copy()).cuda()
                reflection = torch.from_numpy(arrays['reflection'].reshape(1).copy()).cuda().long()
                valid = torch.from_numpy(arrays['valid_mask'][2::4, 2::4].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            feature = model.predict(image)['feature']
            observed = chart[valid]
            truth = rigid_points_090(state, reflection, observed)[0]
            seed = int(hashlib.sha256(record['section_id'].encode()).hexdigest()[:15], 16)
            generator = torch.Generator(device='cuda').manual_seed(seed)
            updates = (2 * torch.rand((64, 9), device='cuda', generator=generator) - 1) * limits
            updates[:, :3] *= 3
            updates[:, 3:6] *= 3
            updates[:, 6:] *= 2
            states = compose_full_frame_state(state.expand(64, -1), updates)
            errors = (rigid_points_090(states, reflection.expand(64), observed) -
                      truth).norm(dim=-1).mean(-1) / 1000
            eligible = (errors >= 1.5) & (errors <= 3)
            candidates = (errors - 2.2).abs().masked_fill(~eligible, 100).topk(8,
                largest=False).indices
            wrong, _, _, _ = render_coherent_atlas_143(atlas, states[candidates],
                reflection.expand(8), offsets.expand(8, -1), weights.expand(8, -1))
            exact, _, _, _ = render_coherent_atlas_143(atlas, state, reflection, offsets, weights)
            exact_support = exact[0, 1, 3] > .5
            wrong_support = wrong[:, 1, 3] > .5
            overlap = ((wrong_support & exact_support).sum((1, 2)) /
                       (wrong_support | exact_support).sum((1, 2)).clamp_min(1))
            choice = int(overlap.argmax())
            wrong = wrong[choice:choice + 1]
            common = ((exact[:, 1:2] > .5) & (wrong[:, 1:2] > .5)).float()
            shared_support = common.expand(2, -1, -1, -1, -1)
            full = torch.cat((exact, wrong), 0)
            full = torch.cat((full[:, :1] * shared_support, shared_support), 1)
            support = torch.cat((torch.zeros_like(full[:, :1]), shared_support), 1)
            scrambled = full.clone()
            for slot in range(2):
                for depth in range(7):
                    mask = common[0, 0, depth].bool()
                    values = scrambled[slot, 0, depth][mask]
                    permutation = torch.randperm(len(values), device='cuda', generator=generator)
                    scrambled[slot, 0, depth][mask] = values[permutation]
            energies = {}
            for arm, pair in (('full', full), ('support_only', support),
                              ('scrambled', scrambled)):
                field = fields['full' if arm == 'scrambled' else arm]
                with torch.autocast('cuda', dtype=torch.float16):
                    output = field(image.expand(2, -1, -1, -1), pair, 500.,
                                   source_feature=feature.expand(2, -1, -1, -1))
                energies[arm] = [float(v) for v in output['energy']]
            result.append({'section_id': record['section_id'],
                'wrong_rigid_mm': float(errors[candidates[choice]]),
                'central_support_iou': float(overlap[choice]),
                'common_support_fraction': float(common[:, :, 3].mean()),
                'energy': energies})
    summary = {'cohort': cohort, 'sections': len(result),
        'mean_central_support_iou': float(np.mean([r['central_support_iou'] for r in result])),
        'mean_common_support_fraction': float(np.mean([r['common_support_fraction'] for r in result]))}
    for arm in ('full', 'support_only', 'scrambled'):
        margin = np.array([r['energy'][arm][1] - r['energy'][arm][0] for r in result])
        summary[arm] = {'rank_percent': float(100 * ((margin > 0).mean() + .5 * (margin == 0).mean())),
                        'median_margin': float(np.median(margin))}
    print(json.dumps(summary))
