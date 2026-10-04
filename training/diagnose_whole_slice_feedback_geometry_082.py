"""Geometric capture of exact 061 DEV correspondences by frozen 081 search windows."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
os.environ['CUDA_CACHE_PATH'] = str(root / 'cache/cuda')
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.whole_slice_atlas_feedback_081 import WholeSliceAtlasFeedback081

panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
pilot = root / 'runs/whole_slice_joint_feedback_081_pilot'
evaluation = root / 'runs/whole_slice_joint_feedback_081_development_eval'
out = root / 'runs/whole_slice_feedback_geometry_082'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def query_centres(target, valid, block):
    lo = np.arange(256 // block) * block + block // 2 - 1
    hi = lo + 1
    yy, xx = np.meshgrid(lo, lo, indexing='ij')
    valid_query = valid[yy, xx] & valid[yy, xx + 1] & valid[yy + 1, xx] & valid[yy + 1, xx + 1]
    truth = (target[yy, xx] + target[yy, xx + 1] + target[yy + 1, xx]
             + target[yy + 1, xx + 1]) / 4
    xy = np.stack((xx + .5, yy + .5), -1)
    return torch.from_numpy(truth[valid_query].copy()).cuda(), torch.from_numpy(xy[valid_query].copy()).cuda()


def capture(state, reflection, truth, xy, radius):
    centre, frame, basis = full_frame_state_to_components(state)
    local = torch.einsum('cni,cij->cnj', truth[None] - centre[:, None], frame)
    chart = .5 + torch.einsum('cni,cij->cnj', local[..., :2], torch.linalg.inv(basis).transpose(-1, -2))
    qx = torch.where(reflection[:, None].bool(), (255 - xy[:, 0])[None] / 256,
                     xy[:, 0][None] / 256)
    lateral = 32 * torch.maximum((chart[..., 0] - qx).abs(),
                                  (chart[..., 1] - xy[:, 1][None] / 256).abs())
    normal = local[..., 2].abs() <= 6000
    inside = lateral <= radius
    flags = torch.stack((normal, inside, normal & inside), -1)
    return flags.cpu().numpy(), lateral.cpu().numpy(), local[..., 2].abs().cpu().numpy()


def fractions(flags):
    return flags.mean(1).tolist()


records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
frozen_rows = {row['section_id']: row for row in map(json.loads, (evaluation / 'rows.jsonl').open())
               if row['set'] == 'synthetic' and row['step'] == 3000}
checkpoint = torch.load(pilot / 'joint_step_03000.pt', map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64, atlas_conditioning=True,
    fit_quality=True, vector_refinement=True, candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(checkpoint['model'], strict=True)
model.requires_grad_(False)
head = WholeSliceAtlasFeedback081().cuda().eval()
head.load_state_dict(checkpoint['feedback'], strict=True)
head.requires_grad_(False)
del checkpoint
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
out.mkdir(parents=True, exist_ok=False)
provenance = {'checkpoint': str(pilot / 'joint_step_03000.pt'),
              'checkpoint_sha256': hashlib.sha256((pilot / 'joint_step_03000.pt').read_bytes()).hexdigest(),
              'panel_records_sha256': hashlib.sha256((panel / 'records.jsonl').read_bytes()).hexdigest(),
              'evaluation_rows_sha256': hashlib.sha256((evaluation / 'rows.jsonl').read_bytes()).hexdigest(),
              'sections': len(records), 'synthetic_plans': len({r['synthetic_subject_plan_id'] for r in records}),
              'window': {'depth_centres_um': list(range(-6000, 6001, 1500)),
                         'fine': '32x32 query, +/-2 fine key cells = +/-2/32 chart units',
                         'coarse': '16x16 query, +/-4 coarse key cells = +/-8/32 chart units',
                         'atlas_margin_fine_cells': 8,
                         'normal_test': '|signed normal coordinate| <= 6000 um',
                         'query_truth': 'four-neighbour bilinear CCF correspondence; all four visible pixels required'},
              'selected': 'frozen 081 evaluation score winner; no truth used for selection',
              'population': 'eligible 061 DEV synthetic deformation plans, not biological animals'}
(out / 'config.json').write_text(json.dumps(provenance, indent=2))

rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for record in records:
        with np.load(panel / record['file'], allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = arrays['target_centre_um']
            valid = arrays['valid_mask'].astype(bool)
            offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
            weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
        queries = [query_centres(target, valid, block) for block in (8, 16)]
        prediction = model.predict(image)
        prior = (prediction['log_mass'][..., None] + torch.stack((
            F.logsigmoid(-prediction['reflection_logit']),
            F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
        choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                            prior[:, 32:].topk(6, -1).indices + 32), -1)
        state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        selected_branch = frozen_rows[record['section_id']]['model_selected_branch']
        selected_slot = int((choice[0] == selected_branch).nonzero()[0, 0])
        corrected = []
        for start in range(0, 14, 2):
            c, s = choice[:, start:start + 2], state[:, start:start + 2]
            reflection = c % 2
            first = head(prediction['feature'], s, reflection, atlas, offsets, weights,
                         source_shape=(256, 256))
            selected = {**prediction, 'state': first['state'],
                        'log_mass': prediction['log_mass'].gather(1, c // 2),
                        'reflection_logit': prediction['reflection_logit'].gather(1, c // 2)}
            mapped = model.map(selected, offsets, torch.arange(s.shape[1], device='cuda')[None],
                reflection, (64, 64), atlas, weights, return_refinement_feature=True,
                feature_side=64, source_shape=(256, 256), spatial_evidence=first['spatial_evidence'])
            local = mapped['local_displacement_um'] / 1000
            magnitude = local.square().sum(2).sqrt().mean((-2, -1))
            roughness = ((local[..., 1:] - local[..., :-1]).square().sum(2).sqrt().mean((-2, -1))
                         + (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).sqrt().mean((-2, -1)))
            support = mapped['atlas_pair'][:, :, 1].mean((-2, -1))
            reliability = mapped['correspondence_logit'].sigmoid().mean((-3, -2, -1))
            summary = torch.stack((magnitude, roughness, support, reliability), -1)
            second = head(prediction['feature'], first['state'], reflection, atlas, offsets,
                          weights, source_shape=(256, 256), fit_summary=summary)
            corrected.append(second['state'])
        stage_states = {'blind': state[0], 'feedback2': torch.cat(corrected, 1)[0]}
        row = {'section_id': record['section_id'],
               'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
               'appearance_mode': record['appearance_mode'],
               'valid_pixels': int(valid.sum()), 'query_centres': [len(q[0]) for q in queries],
               'branches': choice[0].tolist(), 'selected_slot': selected_slot, 'stages': {}}
        for stage, states in stage_states.items():
            metrics = {}
            for name, radius, query in zip(('fine', 'coarse'), (2, 8), queries):
                flags, lateral, normal_um = capture(states, choice[0] % 2, *query, radius)
                metrics[name] = {'per_candidate': fractions(flags),
                                 'union_per_pixel': [float(v) for v in flags.any(0).mean(0)],
                                 'selected_normal_abs_um_p50_p90': np.quantile(
                                     normal_um[selected_slot], [.5, .9]).tolist(),
                                 'selected_lateral_fine_cells_p50_p90': np.quantile(
                                     lateral[selected_slot], [.5, .9]).tolist()}
            row['stages'][stage] = metrics
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if len(rows) % 25 == 0:
            print(json.dumps({'sections': len(rows)}), flush=True)

summary = {'sections': len(rows), 'synthetic_plans': provenance['synthetic_plans'],
           'query_centres': {name: {'total': int(sum(row['query_centres'][i] for row in rows)),
                                    'min_per_section': int(min(row['query_centres'][i] for row in rows))}
                             for i, name in enumerate(('fine', 'coarse'))}, 'stages': {}}
for stage in ('blind', 'feedback2'):
    summary['stages'][stage] = {}
    for scale in ('fine', 'coarse'):
        groups = {name: [] for name in ('selected', 'best14_single', 'best8_old_single',
                                        'best6_anchor_single', 'union14_per_pixel')}
        winner_anchor = 0
        for row in rows:
            data = row['stages'][stage][scale]
            candidate = np.asarray(data['per_candidate'])
            chosen = {'selected': row['selected_slot'],
                      'best14_single': int(candidate[:, 2].argmax()),
                      'best8_old_single': int(candidate[:8, 2].argmax()),
                      'best6_anchor_single': int(candidate[8:, 2].argmax()) + 8}
            winner_anchor += chosen['best14_single'] >= 8
            for name, slot in chosen.items():
                groups[name].append((row['synthetic_subject_plan_id'], candidate[slot]))
            groups['union14_per_pixel'].append((row['synthetic_subject_plan_id'],
                                                np.asarray(data['union_per_pixel'])))
        summary['stages'][stage][scale] = {'best14_winner_anchor_sections': winner_anchor}
        for name, entries in groups.items():
            values = np.asarray([entry[1] for entry in entries])
            plans = {entry[0] for entry in entries}
            summary['stages'][stage][scale][name] = {
                'mean_fraction_normal_lateral_joint': values.mean(0).tolist(),
                'plan_equal_mean_fraction_normal_lateral_joint': [float(np.mean([
                    values[[i for i, entry in enumerate(entries) if entry[0] == plan], j].mean()
                    for plan in plans])) for j in range(3)],
                'sections_joint_ge_50_90_100_percent': [int((values[:, 2] >= threshold).sum())
                                                       for threshold in (.5, .9, 1.)]}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({'event': 'complete', 'summary': str(out / 'summary.json')}), flush=True)
