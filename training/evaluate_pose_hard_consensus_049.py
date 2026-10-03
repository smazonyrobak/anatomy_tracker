"""Frozen 046 hard matches, spatial consensus, and full-plane pose correction."""
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
from training.arbitrary_plane_full_frame_primitives import (
    full_frame_state_from_components, full_frame_state_to_components,
)
from training.arbitrary_plane_geometry import physical_ouv_to_frame
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.atlas_oriented_patch_025 import extract_patches
from training.pose_feedback_global_041 import PoseFeedbackGlobal041
from training.pose_fine_match_046 import PoseFineMatch046
from training.pose_fine_patch_045 import atlas_key_patches

panel = root / 'data/pose_feedback_037_fresh_synthetic_dev_panel_001'
real = root / 'data/joint_v7_allen_fullcanvas_192_001'
out = root / 'runs/pose_hard_consensus_049_diagnostic'
parent = root / 'runs/one_shot_exposure_019/joint_step_18000.pt'
coarse_path = root / 'runs/pose_feedback_global_041_pilot/joint_step_01500.pt'
fine_path = root / 'runs/pose_fine_subgrid_046_pilot/joint_step_02000.pt'
seed, beam, hypotheses, shortlist = 2026104900, 8, 128, 16
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
generator = torch.Generator(device='cuda').manual_seed(seed)
synthetic = [json.loads(line) for line in (panel / 'records.jsonl').open()
             if json.loads(line)['eligible']]
real_records = [json.loads(line) for line in (real / 'records.jsonl').open()]
real_dev = [row for row in real_records if row['training_split'] == 'development']
assert len(synthetic) == 177 and len(real_dev) == 64
real_images = np.load(real / 'images.npy', mmap_mode='r')
with np.load(real / 'geometry.npz', allow_pickle=False) as arrays:
    affines = arrays['model_pixel_to_ap_dv_ml_um'].copy()
    thickness = arrays['thickness_um'].copy()
atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
pose = OneShotJointSliceModel(modes=16, atlas_conditioning=True, fit_quality=True,
                             vector_refinement=True, candidate_ranking=True,
                             fitted_ranking=True).cuda().eval()
pose.load_state_dict(torch.load(parent, map_location='cpu', weights_only=True)['model'])
coarse = PoseFeedbackGlobal041().cuda().eval()
coarse.load_state_dict(torch.load(coarse_path, map_location='cpu', weights_only=True)['head'])
fine = PoseFineMatch046().cuda().eval()
fine.load_state_dict(torch.load(fine_path, map_location='cpu', weights_only=True)['model'])
out.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
config = {'seed': seed, 'beam': beam, 'hypotheses': hypotheses,
          'queries': 16, 'shortlist': shortlist, 'consensus_radius_um': 1500,
          'parent_sha256': sha(parent), 'coarse_sha256': sha(coarse_path),
          'fine_sha256': sha(fine_path), 'panel_records_sha256': sha(panel / 'records.jsonl'),
          'real_records_sha256': sha(real / 'records.jsonl'),
          'source_sha256': sha(Path(__file__)),
          'calibrated': False, 'public_benchmark_used': False}
(out / 'config.json').write_text(json.dumps(config, indent=2))
positions = torch.arange(256, device='cuda').reshape(16, 16)
corners = torch.tensor([[0., 0.], [255., 0.], [0., 255.], [255., 255.],
                        [127.5, 127.5]], device='cuda') / 256


def physical_points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    chart = chart[:, None].expand(-1, state.shape[1], -1, -1).clone()
    chart[..., 0] = torch.where(reflection[..., None].bool(),
                                255 / 256 - chart[..., 0], chart[..., 0])
    return centre[..., None, :] + torch.einsum('bkij,bkqj->bkqi',
        frame[..., :, :2] @ basis, chart - .5)


def state_from_fit(fitted):
    centre, u, v = fitted.unbind(-2)
    return full_frame_state_from_components(*physical_ouv_to_frame(
        torch.stack((centre - .5 * (u + v), u, v), -2)))


rows = []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for subset, records in (('synthetic', synthetic), ('real_weak_allen', real_dev)):
        for record in records:
            if subset == 'synthetic':
                path = panel / record['file']
                assert sha(path) == record['sha256']
                with np.load(path, allow_pickle=False) as arrays:
                    image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                    target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                    valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                    offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                    weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
                ids = valid.flatten().nonzero().flatten()
                chart_eval = torch.stack((ids.remainder(256),
                    ids.div(256, rounding_mode='floor')), -1).float()[None] / 256
                reference = target.reshape(-1, 3)[ids]
                truth_grid = target[8::16, 8::16].reshape(256, 3)
                valid_grid = valid[8::16, 8::16].flatten()
            else:
                index = record['array_row_index']
                image = np.concatenate((real_images[index].astype(np.float32),
                    np.zeros((4, 192, 192), dtype=np.float32)))[None]
                image = F.interpolate(torch.from_numpy(image).cuda(), (256, 256),
                                      mode='bilinear', align_corners=False)
                offsets = torch.linspace(-.5, .5, 9, device='cuda')[None] * float(thickness[index])
                weights = torch.ones_like(offsets)
                weights[:, [0, -1]] = .5
                weights /= weights.sum(-1, keepdim=True)
                chart_eval = corners[None]
                affine = torch.as_tensor(affines[index], device='cuda', dtype=torch.float32)
                reference = affine[:, 2] + 192 * corners[:, :1] * affine[:, 0] \
                            + 192 * corners[:, 1:] * affine[:, 1]
            prediction = pose.predict(image)
            prior_score = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = prior_score.topk(beam, -1).indices
            state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            reflection = choice % 2
            coarse_result = coarse(prediction, image, state, reflection, atlas, offsets, weights)
            visibility = coarse_result['visibility_logit'][0]
            query = torch.cat([positions[y:y + 8, x:x + 8].flatten()[
                visibility[positions[y:y + 8, x:x + 8].flatten()].topk(4).indices]
                for y in (0, 8) for x in (0, 8)])
            query_yx = torch.stack((8 + 16 * (query // 16),
                                    8 + 16 * (query % 16)), -1).float()
            image_patches = extract_patches(image,
                torch.zeros(16, device='cuda', dtype=torch.long), query_yx)
            centre, frame, basis = full_frame_state_to_components(state)
            edges = frame[..., :, :2] @ basis
            qx = ((query % 16).float() + .5) / 16
            qy = ((query // 16).float() + .5) / 16
            qchart = torch.stack((qx, qy), -1)[None, None].expand(1, beam, -1, -1).clone()
            qchart[..., 0] = torch.where(reflection[..., None].bool(),
                                         255 / 256 - qchart[..., 0], qchart[..., 0])
            design = torch.cat((torch.ones_like(qchart[..., :1]), qchart - .5), -1)
            prior_fit = torch.stack((centre, edges[..., :, 0], edges[..., :, 1]), -2)
            matches, confidence = [], []
            true_recall = []
            for candidate in range(beam):
                top = coarse_result['logits'][0, candidate, query].topk(shortlist)
                world = coarse_result['key_world'][0, candidate, top.indices]
                support = coarse_result['key_support'][0, candidate, top.indices] > .1
                rotation = frame[0, candidate]
                relative = (world - coarse_result['query_base'][0, candidate, query, None]) @ rotation
                score_parts = []
                for first in range(0, 16, 4):
                    key = world[first:first + 4].reshape(-1, 3)
                    patch = atlas_key_patches(atlas, key,
                        state[0, candidate][None].expand(len(key), -1),
                        reflection[0, candidate][None].expand(len(key)),
                        offsets[0][None].expand(len(key), -1),
                        weights[0][None].expand(len(key), -1))
                    score_parts.append(fine(image_patches[first:first + 4], patch,
                        relative[first:first + 4], top.values[first:first + 4])[0])
                score = torch.cat(score_parts).masked_fill(~support, -1e4)
                probability = (score / 2).softmax(-1)
                selected = score.argmax(-1)
                hard = world.gather(1, selected[:, None, None].expand(-1, 1, 3))[:, 0]
                good = support.gather(1, selected[:, None])[:, 0]
                matches.append(hard)
                confidence.append(visibility[query].sigmoid() * probability.amax(-1) * good)
                if subset == 'synthetic':
                    true_recall.append(float(((hard - truth_grid[query]).norm(dim=-1) <= 1500)
                        [valid_grid[query]].float().mean()))
            hard = torch.stack(matches, 0)[None]
            confidence = torch.stack(confidence, 0)[None]
            ridge = torch.diag(state.new_tensor((2., 1., 1.)))
            lhs = torch.einsum('bkqi,bkq,bkqj->bkij', design, confidence, design) + ridge
            rhs = torch.einsum('bkqi,bkq,bkqj->bkij', design, confidence, hard) + ridge @ prior_fit
            all_fit = torch.linalg.solve(lhs, rhs)
            all_state = state_from_fit(all_fit)
            draw = torch.multinomial(confidence.reshape(beam, 16).clamp_min(1e-5),
                hypotheses * 4, replacement=True, generator=generator).reshape(1, beam, hypotheses, 4)
            sample_chart = torch.gather(design[:, :, None].expand(-1, -1, hypotheses, -1, -1),
                3, draw[..., None].expand(-1, -1, -1, -1, 3))
            sample_world = torch.gather(hard[:, :, None].expand(-1, -1, hypotheses, -1, -1),
                3, draw[..., None].expand(-1, -1, -1, -1, 3))
            weak = .05 * ridge
            hypothesis_fit = torch.linalg.solve(
                torch.einsum('bkhqi,bkhqj->bkhij', sample_chart, sample_chart) + weak,
                torch.einsum('bkhqi,bkhqj->bkhij', sample_chart, sample_world)
                    + weak @ prior_fit[:, :, None])
            residual = (torch.einsum('bkqi,bkhij->bkhqj', design, hypothesis_fit) -
                        hard[:, :, None]).norm(dim=-1)
            consensus_score = (confidence[:, :, None] * (residual <= 1500)).sum(-1)
            winner = consensus_score.argmax(-1)
            chosen = hypothesis_fit.gather(2, winner[..., None, None, None].expand(
                -1, -1, 1, 3, 3))[:, :, 0]
            inlier = (design @ chosen - hard).norm(dim=-1) <= 1500
            weight = confidence * inlier
            fit = torch.linalg.solve(
                torch.einsum('bkqi,bkq,bkqj->bkij', design, weight, design) + ridge,
                torch.einsum('bkqi,bkq,bkqj->bkij', design, weight, hard) + ridge @ prior_fit)
            robust_state = state_from_fit(fit)
            gated_state = state_from_fit(prior_fit + coarse_result['correction_gate'][..., None, None]
                                          * (fit - prior_fit))
            errors = {name: (physical_points(candidate_state, reflection, chart_eval) -
                reference[None, None]).norm(dim=-1).mean(-1)[0]
                for name, candidate_state in (('prior', state), ('hard_all', all_state),
                    ('robust_raw', robust_state), ('robust_gated', gated_state))}
            row = {'set': subset,
                   **{key: record[key] for key in ('animal_id', 'specimen_id',
                       'experiment_id', 'section_id')},
                   **({'synthetic_subject_plan_id': record['synthetic_subject_plan_id'],
                       'appearance_mode': record['appearance_mode'],
                       'sha256': record['sha256']} if subset == 'synthetic' else {}),
                   **{f'{name}_top1_um': float(error[0]) for name, error in errors.items()},
                   **{f'{name}_best8_um': float(error.min()) for name, error in errors.items()},
                   'top1_inlier_fraction': float(inlier[0, 0].float().mean()),
                   'best_prior_inlier_fraction': float(inlier[0, int(errors['prior'].argmin())].float().mean())}
            if subset == 'synthetic':
                row['top1_true_match_recall_1500um'] = true_recall[0]
                row['best_prior_true_match_recall_1500um'] = true_recall[int(errors['prior'].argmin())]
                row['selected_query_valid_fraction'] = float(valid_grid[query].float().mean())
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'set': subset, 'rows': len(records)}), flush=True)

summary = {'synthetic_eligible_sections': len(synthetic), 'synthetic_subjects': 8,
           'real_weak_sections': len(real_dev), 'real_weak_donors': 6,
           'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9}
for subset, group in (('synthetic', 'synthetic_subject_plan_id'),
                      ('real_weak_allen', 'animal_id')):
    selected = [row for row in rows if row['set'] == subset]
    identities = sorted({row[group] for row in selected})
    fields = [f'{name}_{endpoint}_um' for name in
              ('prior', 'hard_all', 'robust_raw', 'robust_gated')
              for endpoint in ('top1', 'best8')]
    fields += ['top1_inlier_fraction', 'best_prior_inlier_fraction']
    if subset == 'synthetic':
        fields += ['top1_true_match_recall_1500um',
                   'best_prior_true_match_recall_1500um', 'selected_query_valid_fraction']
    summary[subset] = {field: float(np.mean([np.mean([row[field] for row in selected
        if row[group] == identity]) for identity in identities])) for field in fields}
    if subset == 'real_weak_allen':
        summary['worst_real_donor_regression_um'] = {name: float(max(np.mean([
            row[f'{name}_top1_um'] - row['prior_top1_um'] for row in selected
            if row['animal_id'] == donor]) for donor in identities))
            for name in ('hard_all', 'robust_raw', 'robust_gated')}
summary['stage_promising'] = {name: bool(
    summary['synthetic']['prior_best8_um'] - summary['synthetic'][f'{name}_best8_um'] >= 200
    and summary['synthetic'][f'{name}_top1_um'] -
        summary['synthetic']['prior_top1_um'] <= 200
    and summary['worst_real_donor_regression_um'][name] <= 200)
    for name in ('hard_all', 'robust_raw', 'robust_gated')}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'), 'rows': len(rows),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'stage_promising': summary['stage_promising']}), flush=True)
