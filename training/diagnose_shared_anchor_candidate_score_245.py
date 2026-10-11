"""Replay frozen 244 DEV candidates at readout and training geometry."""
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

from training import arbitrary_plane_allen_atlas_binding_v6 as allen
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

training = root / 'runs/shared_anchor_joint_feedback_244_train_001'
panel = root / 'data/one_shot_fresh_synthetic_dev_panel_001'
out = root / 'runs/shared_anchor_candidate_score_245_001'
steps = (1000, 3000, 6000)
geometries = (('readout', 256, 64, 96, 8), ('training', 192, 48, 64, 3))
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


records = [json.loads(line) for line in (panel / 'records.jsonl').open()]
selected = [row for identity in sorted({row['animal_id'] for row in records})
            for row in [r for r in records if r['animal_id'] == identity and r['eligible']][:4]]
assert len(selected) == 32 and len({row['animal_id'] for row in selected}) == 8
assert all(sha(panel / row['file']) == row['sha256'] for row in selected)
atlas = torch.from_numpy(allen._decode_and_preprocess_allen_v6()[0]).cuda()
atlas_null = torch.stack((torch.roll(atlas[0], 80, 0), atlas[1]))
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64, shared_anchor_pose=True,
                              atlas_conditioning=True, vector_refinement=True,
                              fitted_ranking=True).cuda().eval()

out.mkdir(parents=True, exist_ok=False)
rows, cases = [], []
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for step in steps:
        checkpoint = torch.load(training / f'joint_step_{step:05d}.pt',
                                map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint

        for record in selected:
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image256 = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                reference = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            indices = valid.flatten().nonzero().flatten()
            xy = torch.stack((indices.remainder(256),
                              indices.div(256, rounding_mode='floor')), -1).float()
            target = reference.reshape(-1, 3)[indices]

            for geometry, side, first_side, fit_side, beam in geometries:
                image = image256 if side == 256 else F.interpolate(
                    image256, (side, side), mode='bilinear', align_corners=False)
                prediction = model.predict(image)
                prior = (prediction['log_mass'][0, :, None] + torch.stack((
                    F.logsigmoid(-prediction['reflection_logit'][0]),
                    F.logsigmoid(prediction['reflection_logit'][0])), -1)).flatten()
                choice = prior.topk(beam).indices
                state = prediction['state'][:, choice // 2]
                reflection = (choice % 2)[None]
                branch = {**prediction, 'state': state,
                          'log_mass': prediction['log_mass'][:, choice // 2],
                          'reflection_logit': prediction['reflection_logit'][:, choice // 2]}
                index = torch.arange(beam, device='cuda')[None]
                first = model.map(branch, offsets, index, reflection,
                                  (first_side, first_side), atlas, weights,
                                  return_refinement_feature=True, feature_side=first_side,
                                  source_shape=image.shape[-2:])
                refined_state, delta, _ = model.refine(first['refinement_feature'], state)
                refined = {**branch, 'state': refined_state}
                mapped = model.map(refined, offsets, index, reflection,
                                   (fit_side, fit_side), atlas, weights,
                                   feature_side=fit_side, source_shape=image.shape[-2:])
                fitted_base = model.score_fitted_candidates(
                    image, refined, mapped, atlas, weights)[0]
                null_base = model.score_fitted_candidates(
                    image, refined, mapped, atlas_null, weights)[0]
                fitted = fitted_base + delta[0]
                null = null_base + delta[0]

                chart = (xy + .5) / 256 - .5 / side
                chart = chart[None, None].expand(1, beam, -1, -1).clone()
                chart[..., 0] = torch.where(reflection[..., None].bool(),
                                            (side - 1) / side - chart[..., 0],
                                            chart[..., 0])
                centre, frame, basis = full_frame_state_to_components(refined_state)
                rigid_sites = centre[..., None, :] + torch.einsum(
                    '...ij,...pj->...pi', frame[..., :2] @ basis, chart - .5)
                rigid_error = (rigid_sites[0] - target).norm(dim=-1).mean(-1)

                surface = mapped['centre_surface_ccf_ap_dv_ml_um'][0].permute(0, 3, 1, 2)
                grid = ((xy + .5) * (2 / 256) - 1)[None, None].expand(
                    beam, 1, -1, -1)
                mapped_sites = F.grid_sample(
                    surface, grid, mode='bilinear', padding_mode='border',
                    align_corners=False)[:, :, 0].transpose(1, 2)
                mapped_error = (mapped_sites - target).norm(dim=-1).mean(-1)

                local = mapped['local_displacement_um'][0] / 1000
                magnitude = local.square().sum(1, keepdim=True)
                dx = F.pad((local[..., 1:] - local[..., :-1]).square().sum(1, keepdim=True),
                           (0, 1, 0, 0))
                dy = F.pad((local[..., 1:, :] - local[..., :-1, :]).square().sum(1, keepdim=True),
                           (0, 0, 0, 1))
                penalty = (magnitude + .1 * (dx + dy + 1e-8)).mean((1, 2, 3))
                local_rms_um = mapped['local_displacement_um'][0].square().mean(
                    (1, 2, 3)).sqrt()
                selected_prior = prior[choice]
                learned = fitted_base - selected_prior + .1 * penalty
                learned_null = null_base - selected_prior + .1 * penalty

                errors = rigid_error.cpu().numpy()
                scores = {'prior': selected_prior.cpu().numpy(),
                          'fitted': fitted.cpu().numpy(),
                          'fitted_without_prior': (fitted - selected_prior).cpu().numpy(),
                          'score_only_null': null.cpu().numpy()}
                pair = np.triu(np.abs(errors[:, None] - errors[None, :]) >= 250, 1)
                concordance = {}
                for name, values in scores.items():
                    direction = ((values[:, None] - values[None, :])
                                 * (errors[None, :] - errors[:, None]))[pair]
                    concordance[name] = (float(np.mean((direction > 0) + .5 * (direction == 0)))
                                         if len(direction) else None)
                chosen = {name: int(values.argmax()) for name, values in scores.items()}
                case = {'step': step, 'geometry': geometry,
                        'animal_id': record['animal_id'],
                        'panel_physical_section_id': record['panel_physical_section_id'],
                        'appearance_mode': record['appearance_mode'],
                        'unique_pose_modes': len(set((choice // 2).tolist())),
                        'prior_selected_rigid_um': float(rigid_error[chosen['prior']]),
                        'fitted_selected_rigid_um': float(rigid_error[chosen['fitted']]),
                        'null_selected_rigid_um': float(rigid_error[chosen['score_only_null']]),
                        'best_rigid_um': float(rigid_error.min()),
                        'best_selected': chosen['fitted'] == int(rigid_error.argmin()),
                        'score_concordance': concordance,
                        'delta_absmax': float(delta.abs().max()),
                        'learned_score_std': float(learned.std(unbiased=False)),
                        'prior_score_std': float(selected_prior.std(unbiased=False)),
                        'penalty_score_std': float((.1 * penalty).std(unbiased=False)),
                        'learned_null_change_absmean': float((learned - learned_null).abs().mean())}
                cases.append(case)
                for k in range(beam):
                    row = {'step': step, 'geometry': geometry,
                           **{key: record[key] for key in (
                               'animal_id', 'specimen_id', 'experiment_id', 'section_id',
                               'panel_physical_section_id', 'synthetic_subject_plan_id')},
                           'panel_file_sha256': record['sha256'],
                           'branch': int(choice[k]), 'pose_mode': int(choice[k] // 2),
                           'prior_log_score': float(selected_prior[k]),
                           'fitted_log_score': float(fitted[k]),
                           'fitted_without_prior': float(fitted[k] - selected_prior[k]),
                           'score_only_null': float(null[k]),
                           'learned_match_score': float(learned[k]),
                           'learned_null_score': float(learned_null[k]),
                           'deformation_penalty': float(.1 * penalty[k]),
                           'pose_delta_score': float(delta[0, k]),
                           'rigid_error_um': float(rigid_error[k]),
                           'mapped_error_um': float(mapped_error[k]),
                           'local_rms_um': float(local_rms_um[k])}
                    rows.append(row)
                    stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(json.dumps({'step': step, 'cases': len(selected) * len(geometries)}), flush=True)

summary = []
for step in steps:
    for geometry, *_ in geometries:
        group = [case for case in cases if case['step'] == step and case['geometry'] == geometry]
        identities = sorted({case['animal_id'] for case in group})
        metrics = ('prior_selected_rigid_um', 'fitted_selected_rigid_um',
                   'null_selected_rigid_um', 'best_rigid_um', 'unique_pose_modes',
                   'best_selected', 'delta_absmax', 'learned_score_std',
                   'prior_score_std', 'penalty_score_std',
                   'learned_null_change_absmean')
        summary.append({'step': step, 'geometry': geometry, 'cases': len(group),
                        'identities': len(identities),
                        'identity_equal_mean': {name: float(np.mean([
                            np.mean([case[name] for case in group if case['animal_id'] == identity])
                            for identity in identities])) for name in metrics},
                        'identity_equal_concordance': {name: float(np.mean([
                            np.mean([case['score_concordance'][name] for case in group
                                     if case['animal_id'] == identity
                                     and case['score_concordance'][name] is not None])
                            for identity in identities if any(
                                case['animal_id'] == identity
                                and case['score_concordance'][name] is not None
                                for case in group)])) for name in (
                                'prior', 'fitted', 'fitted_without_prior', 'score_only_null')}})
(out / 'cases.jsonl').write_text(''.join(json.dumps(case) + '\n' for case in cases))
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({
    'steps': steps, 'geometries': geometries, 'cases': len(cases), 'rows': len(rows),
    'checkpoint_sha256': {str(step): sha(training / f'joint_step_{step:05d}.pt') for step in steps},
    'panel_sha256': {name: sha(panel / name) for name in ('records.jsonl', 'protocol.json', 'completed.json')},
    'selected_panel_file_sha256': {row['file']: row['sha256'] for row in selected},
    'atlas_template_sha256': allen.TEMPLATE_RAW_SHA256_V6,
    'source_sha256': sha(Path(__file__)),
    'rows_sha256': sha(out / 'rows.jsonl'), 'cases_sha256': sha(out / 'cases.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False,
    'scope': 'same 32 frozen synthetic DEV sections, no train/CAL/FINAL/public data',
}, indent=2))
print(json.dumps({'event': 'complete', 'cases': len(cases), 'rows': len(rows)}), flush=True)
