"""Read-only synthetic-DEV evaluation of the 088 null-correspondence head."""
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
from scipy.stats import spearmanr

from training.arbitrary_plane_allen_atlas_binding_v6 import _decode_and_preprocess_allen_v6
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel
from training.whole_slice_atlas_feedback_083 import synthetic_match_targets
from training.whole_slice_atlas_feedback_088 import WholeSliceAtlasFeedback088

panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
run = root / 'runs/outlier_correspondence_088_train'
rank_run = root / 'runs/allbeam_candidate_ranking_086_development_diagnostic'
out = root / 'runs/outlier_correspondence_088_development_eval'
side = 256
steps = (0, 2000, 10000, 20000)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def points(state, reflection, chart):
    centre, frame, basis = full_frame_state_to_components(state)
    xy = chart.expand(*state.shape[:-1], *chart.shape[-2:]).clone()
    xy[..., 0] = torch.where(reflection[..., None].bool(), 255 / side - xy[..., 0], xy[..., 0])
    return centre[..., None, :] + torch.einsum(
        '...ij,...pj->...pi', frame[..., :, :2] @ basis, xy - .5)


def fit_summary(mapped):
    local = mapped['local_displacement_um'] / 1000
    magnitude = local.square().sum(2).sqrt().mean((-2, -1))
    roughness = ((local[..., 1:] - local[..., :-1]).square().sum(2).sqrt().mean((-2, -1))
                 + (local[..., 1:, :] - local[..., :-1, :]).square().sum(2).sqrt().mean((-2, -1)))
    support = mapped['atlas_pair'][:, :, 1].mean((-2, -1))
    reliability = mapped['correspondence_logit'].sigmoid().mean((-3, -2, -1))
    return torch.stack((magnitude, roughness, support, reliability), -1)


def candidate_prediction(prediction, branch, state):
    mode = branch // 2
    return {**prediction, 'state': state,
            'log_mass': prediction['log_mass'].gather(1, mode),
            'reflection_logit': prediction['reflection_logit'].gather(1, mode)}


def field_at_chart(surface, chart):
    count = surface.shape[1]
    grid = (chart + .5 / side) * 2 - 1
    grid = grid[None].expand(count, -1, -1).reshape(count, 1, len(chart), 2)
    return F.grid_sample(surface[0].permute(0, 3, 1, 2), grid,
                         padding_mode='border', align_corners=False).squeeze(2).transpose(1, 2)


receipt = json.loads((run / 'completed.json').read_text())
training = json.loads((run / 'config.json').read_text())
panel_receipt = json.loads((panel / 'completed.json').read_text())
rank_receipt = json.loads((rank_run / 'completed.json').read_text())
assert receipt['batches'] == training['batches'] == 20000
assert tuple(training['checkpoints']) == steps
assert not receipt['calibrated'] and not receipt['public_benchmark_used']
assert sha(run / 'config.json') == receipt['config_sha256']
assert sha(run / 'draws.jsonl') == receipt['draws_sha256']
assert sha(run / 'training.jsonl') == receipt['training_sha256']
assert sha(training['parent']) == training['parent_sha256']
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in training['source_sha256'].items())
assert all(sha(Path(__file__).parent / name) == digest
           for name, digest in training['synthetic_provenance']['source_sha256'].items())
assert all(sha(run / f'match_step_{step:05d}.pt') == receipt['checkpoint_sha256'][str(step)]
           for step in steps)
assert sha(panel / 'completed.json') == '3fee767a1b33b83a13560a6766e38fa9502d967e54bbddd5aafd74994d5c5a8b'
assert sha(panel / 'records.jsonl') == panel_receipt['records_sha256'] \
       == '481a66fd4697e5ee6de737787341ddb426b0b8655c9bc9c02d1234522a3d766a'
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
assert len(records) == 246 and len({row['synthetic_subject_plan_id'] for row in records}) == 8
assert sha(rank_run / 'completed.json') == '5eb6dc84b615f3ce01ee9ac84ce124057a42a6dda39dbc2b999c87c1e05b3711'
assert rank_receipt['sections'] == 246 and rank_receipt['candidates'] == 246 * 14
assert sha(rank_run / 'config.json') == rank_receipt['config_sha256']
assert sha(rank_run / 'candidates.jsonl') == rank_receipt['candidates_sha256']
assert sha(rank_run / 'sections.jsonl') == rank_receipt['sections_sha256']
assert sha(rank_run / 'summary.json') == rank_receipt['summary_sha256']
ranked = {}
for row in map(json.loads, (rank_run / 'candidates.jsonl').open()):
    ranked.setdefault(row['section_id'], {})[row['beam_slot']] = row['branch_id']
assert len(ranked) == len(records) and all(len(branches) == 14 for branches in ranked.values())

atlas = torch.from_numpy(_decode_and_preprocess_allen_v6()[0]).cuda()
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.requires_grad_(False)
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'training_completed_sha256': sha(run / 'completed.json'),
    'training_config_sha256': sha(run / 'config.json'),
    'checkpoint_sha256': receipt['checkpoint_sha256'],
    'panel_completed_sha256': sha(panel / 'completed.json'),
    'panel_records_sha256': sha(panel / 'records.jsonl'),
    'ranking_086_completed_sha256': sha(rank_run / 'completed.json'),
    'ranking_086_candidates_sha256': sha(rank_run / 'candidates.jsonl'),
    'evaluator_sha256': sha(__file__), 'checkpoints': steps,
    'beam': 'frozen 085 prior: top 8 old branches and top 6 anchor branches, no truth insertion',
    'labels': '083 local target and atlas support >= 0.5; all other visible centres are null; background null reported separately',
    'matchability_rank_target': 'first-pass 14-branch mean rigid CCF error on 1024 fixed surviving pixels, before 088 update',
    'secondary_rank_target': 'two-pass mapped96 CCF error on those pixels; downstream readout under 083-to-088 spatial shift',
    'mapped_metric': 'full two-pass 088 head, 085 fitted score plus 088 quality, mapped96 CCF error on the same fixed 1024 pixels',
    'mapped_caveat': '085 mapper and fitted scorer were trained with 083 spatial evidence; 088 matchability replaces the confidence channel',
    'calibrated': False, 'public_benchmark_used': False}, indent=2))

rows, sections = [], []
with torch.inference_mode(), (out / 'candidates.jsonl').open('w') as candidate_stream, \
        (out / 'sections.jsonl').open('w') as section_stream:
    for step in steps:
        checkpoint = torch.load(run / f'match_step_{step:05d}.pt',
                                map_location='cpu', weights_only=True)
        assert checkpoint['step'] == step and not checkpoint['calibrated']
        model.load_state_dict(checkpoint['model'], strict=True)
        head = WholeSliceAtlasFeedback088().cuda().eval()
        head.load_state_dict(checkpoint['head'], strict=True)
        head.requires_grad_(False)
        del checkpoint
        for number, record in enumerate(records):
            if step == 0:
                assert sha(panel / record['file']) == record['sha256']
            with np.load(panel / record['file'], allow_pickle=False) as arrays:
                image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
                target = torch.from_numpy(arrays['target_centre_um'][None].copy()).cuda()
                valid = torch.from_numpy(arrays['valid_mask'][None].copy()).cuda().bool()
                offsets = torch.from_numpy(arrays['offsets_um'][None].copy()).cuda()
                weights = torch.from_numpy(arrays['weights'][None].copy()).cuda()
            indices = valid.flatten().nonzero().flatten()
            chosen = indices[torch.linspace(0, len(indices) - 1, 1024,
                                            device='cuda').round().long()]
            chart = torch.stack((chosen.remainder(side),
                chosen.div(side, rounding_mode='floor')), -1).float() / side
            reference = target.reshape(-1, 3)[chosen]
            prediction = model.predict(image)
            prior = (prediction['log_mass'][..., None] + torch.stack((
                F.logsigmoid(-prediction['reflection_logit']),
                F.logsigmoid(prediction['reflection_logit'])), -1)).flatten(1)
            choice = torch.cat((prior[:, :32].topk(8, -1).indices,
                                prior[:, 32:].topk(6, -1).indices + 32), -1)
            assert choice[0].tolist() == [ranked[record['section_id']][slot]
                                          for slot in range(14)]
            state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
            rigid_error = (points(state, choice % 2, chart) - reference).norm(dim=-1).mean(-1)[0]
            candidates, scores, surfaces = [], [], []
            for start in range(0, 14, 2):
                branch = choice[:, start:start + 2]
                reflection = branch % 2
                initial = state[:, start:start + 2]
                labels = synthetic_match_targets(target, valid, initial, reflection)
                first = head(prediction['feature'], initial, reflection, atlas, offsets,
                             weights, return_match_logits=True)
                metrics = [{} for _ in range(2)]
                for scale, width in (('fine', 32), ('coarse', 16)):
                    logits = first[f'{scale}_match_logits']
                    index = labels[f'{scale}_index']
                    support = first[f'{scale}_match_support'].gather(2, index[:, :, None]).squeeze(2)
                    visible = F.interpolate(valid[:, None].float(), (width, width),
                                            mode='bilinear', align_corners=False)[:, 0] == 1
                    positive = labels[f'{scale}_mask'] & (support >= .5)
                    null_visible = visible[:, None] & ~positive
                    null_background = ~visible[:, None].expand_as(positive)
                    target_index = torch.where(positive, index, logits.shape[2] - 1)
                    ce = F.cross_entropy(logits.flatten(0, 1), target_index.flatten(0, 1),
                                         reduction='none').reshape_as(index)
                    for slot in range(2):
                        for name, mask in (('positive', positive[0, slot]),
                                           ('null_visible', null_visible[0, slot]),
                                           ('null_background', null_background[0, slot])):
                            count = int(mask.sum())
                            metrics[slot][f'{scale}_{name}_count'] = count
                            metrics[slot][f'{scale}_{name}_ce'] = (
                                float(ce[0, slot][mask].mean()) if count else None)
                selected = candidate_prediction(prediction, branch, first['state'])
                index = torch.arange(2, device='cuda')[None]
                mapped64 = model.map(selected, offsets, index, reflection, (64, 64),
                                     atlas, weights, return_refinement_feature=True,
                                     feature_side=64, source_shape=(side, side),
                                     spatial_evidence=first['spatial_evidence'])
                second = head(prediction['feature'], first['state'], reflection,
                              atlas, offsets, weights, fit_summary=fit_summary(mapped64))
                selected['state'] = second['state']
                mapped96 = model.map(selected, offsets, index, reflection, (96, 96),
                                     atlas, weights, feature_side=96,
                                     source_shape=(side, side),
                                     spatial_evidence=second['spatial_evidence'])
                score = model.score_fitted_candidates(image, selected, mapped96,
                                                      atlas, weights) + second['quality_logit']
                scores.append(score)
                surfaces.append(mapped96['centre_surface_ccf_ap_dv_ml_um'])
                for slot in range(2):
                    candidates.append({**record, **metrics[slot], 'step': step,
                        'beam_slot': start + slot, 'branch_id': int(branch[0, slot]),
                        'family': 'old' if int(branch[0, slot]) < 32 else 'anchor',
                        'mode_index': int(branch[0, slot] // 2),
                        'reflection': int(reflection[0, slot]),
                        'prior_score': float(prior[0, branch[0, slot]]),
                        'match_probability': float(first['match_probability'][0, slot]),
                        'initial_rigid_error_um': float(rigid_error[start + slot]),
                        'fitted_score': float(score[0, slot])})
            score = torch.cat(scores, 1)[0]
            surface = torch.cat(surfaces, 1)
            mapped_error = (field_at_chart(surface, chart) - reference[None]).norm(dim=-1).mean(-1)
            match = np.array([row['match_probability'] for row in candidates])
            rigid = rigid_error.cpu().numpy()
            selected = int(score.argmax())
            match_selected = int(match.argmax())
            best = int(mapped_error.argmin())
            rigid_best = int(rigid.argmin())
            match_order = np.argsort(-match, kind='stable')
            rigid_order = np.argsort(rigid, kind='stable')
            match_rank = np.empty(14, dtype=int)
            rigid_rank = np.empty(14, dtype=int)
            match_rank[match_order] = np.arange(1, 15)
            rigid_rank[rigid_order] = np.arange(1, 15)
            rank = float(spearmanr(match, -rigid).statistic)
            mapped_rank = float(spearmanr(match, -mapped_error.cpu().numpy()).statistic)
            for slot, row in enumerate(candidates):
                row.update({'mapped96_error_um': float(mapped_error[slot]),
                    'matchability_rank': int(match_rank[slot]),
                    'rigid_error_rank': int(rigid_rank[slot]),
                    'fitted_selected': slot == selected,
                    'matchability_selected': slot == match_selected,
                    'rigid_best': slot == rigid_best,
                    'mapped_best': slot == best})
                rows.append(row)
                candidate_stream.write(json.dumps(row, allow_nan=False) + '\n')
            section = {key: record[key] for key in ('section_id', 'animal_id',
                'synthetic_animal_id', 'specimen_id', 'experiment_id',
                'synthetic_subject_plan_id',
                'synthetic_subject_realization_id', 'panel_physical_section_id',
                'appearance_mode', 'file', 'sha256')}
            section.update({'step': step, 'matchability_vs_initial_rigid_spearman': rank,
                'matchability_vs_mapped96_spearman': mapped_rank,
                'fitted_selected_branch': int(choice[0, selected]),
                'matchability_selected_branch': int(choice[0, match_selected]),
                'rigid_best_branch': int(choice[0, rigid_best]),
                'mapped_best_branch': int(choice[0, best]),
                'fitted_selected_mapped96_um': float(mapped_error[selected]),
                'matchability_selected_mapped96_um': float(mapped_error[match_selected]),
                'best14_mapped96_um': float(mapped_error[best]),
                'matchability_selected_initial_rigid_um': float(rigid[match_selected]),
                'best14_initial_rigid_um': float(rigid[rigid_best])})
            sections.append(section)
            section_stream.write(json.dumps(section, allow_nan=False) + '\n')
        candidate_stream.flush()
        section_stream.flush()
        print(json.dumps({'checkpoint': step, 'sections': len(records)}), flush=True)

metrics = ('matchability_vs_initial_rigid_spearman', 'matchability_vs_mapped96_spearman',
           'fitted_selected_mapped96_um',
           'matchability_selected_mapped96_um', 'best14_mapped96_um',
           'matchability_selected_initial_rigid_um', 'best14_initial_rigid_um')
ce_names = tuple(f'{scale}_{label}_ce' for scale in ('fine', 'coarse')
                 for label in ('positive', 'null_visible', 'null_background'))
summary = {}
for step in steps:
    group = [row for row in sections if row['step'] == step]
    plan_ids = sorted({row['synthetic_subject_plan_id'] for row in group})
    by_plan = {plan: {metric: float(np.mean([row[metric] for row in group
        if row['synthetic_subject_plan_id'] == plan])) for metric in metrics}
        for plan in plan_ids}
    result = {'sections': len(group), 'plans': len(plan_ids),
        'plan_equal_mean': {metric: float(np.mean([by_plan[plan][metric] for plan in plan_ids]))
                            for metric in metrics}}
    for scale in ('fine', 'coarse'):
        for label in ('positive', 'null_visible', 'null_background'):
            name = f'{scale}_{label}_ce'
            per_plan = [np.mean([row[name] for row in rows if row['step'] == step
                        and row['synthetic_subject_plan_id'] == plan and row[name] is not None])
                        for plan in plan_ids]
            result[name] = float(np.mean(per_plan))
            result[f'{scale}_{label}_count'] = sum(row[f'{scale}_{label}_count']
                for row in rows if row['step'] == step)
    result['by_appearance'] = {}
    for appearance in ('exact_black', 'raw', 'imperfect_brush'):
        appearance_sections = [row for row in group if row['appearance_mode'] == appearance]
        appearance_candidates = [row for row in rows if row['step'] == step
                                 and row['appearance_mode'] == appearance]
        appearance_plans = sorted({row['synthetic_subject_plan_id']
                                   for row in appearance_sections})
        appearance_result = {'sections': len(appearance_sections),
            'plans': len(appearance_plans),
            'plan_equal_mean': {metric: float(np.mean([
                np.mean([row[metric] for row in appearance_sections
                         if row['synthetic_subject_plan_id'] == plan])
                for plan in appearance_plans])) for metric in metrics}}
        for name in ce_names:
            appearance_result[name] = float(np.mean([
                np.mean([row[name] for row in appearance_candidates
                         if row['synthetic_subject_plan_id'] == plan and row[name] is not None])
                for plan in appearance_plans]))
        result['by_appearance'][appearance] = appearance_result
    summary[str(step)] = result
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'completed.json').write_text(json.dumps({
    'sections': len(sections), 'candidates': len(rows),
    'config_sha256': sha(out / 'config.json'),
    'candidates_sha256': sha(out / 'candidates.jsonl'),
    'sections_sha256': sha(out / 'sections.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
