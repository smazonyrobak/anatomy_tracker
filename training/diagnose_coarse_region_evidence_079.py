"""Frozen 059 blind-proposal selection with unavailable oracle Allen region labels."""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path('I:/AnatomyTracker')
os.environ['TEMP'] = os.environ['TMP'] = str(root / 'tmp')
os.environ['TORCH_HOME'] = str(root / 'cache/torch')
sys.dont_write_bytecode = True

import nrrd
import numpy as np
import torch
import torch.nn.functional as F

from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components
from training.arbitrary_plane_one_shot_model import OneShotJointSliceModel

repo = Path(__file__).resolve().parent.parent
panel = root / 'data/pose_feedback_061_fresh_synthetic_dev_panel_001'
parent_run = root / 'runs/normal_conditioned_cost_volume_077_pilot'
previous = root / 'runs/normal_conditioned_cost_volume_077_development_eval/rows.jsonl'
annotation_path = root / 'data/Allen Brain Atlas 25um/annotation_25.nrrd'
ontology_path = root / 'data/joint_v7_probe_observation_cpu_check_001/allen_structure_graph_1.json'
out = root / 'runs/coarse_region_evidence_079_development_diagnostic'
side = 256
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def labels_at(points):
    voxel = np.floor(points.reshape(-1, 3) / 25).astype(np.int64)
    inside = ((voxel >= 0) & (voxel < np.array(annotation.shape))).all(1)
    raw = np.zeros(len(voxel), np.int64)
    raw[inside] = annotation[tuple(voxel[inside].T)]
    indices = np.searchsorted(annotation_ids, raw)
    return raw.reshape(points.shape[:-1]), annotation_classes[indices].reshape(points.shape[:-1])


def identity_mean(group, metric):
    return float(np.mean([np.mean([row[metric] for row in group
                                    if row['synthetic_subject_plan_id'] == identity])
                          for identity in {row['synthetic_subject_plan_id'] for row in group}]))


training = json.loads((parent_run / 'config.json').read_text())
checkpoint_path = Path(training['parent'])
assert training['parent_sha256'] == sha(checkpoint_path)
records = [row for row in map(json.loads, (panel / 'records.jsonl').open()) if row['eligible']]
baseline = {row['section_id']: row for row in map(json.loads, previous.open())
            if row['set'] == 'synthetic' and row['step'] == 0}
assert len(records) == len(baseline) == 246
assert len({row['synthetic_subject_plan_id'] for row in records}) == 8
support_cutoff = float(np.quantile([row['valid_pixels'] / side**2 for row in records], .25))

ontology = json.loads(ontology_path.read_text())['msg'][0]
coarse_roots = {
    315: 'isocortex', 698: 'olfactory', 1089: 'hippocampal_formation',
    703: 'cortical_subplate', 477: 'striatum', 803: 'pallidum',
    549: 'thalamus', 1097: 'hypothalamus', 313: 'midbrain',
    771: 'pons', 354: 'medulla', 512: 'cerebellum',
    1009: 'fiber_tracts', 73: 'ventricles'}
root_to_class = {region: i + 1 for i, region in enumerate(coarse_roots)}
id_to_class = {0: 0}
stack = [(ontology, 0)]
while stack:
    node, inherited = stack.pop()
    region_class = root_to_class.get(node['id'], inherited)
    id_to_class[node['id']] = region_class
    stack.extend((child, region_class) for child in node['children'])
assert all(region in id_to_class for region in coarse_roots)
annotation = nrrd.read(str(annotation_path), index_order='F')[0]
annotation_ids = np.unique(annotation)
annotation_classes = np.array([id_to_class.get(int(region), 0)
                               for region in annotation_ids], dtype=np.int16)

frozen = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = OneShotJointSliceModel(modes=16, normal_anchor_count=64,
    atlas_conditioning=True, fit_quality=True, vector_refinement=True,
    candidate_ranking=True, fitted_ranking=True).cuda().eval()
model.load_state_dict(frozen['model'], strict=True)
model.requires_grad_(False)
del frozen

out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps({
    'source_sha256': sha(Path(__file__)),
    'protocol_sha256': sha(repo / 'docs/publication/COARSE_REGION_EVIDENCE_079_PROTOCOL_20261004.md'),
    'model_source_sha256': sha(repo / 'training/arbitrary_plane_one_shot_model.py'),
    'frame_source_sha256': sha(repo / 'training/arbitrary_plane_full_frame_primitives.py'),
    'parent_checkpoint': str(checkpoint_path), 'parent_sha256': training['parent_sha256'],
    'parent_config_sha256': sha(parent_run / 'config.json'),
    'synthetic_records_sha256': sha(panel / 'records.jsonl'),
    'synthetic_panel_receipt_sha256': sha(panel / 'completed.json'),
    'previous_evaluation_rows_sha256': sha(previous),
    'annotation_path': str(annotation_path), 'annotation_sha256': sha(annotation_path),
    'ontology_path': str(ontology_path), 'ontology_sha256': sha(ontology_path),
    'annotation_shape_ap_dv_ml': list(annotation.shape),
    'annotation_unassigned_ids': [int(x) for x, c in zip(annotation_ids, annotation_classes)
                                   if c == 0 and x != 0],
    'coarse_roots': coarse_roots, 'pixels_per_section': 1024,
    'low_support_cutoff': support_cutoff,
    'proposal_source': 'frozen 059 blind image-prior top 8 legacy + top 6 normal-anchor branches',
    'oracle_used_for_candidate_generation': False,
    'oracle_used_for_scoring': True, 'calibrated': False, 'public_benchmark_used': False}, indent=2))

rows = []
pair_i, pair_j = np.triu_indices(14, 1)
with torch.inference_mode(), (out / 'rows.jsonl').open('w') as stream:
    for number, record in enumerate(records, 1):
        path = panel / record['file']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            image = torch.from_numpy(arrays['inputs'][None].copy()).cuda()
            target = torch.from_numpy(arrays['target_centre_um'].copy()).cuda()
            valid = torch.from_numpy(arrays['valid_mask'].copy()).cuda().bool()
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
        state = prediction['state'].gather(1, (choice // 2)[..., None].expand(-1, -1, 12))
        centre, frame, basis = full_frame_state_to_components(state)
        xy = chart[None, None].expand(1, 14, 1024, 2).clone()
        xy[..., 0] = torch.where((choice % 2)[..., None].bool(), 255 / side - xy[..., 0], xy[..., 0])
        candidate = centre[:, :, None] + torch.einsum(
            'bkij,bkpj->bkpi', frame[..., :, :2] @ basis, xy - .5)
        errors = (candidate - reference[None, None]).norm(dim=-1).mean(-1)[0].cpu().numpy()
        prior_scores = prior.gather(1, choice)[0].cpu().numpy()
        parent_i = int(np.argmax(prior_scores))
        old = baseline[record['section_id']]
        assert abs(float(errors[parent_i]) - old['parent_selected_um']) < 1
        assert abs(float(errors.min()) - old['parent_best14_oracle_um']) < 1

        target_raw, target_labels = labels_at(reference.cpu().numpy())
        candidate_raw, candidate_labels = labels_at(candidate[0].cpu().numpy())
        present = [region for region in range(1, 15)
                   if np.count_nonzero(target_labels == region) >= 16]
        scores = np.zeros(14, np.float64)
        for region in present:
            target_region = target_labels == region
            candidate_region = candidate_labels == region
            scores += 2 * np.count_nonzero(candidate_region & target_region, axis=1) \
                      / (np.count_nonzero(candidate_region, axis=1)
                         + np.count_nonzero(target_region))
        if present:
            scores /= len(present)
        selected_i = int(np.lexsort((-prior_scores, -scores))[0])
        best_i = int(np.argmin(errors))
        target_foreground = target_labels != 0
        candidate_foreground = candidate_labels != 0
        support_dice = 2 * np.count_nonzero(candidate_foreground & target_foreground, axis=1) \
                       / np.maximum(np.count_nonzero(candidate_foreground, axis=1)
                                    + np.count_nonzero(target_foreground), 1)
        score_diff = scores[pair_i] - scores[pair_j]
        error_diff = errors[pair_i] - errors[pair_j]
        comparable = (score_diff != 0) & (error_diff != 0)
        concordance = float(np.mean((score_diff[comparable] * error_diff[comparable]) < 0)) \
                      if comparable.any() else None
        row = {key: record[key] for key in ('section_id', 'animal_id', 'synthetic_animal_id',
            'specimen_id', 'experiment_id', 'synthetic_subject_plan_id',
            'panel_physical_section_id', 'sha256', 'appearance_mode')}
        row.update({'visible_fraction': record['valid_pixels'] / side**2,
            'low_support': record['valid_pixels'] / side**2 <= support_cutoff,
            'parent_selected_um': float(errors[parent_i]),
            'semantic_selected_um': float(errors[selected_i]),
            'best14_oracle_um': float(errors[best_i]),
            'parent_index': parent_i, 'semantic_index': selected_i, 'best14_index': best_i,
            'chosen_branches': choice[0].cpu().tolist(),
            'candidate_errors_um': errors.tolist(),
            'candidate_prior_log_scores': prior_scores.tolist(),
            'candidate_semantic_scores': scores.tolist(),
            'candidate_support_dice': support_dice.tolist(),
            'candidate_exact_id_agreement': np.mean(candidate_raw == target_raw, axis=1).tolist(),
            'target_present_classes': [coarse_roots[root_id] for root_id in coarse_roots
                                       if root_to_class[root_id] in present],
            'target_unassigned_fraction': float(np.mean(target_labels == 0)),
            'score_error_pairwise_concordance': concordance})
        rows.append(row)
        stream.write(json.dumps(row) + '\n')
        if number % 50 == 0 or number == len(records):
            stream.flush()
            print(json.dumps({'event': 'sections', 'done': number, 'total': len(records)}), flush=True)

metrics = ('parent_selected_um', 'semantic_selected_um', 'best14_oracle_um')
summary = {'sections': len(rows),
    'identities': len({row['synthetic_subject_plan_id'] for row in rows}),
    'identity_equal_mean_um': {metric: identity_mean(rows, metric) for metric in metrics},
    'physically_best_branch_fraction': float(np.mean([
        row['semantic_index'] == row['best14_index'] for row in rows])),
    'score_error_pairwise_concordance_mean': float(np.mean([
        row['score_error_pairwise_concordance'] for row in rows
        if row['score_error_pairwise_concordance'] is not None])),
    'sections_with_no_present_target_class': sum(not row['target_present_classes'] for row in rows),
    'by_identity': {str(identity): {metric: float(np.mean([row[metric] for row in rows
        if row['synthetic_subject_plan_id'] == identity])) for metric in metrics}
        for identity in sorted({row['synthetic_subject_plan_id'] for row in rows})},
    'by_appearance': {name: {metric: identity_mean(subset, metric) for metric in metrics}
        for name in sorted({row['appearance_mode'] for row in rows})
        if (subset := [row for row in rows if row['appearance_mode'] == name])},
    'by_support': {str(low): {metric: identity_mean(subset, metric) for metric in metrics}
        for low in (True, False)
        if (subset := [row for row in rows if row['low_support'] == low])}}
gate = {
    'semantic_selected_le_1800_um': summary['identity_equal_mean_um']['semantic_selected_um'] <= 1800,
    'improvement_ge_500_um': summary['identity_equal_mean_um']['parent_selected_um']
                              - summary['identity_equal_mean_um']['semantic_selected_um'] >= 500,
    'appearance_le_parent_plus_200_um': all(s['semantic_selected_um'] <= s['parent_selected_um'] + 200
                                             for s in summary['by_appearance'].values()),
    'support_le_parent_plus_200_um': all(s['semantic_selected_um'] <= s['parent_selected_um'] + 200
                                          for s in summary['by_support'].values())}
gate['pass'] = all(gate.values())
summary['gate'] = gate
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
(out / 'completed.json').write_text(json.dumps({'rows': len(rows),
    'config_sha256': sha(out / 'config.json'), 'rows_sha256': sha(out / 'rows.jsonl'),
    'summary_sha256': sha(out / 'summary.json'),
    'calibrated': False, 'public_benchmark_used': False}, indent=2))
print(json.dumps({'event': 'complete', 'sections': len(rows), 'gate': gate}), flush=True)
