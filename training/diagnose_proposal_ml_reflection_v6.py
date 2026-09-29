import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TEMP'] = r'I:\AnatomyTracker\tmp'
os.environ['TMP'] = r'I:\AnatomyTracker\tmp'
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from training.arbitrary_plane_full_frame_primitives import full_frame_state_to_components

torch.set_num_threads(4)
root = Path(r'I:\AnatomyTracker\runs')
prepared = root/'joint_v6_proposal_substantive_001'
dev = torch.load(prepared/'internal_development_prepared.pt',map_location='cpu',weights_only=False)
catalogue = torch.load(prepared/'catalogue.pt',map_location='cpu',weights_only=False)
normals = catalogue['arrays']['cell_normal_ap_dv_ml_float64']
normal_grid = normals[::256]
assert len(normal_grid)==384 and np.array_equal(normals,np.repeat(normal_grid,256,axis=0))
_,truth_frames,_ = full_frame_state_to_components(dev['truth_state'].double())
truth = truth_frames[:,:,2].numpy()
cosine = np.abs(truth@normal_grid.T).clip(0,1)
reflected_cosine = np.abs(truth@(normal_grid*np.array([1.,1.,-1.])).T).clip(0,1)
angle = np.rad2deg(np.arccos(cosine))
symmetry_angle = np.rad2deg(np.arccos(np.maximum(cosine,reflected_cosine)))
group = np.array([row['animal_id'] for row in dev['records']])
modes = np.array([row['selected_mode'] for row in dev['records']])
all_rows = np.arange(len(truth))
summary = {
    'scope':'Normal-only post-hoc ML-reflection diagnostic; original endpoints unchanged; mirror is not interchangeable for hemisphere, anatomical-site localization or surgery',
    'reflection_ap_dv_ml':[[1,0,0],[0,1,0],[0,0,-1]],
    'angular_formula':'degrees(acos(max(abs(n_truth dot n_pred),abs(n_truth dot S_ML n_pred))))',
    'top32_scope':'Truth-selected best normal among32 highest-probability cells; optimistic normal-only candidate capture, not achieved registration',
    'probability_scope':'Raw uncalibrated full-catalogue probability marginal over offset/roll; union cones avoid double counting; no coordinate/hemisphere correctness claim',
    'offsets_checked':False,
    'catalogue_support_origin_ap_dv_ml_um':catalogue['support_geometry']['support_origin_ap_dv_ml_um'],
    'results':{},
}
for name, step in (('joint_v6_proposal_curriculum_003',20000),('joint_v6_proposal_precision_recovery_004',10000)):
    run = root/name
    raw_path = run/f'development_log_probability_step_{step:05d}.npy'
    raw = np.load(raw_path,mmap_mode='r')
    normal_mass = np.empty((640,384),dtype=np.float64)
    top32 = np.empty((640,32),dtype=np.int64)
    prediction = raw.argmax(1)
    for start in range(0,len(truth),32):
        block = np.asarray(raw[start:start+32],dtype=np.float64)
        normal_mass[start:start+len(block)] = np.exp(block).reshape(len(block),384,256).sum(2)
    for row in all_rows:
        cutoff = np.partition(raw[row],-32)[-32]
        chosen = np.flatnonzero(raw[row]>cutoff)
        chosen = np.concatenate((chosen,np.flatnonzero(raw[row]==cutoff)[:32-len(chosen)]))
        top32[row] = chosen[np.lexsort((chosen,-raw[row,chosen]))]
    metrics = {
        'map_absolute_angle_deg':angle[all_rows,prediction//256],
        'map_reflection_minimum_angle_deg':symmetry_angle[all_rows,prediction//256],
        'top32_best_absolute_angle_deg':angle[all_rows[:,None],top32//256].min(1),
        'top32_best_reflection_minimum_angle_deg':symmetry_angle[all_rows[:,None],top32//256].min(1),
        'top32_unique_normal_count':np.array([len(np.unique(indices//256)) for indices in top32]),
    }
    for radius in (5,10,15):
        true_cone = angle<radius
        reflected_cone = np.rad2deg(np.arccos(reflected_cosine))<radius
        for label,mask in (('true',true_cone),('reflected',reflected_cone),('union',true_cone|reflected_cone)):
            metrics[f'normal_mass_within_{radius}deg_{label}'] = (normal_mass*mask).sum(1)
    rescue = {}
    for kind in ('map','top32_best'):
        original = metrics[f'{kind}_absolute_angle_deg']
        reflected = metrics[f'{kind}_reflection_minimum_angle_deg']
        eligible = original>30
        rescued = eligible&(reflected<10)
        rescue[kind] = {
            'original_over30_count':int(eligible.sum()),'over30_to_under10_count':int(rescued.sum()),
            'fraction_of_original_over30':float(rescued.sum()/eligible.sum()),
            'fraction_of_all_rows':float(rescued.mean()),'row_indices':np.flatnonzero(rescued).tolist(),
            'original_under10_count':int((original<10).sum()),'reflection_minimum_under10_count':int((reflected<10).sum()),
        }
    subsets = {'all':np.ones(640,dtype=bool),'identifiable':dev['weight'].numpy()>0,**{mode:modes==mode for mode in np.unique(modes)}}
    aggregates = {subset:{'row_count':int(selected.sum()),'synthetic_group_macro':{metric:float(np.mean([values[selected&(group==key)].mean() for key in np.unique(group[selected])])) for metric,values in metrics.items()}} for subset,selected in subsets.items()}
    # Reflection-minimum error improves even unrelated predictions; show that null.
    rng = np.random.default_rng(20260929)
    null = []
    for _ in range(100):
        permutation = rng.permutation(len(truth))
        null.append([
            angle[all_rows,prediction[permutation]//256].mean(),
            symmetry_angle[all_rows,prediction[permutation]//256].mean(),
            angle[all_rows[:,None],top32[permutation]//256].min(1).mean(),
            symmetry_angle[all_rows[:,None],top32[permutation]//256].min(1).mean(),
        ])
    null = np.asarray(null)
    null_record = {'scope':'100 deterministic cross-section shuffles of the same predictions; descriptive geometric control, not a biological significance test', 'mean':dict(zip(('map_absolute_angle_deg','map_reflection_minimum_angle_deg','top32_best_absolute_angle_deg','top32_best_reflection_minimum_angle_deg'),null.mean(0).tolist()))}
    with raw_path.open('rb') as stream:
        raw_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    record = {'step':step,'raw_sha256':raw_sha,'aggregates':aggregates,'severe_error_rescue':rescue,'prediction_shuffle_control':null_record,'maximum_probability_mass_normalization_error':float(np.abs(normal_mass.sum(1)-1).max())}
    summary['results'][name] = record
    np.savez(Path(r'I:\AnatomyTracker\tmp')/f'{name}_ml_reflection_diagnostic_rows.npz',prediction=prediction,top32=top32,normal_mass=normal_mass,**metrics)
    print(json.dumps({'run':name,'macro':aggregates['all']['synthetic_group_macro'],'rescue':{key:{field:value for field,value in values.items() if field!='row_indices'} for key,values in rescue.items()},'shuffle':null_record},indent=2),flush=True)
Path(r'I:\AnatomyTracker\tmp\proposal_ml_reflection_diagnostic_20260929.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
