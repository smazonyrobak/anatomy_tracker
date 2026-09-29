import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TEMP'] = r'I:\AnatomyTracker\tmp'
os.environ['TMP'] = r'I:\AnatomyTracker\tmp'
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch

root = Path(r'I:\AnatomyTracker\runs')
run = root/'joint_v6_proposal_curriculum_003'
prepared = root/'joint_v6_proposal_substantive_001'
cache = root/'arbitrary_plane_finite_v6_substantive_data_001'/'internal_development_cache'
dev = torch.load(prepared/'internal_development_prepared.pt',map_location='cpu',weights_only=False)
catalogue = torch.load(run/'catalogue.pt',map_location='cpu',weights_only=False)
rows = np.load(run/'development_rows_step_20000.npz')
prediction = rows['prediction']
states = np.concatenate((dev['truth_state'].numpy(),catalogue['arrays']['cell_states_float64'][prediction]))

# Independent NumPy state -> physical O/U/V -> cross product; no geometry helpers.
first,second = states[:,3:6],states[:,6:9]
ex = first/np.linalg.norm(first,axis=1,keepdims=True)
ey = second-(second*ex).sum(1,keepdims=True)*ex
ey /= np.linalg.norm(ey,axis=1,keepdims=True)
U = ex*np.exp(states[:,9,None])
V = (ex*states[:,11,None]+ey)*np.exp(states[:,10,None])
O = states[:,:3]-(U+V)/2
normal = np.cross(U,V)
normal /= np.linalg.norm(normal,axis=1,keepdims=True)
truth,predicted = normal[:640],normal[640:]
dot = np.clip(np.einsum('ij,ij->i',truth,predicted),-1,1)
signed_radians = np.arccos(dot)
projective_radians = np.arccos(np.abs(dot))
angle = projective_radians*180/np.pi
cross_angle = np.arctan2(np.linalg.norm(np.cross(truth,predicted),axis=1),np.abs(dot))*180/np.pi

# Tiny JSON geometry only: an independent second route to all saved truth normals.
manifest = json.loads((cache/'manifest.json').read_text())
quicknii = []
matching_rows = True
for record,identity in zip(manifest['rows'],dev['records']):
    meta = json.loads((cache/record['metadata_relative_path']).read_text())
    quicknii.append(meta['canonical_effective_quicknii_ouv_float64'])
    matching_rows &= meta['training_row_id']==identity['training_row_id'] and meta['receipt_sha256']==identity['row_receipt_sha256']
quicknii = np.asarray(quicknii)
quick_u = quicknii[:,1][:,[1,2,0]]*np.array([-25.,-25.,25.])
quick_v = quicknii[:,2][:,[1,2,0]]*np.array([-25.,-25.,25.])
quick_normal = np.cross(quick_u,quick_v)
quick_normal /= np.linalg.norm(quick_normal,axis=1,keepdims=True)
quick_angle = np.arccos(np.abs(np.einsum('ij,ij->i',quick_normal,predicted)).clip(0,1))*180/np.pi

conventions = []
shared_max_error = 0.
for permutation in itertools.permutations(range(3)):
    for signs in itertools.product((-1,1),repeat=3):
        transformed_prediction = predicted[:,permutation]*signs
        transformed_truth = truth[:,permutation]*signs
        fixed_angle = np.arccos(np.abs(np.einsum('ij,ij->i',truth,transformed_prediction)).clip(0,1))*180/np.pi
        shared_angle = np.arccos(np.abs(np.einsum('ij,ij->i',transformed_truth,transformed_prediction)).clip(0,1))*180/np.pi
        shared_max_error = max(shared_max_error,float(np.abs(shared_angle-angle).max()))
        conventions.append({'permutation':list(permutation),'signs':list(signs),'mean_deg':float(fixed_angle.mean()),'median_deg':float(np.median(fixed_angle)),'under5_count':int((fixed_angle<5).sum()),'under10_count':int((fixed_angle<10).sum())})
conventions.sort(key=lambda entry:entry['mean_deg'])
rng = np.random.default_rng(20260929)
shuffle_mean = []
for _ in range(100):
    shuffled = predicted[rng.permutation(640)]
    shuffle_mean.append(float(np.arccos(np.abs(np.einsum('ij,ij->i',truth,shuffled)).clip(0,1)).mean()*180/np.pi))
group = np.array([record['animal_id'] for record in dev['records']])
result = {
    'scope':'Targeted CPU NumPy OUV/cross-product convention audit of frozen003; no model inference, no GPU, no active005 access',
    'rows':len(prediction), 'helpers_used':'None: full_frame_state_to_components and all geometry helpers deliberately not imported',
    'independent_group_macro_normal_error_deg':float(np.mean([angle[group==key].mean() for key in np.unique(group)])),
    'independent_mean_normal_error_rad':float(projective_radians.mean()),
    'signed_direction_mean_error_deg':float(signed_radians.mean()*180/np.pi),
    'signed_angle_folded_to_plane_max_difference_rad':float(np.abs(np.minimum(signed_radians,np.pi-signed_radians)-projective_radians).max()),
    'acos_vs_independent_atan2_max_difference_deg':float(np.abs(angle-cross_angle).max()),
    'maximum_error_from_saved_per_row_metric_deg':float(np.abs(angle-rows['plane_angle_deg']).max()),
    'maximum_error_using_direct_quicknii_vectors_deg':float(np.abs(quick_angle-angle).max()),
    'quicknii_geometry_row_receipts_match':bool(matching_rows),
    'quicknii_normal_vs_state_normal_max_vector_difference':float(np.abs(quick_normal-truth).max()),
    'predicted_ouv_normal_vs_stored_catalogue_normal_max_vector_difference':float(np.abs(predicted-catalogue['arrays']['cell_normal_ap_dv_ml_float64'][prediction]).max()),
    'unit_normal_max_norm_error':float(np.abs(np.linalg.norm(normal,axis=1)-1).max()),
    'all_48_shared_signed_axis_conventions_max_angle_difference_deg':shared_max_error,
    'one_sided_fixed_axis_conventions':conventions,
    'permutation_search_scope':'A single fixed convention applied to every predicted normal; diagnostic only, never used to replace endpoints; 48 includes antipodally redundant transforms',
    'row_shuffle_100_mean_deg':float(np.mean(shuffle_mean)),
    'row_shuffle_100_mean_range_deg':[float(min(shuffle_mean)),float(max(shuffle_mean))],
    'row_shift_1_mean_deg':float(np.arccos(np.abs(np.einsum('ij,ij->i',truth,np.roll(predicted,1,axis=0))).clip(0,1)).mean()*180/np.pi),
    'conclusion':'No angle-unit, antipodal folding, cross-product, constant axis/sign convention, or saved-truth conversion discrepancy found. This does not prove the whole learning pipeline is bug-free.',
    'next_discriminating_experiment':'A fixed tiny eight-identifiable-case full encoder+98304-cell proposal-head overfit, preserving actual images, labels, objective and corrected FP32 head; first establish memorization/quantization-limited geometry before another large run.',
}
for name,path in (('final_rows',run/'development_rows_step_20000.npz'),('dev_prepared',prepared/'internal_development_prepared.pt'),('catalogue',run/'catalogue.pt'),('manifest',cache/'manifest.json')):
    with path.open('rb') as stream:
        result[f'{name}_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
Path(r'I:\AnatomyTracker\tmp\proposal_frame_metric_audit_20260929.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({key:value for key,value in result.items() if key!='one_sided_fixed_axis_conventions'},indent=2))
print('best_fixed_conventions',json.dumps(conventions[:4],indent=2))
assert matching_rows and result['maximum_error_from_saved_per_row_metric_deg']<1e-8 and result['maximum_error_using_direct_quicknii_vectors_deg']<1e-8
