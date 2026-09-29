import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TEMP'] = r'I:\AnatomyTracker\tmp'
os.environ['TMP'] = r'I:\AnatomyTracker\tmp'
import json
from pathlib import Path
import numpy as np
from training import arbitrary_plane_allen_atlas_binding_v6 as allen

# Exact pinned raw-NRRD decode/preprocessing; no model/checkpoint/GPU access.
atlas,annotation = allen._decode_and_preprocess_allen_v6()
result = {
    'scope':'Numerical voxel-grid symmetry of the pinned Allen template/support only; not bilateral equivalence of real brains or a license to reverse hemisphere labels',
    'shape_ap_dv_ml':list(annotation.shape), 'origin_ap_dv_ml_um':list(allen.ATLAS_ORIGIN_AP_DV_ML_UM_V6),
    'voxel_size_ap_dv_ml_um':list(allen.ATLAS_VOXEL_SIZE_AP_DV_ML_UM_V6),
    'coordinate_convention':'Physical voxel center = origin + (array_index + 0.5) * spacing',
    'intensity_receipt':allen.INTENSITY_FLOAT32_RECEIPT_V6,
    'annotation_raw_sha256':allen.ANNOTATION_RAW_SHA256_V6,
    'template_raw_sha256':allen.TEMPLATE_RAW_SHA256_V6,
    'comparisons':{},
}
for name,midline in (('voxel_center_extent_midpoint_and_catalogue_support_origin_ML',5700.0),('half_voxel_shifted_ML',5712.5)):
    sums = dict(absolute=0.,squared=0.,union_absolute=0.,union_squared=0.,union_count=0,intersection_count=0,support_total=0,label_equal_union=0,max_difference=0.)
    for start in range(0,annotation.shape[0],32):
        image = atlas[0,start:start+32]
        labels = annotation[start:start+32]
        if name=='voxel_center_extent_midpoint_and_catalogue_support_origin_ML':
            reflected_image = image[:,:,::-1]
            reflected_labels = labels[:,:,::-1]
        else:
            # x' = 456-x: column0 reflects outside this atlas array and is zero-filled.
            reflected_image = np.zeros_like(image)
            reflected_labels = np.zeros_like(labels)
            reflected_image[:,:,1:] = image[:,:,:0:-1]
            reflected_labels[:,:,1:] = labels[:,:,:0:-1]
        support = labels!=0
        reflected_support = reflected_labels!=0
        union = support|reflected_support
        difference = np.abs(image-reflected_image)
        sums['absolute'] += difference.sum(dtype=np.float64)
        sums['squared'] += np.square(difference).sum(dtype=np.float64)
        sums['union_absolute'] += difference[union].sum(dtype=np.float64)
        sums['union_squared'] += np.square(difference[union]).sum(dtype=np.float64)
        sums['union_count'] += int(union.sum())
        sums['intersection_count'] += int((support&reflected_support).sum())
        sums['support_total'] += int(support.sum()+reflected_support.sum())
        sums['label_equal_union'] += int(((labels==reflected_labels)&union).sum())
        sums['max_difference'] = max(sums['max_difference'],float(difference.max()))
    result['comparisons'][name] = {
        'midline_ML_um':midline,
        'intensity_MAE_whole_array':sums['absolute']/annotation.size,
        'intensity_RMSE_whole_array':float(np.sqrt(sums['squared']/annotation.size)),
        'intensity_MAE_support_union':sums['union_absolute']/sums['union_count'],
        'intensity_RMSE_support_union':float(np.sqrt(sums['union_squared']/sums['union_count'])),
        'intensity_max_absolute_difference':sums['max_difference'],
        'support_dice':2*sums['intersection_count']/sums['support_total'],
        'support_disagreement_fraction_union':1-sums['intersection_count']/sums['union_count'],
        'annotation_label_agreement_fraction_union':sums['label_equal_union']/sums['union_count'],
        'annotation_semantics':'Same integer region labels may occur in both hemispheres; label agreement does not preserve laterality',
    }
Path(r'I:\AnatomyTracker\tmp\allen_ml_symmetry_diagnostic_20260929.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
