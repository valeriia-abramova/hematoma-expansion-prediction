import nibabel as nib
import numpy as np
import scipy.ndimage
import os

path = '/home/valeria/Prediction_stroke_lesion/data/Synthetic/'
distance_transform_path = '/home/valeria/Prediction_stroke_lesion/data/DistanceTransformVentricles/'

cases = [f.path for f in os.scandir(path) if f.is_dir()]
cases = [case.split('/')[-1] for case in cases]


patients = [case.split('_')[0] for case in cases]


patients = list(set(patients))

patients = ['pt186',
'pt135',
'pt103',
'pt190',
'pt014',
'pt202',
'pt059',
'pt091',
'pt104',
'pt037',
'pt084',
'pt199',
'pt203',
'pt196',
'pt177',
'pt158',
'pt065',
'pt181',
'pt111',
'pt077',
'pt024',
'pt038']

for case in patients:

    print(case)
    folder = '/FU1'

    baseline_mask = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/HematomaTruetaV8/'+ case+ folder +'/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'))).get_fdata()
    baseline_mask = np.round(baseline_mask)
    baseline_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_V8/',case+'.nii.gz')))
    baselineBrain = baseline_nifti.get_fdata()
    baselineBrain_mask = (baselineBrain > 0.0).astype(int)
    baselineBrain_mask = scipy.ndimage.binary_dilation(baselineBrain_mask,iterations=2)
    eroded = scipy.ndimage.binary_erosion(baselineBrain_mask,iterations=4)

    ventricles = baselineBrain

    ventricles[baselineBrain == 0.0] = 19.0
    ventricles[baselineBrain <= 15.0] = 1.0
    ventricles[baselineBrain > 15.0] = 0.0
    ventricles[eroded == 0.0] = 0.0
    # ventricles = scipy.ndimage.binary_erosion(ventricles, iterations=2)
    # baselineBrain_mask = scipy.ndimage.binary_fill_holes(baselineBrain_mask)
    
    ventricles = scipy.ndimage.binary_opening(ventricles,iterations=3)
    img_labelled, nlabels = scipy.ndimage.label(ventricles)
    label_list = np.arange(1, nlabels + 1) # and 1
    label_volumes = scipy.ndimage.labeled_comprehension(ventricles, img_labelled, label_list, np.sum, float, 0)

    biggest_label = {'idx': None, 'volume': 0}
    for n, label_volume in enumerate(label_volumes):
        if label_volume > biggest_label['volume']:
            biggest_label = {'idx': n + 1, 'volume': label_volume}

    baselineBrain_mask[img_labelled == biggest_label['idx']] = 0.0
    


    # image = baselineBrain_mask - baseline_mask
    # inv_image = np.ones(np.shape(baseline_mask)) - baseline_mask

    dist = scipy.ndimage.distance_transform_edt(baselineBrain_mask)
    dist[baseline_mask == 1.0] = 0.0

    normalized = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))

    nib.Nifti1Image(normalized, baseline_nifti.affine, baseline_nifti.header).to_filename(
                os.path.join(os.path.join(distance_transform_path, case) + '_dist.nii.gz'))

    # nib.Nifti1Image(inv_image, baseline_nifti.affine, baseline_nifti.header).to_filename(
    #             os.path.join(os.path.join(distance_transform_path, case) + '_img.nii.gz'))
