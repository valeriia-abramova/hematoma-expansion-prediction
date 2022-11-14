import os
import nibabel as nib
import numpy as np
import scipy.ndimage.morphology

# creating dilated masks of basal images from Basal_to_FU1 folder (the ones having the follow up and pre-registered to them)

files_path = '/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_mask/'
results_path = '/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/data/'

ITERATIONS = [10,15,20]


cases_notUse = ['pt038', 'pt170', 'pt094', 'pt103', 'pt127', 'pt054',
            'pt095','pt057','pt089','pt168','pt113','pt043',
            'pt140','pt108','pt106','pt098','pt082','pt105',
            'pt096', 'pt115', 'pt093', 'pt091',] # remove those cases bcz basal and its mask have inconsistent dims in axial slices

cases = [f.path for f in os.scandir(files_path) if f.is_file()]
cases = [case.split('/')[-1].split('.nii.gz')[0] for case in cases]
print(cases)
cases = [id for id in cases if id not in cases_notUse]
print(cases)

for case in cases:
    print(case)
    if not os.path.isdir(results_path + case+'/'):
        os.mkdir(results_path + case+'/')

    basal_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_mask/{}.nii.gz'.format(case))))
    basal_mask = basal_nifti.get_fdata()

    basal_imgMask = (nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz'.format(case)))).get_fdata()> 0).astype(int)

    for iter in ITERATIONS:
        mask_new = scipy.ndimage.morphology.binary_dilation(basal_mask,iterations=iter, mask = basal_imgMask)
        mask_new[basal_imgMask == 0.0] = 0

        nib.Nifti1Image(mask_new, basal_nifti.affine, basal_nifti.header).to_filename(
            os.path.join(os.path.join(results_path, case), str(iter) + '.nii.gz'))




