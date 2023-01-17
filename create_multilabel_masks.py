import nibabel as nib
import os
import numpy as np

path = '/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/019-patchBalanced80-ppi500-adam00001-bs32-l1loss-ps323232-border-mask-1000epochs/'

cases = ['pt038', 'pt170', 'pt094', 'pt103', 'pt127', 'pt054', 'pt171',
        'pt095','pt057','pt077','pt043', 'pt005',
        'pt140','pt108','pt098','pt082','pt105']


for case in cases:
    print(case)

    basal_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz'.format(case))))

    basal_lesionMask_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_mask_interp/{}.nii.gz'.format(case))))
    basal_lesionMask = basal_lesionMask_nifti.get_fdata()
    fu1_real_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/HematomaTruetaV7/{}/FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'.format(case)))).get_fdata()
    basal_lesionMask = np.round(basal_lesionMask)
    fu1_real_lesionMask = np.round(fu1_real_lesionMask)
    fu1_pred_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,'results/',case+'_mask.nii.gz'))).get_fdata()

    multilabel = np.zeros(np.shape(basal_lesionMask))

    diff = fu1_real_lesionMask-fu1_pred_lesionMask

    multilabel[fu1_real_lesionMask == 1.0] = 1.0
    multilabel[diff == 1.0] = 2.0 # FN
    multilabel[diff == -1.0] = 3.0 # FP

    nib.Nifti1Image(multilabel, basal_nifti.affine, basal_nifti.header).to_filename(os.path.join(path,'multilabel_masks/',case+'.nii.gz'))