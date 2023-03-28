import numpy as np
import nibabel as nib
# from skimage.metrics import structural_similarity as ssim
import os
import csv
# from scipy.ndimage.morphology import binary_dilation



prepared_data_path = '/home/valeria/MIC3/Prediction_stroke_lesion/data/Synthetic_full/'

cases = [f.path for f in os.scandir(prepared_data_path) if f.is_dir()]
cases = [case.split('/')[-1] for case in cases]
cases = sorted(cases)

all_metrics = []

# cases = ['pt034','pt061','pt163','pt071','pt005','pt054',
# 'pt197','pt040','pt049','pt029','pt094','pt108',
# 'pt183','pt155','pt043','pt127','pt173','pt157','pt170',
# 'pt207','pt208','pt179','pt018','pt056','pt098',
# 'pt171','pt095','pt082','pt057','pt126','pt191',]
for case in cases:
    metrics = {}
    print(case)
    patient = case.split('_')[0]

    # real_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join(real_path,case,'original_fu.nii.gz')))
    # real_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join(real_path,case,'FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'))) # for only original_fu case

    # real = real_nifti.get_fdata()
    # fake = nib.funcs.as_closest_canonical(nib.load(os.path.join(fake_path,case,'original_fu_avg_mask.nii.gz'))).get_fdata()

    # difference = real - fake

    basal_lesionMask_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/MIC3/Prediction_stroke_lesion/data/Basal_to_FU1_V8_mask_interp/{}.nii.gz'.format(patient))))
    basal_lesionMask = basal_lesionMask_nifti.get_fdata()
    fu1_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join(prepared_data_path,case,'mask.nii.gz'))).get_fdata()
    basal_lesionMask = np.round(basal_lesionMask)
    fu1_lesionMask = np.round(fu1_lesionMask)
    borderMask = fu1_lesionMask - basal_lesionMask
    borderMask[borderMask == -1.0] = 0.0

    # predicted_mask = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/034-patchBalanced80-ppi500-adam00001-bs32-l1loss-ps323232-border-mask-1000epochs-TrainOn14FromTest+50Synth/results/{}_mask.nii.gz'.format(case)))).get_fdata()
    # diff = predicted_mask - basal_lesionMask

    nib.Nifti1Image(borderMask, basal_lesionMask_nifti.affine, basal_lesionMask_nifti.header).to_filename(os.path.join(prepared_data_path,case,'diff.nii.gz')) 

