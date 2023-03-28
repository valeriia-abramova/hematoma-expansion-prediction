import numpy as np
from scipy.ndimage.morphology import binary_dilation
import nibabel as nib
import os

basal_path ='/home/valeria/MIC3/Prediction_stroke_lesion/data/Basal_to_FU1_V8_mask_interp/'
fu_path = '/home/valeria/MIC3/Prediction_stroke_lesion/HematomaTruetaV8/'
newDataset_path = '/home/valeria/MIC3/Prediction_stroke_lesion/data/SyntheticDT_full/' # this path will contain synthetically created follow-up images
DT_path = '/home/valeria/MIC3/Prediction_stroke_lesion/data/DistanceTransformVentricles/'


cases = ['pt072',
'pt080',
'pt167',
'pt132',
'pt078',
'pt062',
'pt123',
'pt119',
'pt124',
'pt009',
'pt019',
'pt067',
'pt032',
'pt164',
'pt042',
'pt086',
'pt011',
'pt093',
'pt045',
'pt003',
'pt055',
'pt002',
'pt149',
'pt129',
'pt092',
'pt145',
'pt110',
'pt001',
'pt083',
'pt039',
'pt113',
'pt010',
'pt133',
'pt122',
'pt013',
'pt168',
'pt017',
'pt031',
'pt169',
'pt022',
'pt097',
'pt165',
'pt012',
'pt023',
'pt073',
'pt141',
'pt099',
'pt066',
'pt015',
'pt102',
'pt068',
'pt216',
'pt220',
'pt205',
'pt212',
'pt209',
'pt218',
'pt210',
'pt186',
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
'pt038'
]


growth_rates = ['33','68','100']

cases = ['pt099']
for case in cases:
    for rate in growth_rates:
        if not os.path.isdir(newDataset_path + case + '_' + rate):
                os.mkdir(newDataset_path + case + '_' + rate)

        print(case, ' ', rate)

        folder = '/FU1'
        
        fu_mask_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join(fu_path,case+folder+'/hematoma_mask_vicorob_reviewed_reoriented.nii.gz')))
        fu_mask = fu_mask_nifti.get_fdata()

        fu_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join(fu_path,case+folder+'/CT_SS.nii.gz')))
        fu = fu_nifti.get_fdata()

        basal_mask_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join(basal_path,case + '.nii.gz')))
        basal_mask = basal_mask_nifti.get_fdata()
        basal_mask = np.round(basal_mask)

        distance_transform = nib.funcs.as_closest_canonical(nib.load(os.path.join(DT_path,case + '_dist.nii.gz'))).get_fdata()
        distance_transform[distance_transform == 0.0] = 1.0
        # distance_transform[fu > 0.0] = 1.0

        growth = (float(rate)/100+1)
        vol = np.count_nonzero(basal_mask)
        print('init vol= ', vol)

        target_vol = growth*vol

        i = 0
        j = 0

        thresh = 0.5


        while vol < target_vol:

            j+=1

            synthMask = binary_dilation(fu_mask, iterations=1)

            synthMask_prob = synthMask * distance_transform
            synthMask = np.where(synthMask_prob > thresh, 1, 0)

            if not np.any(synthMask_prob[synthMask_prob < 1.0] > thresh) or j > 5:
                thresh = thresh - 0.05
                j = 0
                print(thresh)

            synthMask[fu == 0.0] = 0.0

            vol = np.count_nonzero(synthMask)


            fu_mask = synthMask
            
            i+=1
        
        print('i= ',i, ' vol= ', vol)

        

        nib.Nifti1Image(synthMask, fu_mask_nifti.affine, fu_mask_nifti.header).to_filename(
            os.path.join(newDataset_path + case + '_' + rate, 'mask.nii.gz'))