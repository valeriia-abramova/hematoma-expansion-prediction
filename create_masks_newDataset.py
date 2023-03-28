import numpy as np
from scipy.ndimage.morphology import binary_dilation
import nibabel as nib
import os

basal_path ='/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_V8_mask_interp/'
fu_path = '/home/valeria/Prediction_stroke_lesion/HematomaTruetaV8/'
newDataset_path = '/home/valeria/Prediction_stroke_lesion/data/Synthetic_full/' # this path will contain synthetically created follow-up images

# cases = [
#     'pt023', 'pt062', 'pt165', 'pt097', 'pt061', 'pt167', 'pt042', 'pt009', 'pt073', 'pt066',
#     'pt031', 'pt002', 'pt013', 'pt017', 'pt169', 'pt039', 'pt040', 'pt049', 'pt071', 'pt132',
#     'pt110', 'pt141', 'pt001', 'pt123', 'pt124', 'pt055', 'pt067', 'pt093', 'pt133', 'pt086',
#     'pt102', 'pt078', 'pt129', 'pt083', 'pt168', 'pt149', 'pt119', 'pt113', 'pt145', 'pt015',
#     'pt003', 'pt018', 'pt034', 'pt032', 'pt010', 'pt072', 'pt019', 'pt092', 'pt045', 'pt011',
#     'pt022', 'pt163', 'pt122', 'pt080', 'pt012', 'pt068', 'pt164', 'pt099']

# cases = [
#     'pt061', 'pt042', 
#     'pt031', 'pt013', 'pt017', 'pt071', 'pt141', 'pt078', 'pt032', 'pt045', 'pt011',
#     'pt122', 'pt012',  'pt164', 'pt049','pt113']

# cases = ['pt113']

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

for case in cases:
    for rate in growth_rates:
        if not os.path.isdir(newDataset_path + case + '_' + rate):
                os.mkdir(newDataset_path + case + '_' + rate)

        print(case, ' ', rate)
        
        fu_mask_nifti = nib.load(os.path.join(fu_path,case,'FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'))
        fu_mask = fu_mask_nifti.get_fdata()

        fu_nifti = nib.load(os.path.join(fu_path,case,'FU1/CT_SS.nii.gz'))
        fu = fu_nifti.get_fdata()

        basal_mask_nifti = nib.load(os.path.join(basal_path,case + '.nii.gz'))
        basal_mask = basal_mask_nifti.get_fdata()
        basal_mask = np.round(basal_mask)

        growth = (float(rate)/100+1)
        vol = np.count_nonzero(basal_mask)
        print('init vol= ', vol)

        target_vol = growth*vol

        i = 0


        # rng = np.random.default_rng()
        # struct = rng.integers(low=0, high= 1,size=(3,3,3), endpoint=True)

        while vol < target_vol:

            # synthMask = binary_dilation(fu_mask, iterations=1, structure=struct)
            synthMask = binary_dilation(fu_mask, iterations=1)
            synthMask[fu == 0.0] = 0.0

            vol = np.count_nonzero(synthMask)
            fu_mask = synthMask
            
            i+=1
        
        print('i= ',i, ' vol= ', vol)

        

        nib.Nifti1Image(synthMask, fu_mask_nifti.affine, fu_mask_nifti.header).to_filename(
            os.path.join(newDataset_path + case + '_' + rate, 'mask.nii.gz'))