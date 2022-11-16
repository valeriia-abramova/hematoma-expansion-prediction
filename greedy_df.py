import numpy as np
import nibabel as nib
import os
import sys
import subprocess

dir_path = "/home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/"
init_path = '/home/valeria/MIC3/Prediction_stroke_lesion/data/Basal_to_FU1_mask/'
image_path = '/home/valeria/MIC3/Prediction_stroke_lesion/data/Basal_to_FU1/'


# all patient ids

cases = [
    'pt018', 'pt100', 'pt169', 'pt153', 'pt013', 'pt067', 'pt171', 'pt109', 'pt068', 'pt063', 'pt009', 'pt161', 'pt114', 'pt087', 
    'pt133', 
    'pt134', 'pt167', 'pt090', 'pt130', 'pt101', 'pt119', 'pt120', 'pt135', 'pt042', 'pt045', 'pt028', 'pt159', 'pt121', 'pt001', 'pt071', 'pt150', 
    'pt010', 'pt070', 'pt077', 'pt085', 'pt016', 'pt023', 'pt128', 
    'pt080', 'pt079', 'pt151', 
    'pt007', 'pt086', 'pt011', 'pt138', 'pt065', 'pt102', 
    'pt116', 'pt084', 'pt124', 'pt025', 'pt051', 'pt021', 'pt003', 'pt047', 'pt131', 'pt049', 'pt015', 'pt032', 'pt141', 'pt136', 'pt041', 'pt099', 
    'pt111', 'pt137', 'pt069', 'pt081', 'pt062', 'pt060', 'pt110', 'pt027', 'pt092', 'pt118', 'pt020', 'pt022', 'pt143', 'pt139', 'pt145', 'pt166', 
    'pt031', 'pt165', 'pt064', 'pt037', 'pt017', 'pt078', 'pt019', 'pt142', 'pt033', 'pt122', 'pt088', 'pt083', 'pt012', 'pt030', 'pt061', 
    'pt152', 'pt163', 'pt048', 'pt005', 'pt044', 'pt072', 'pt076', 'pt075', 'pt039', 'pt164', 'pt132', 'pt125', 'pt073', 
    'pt066', 'pt149', 'pt002', 'pt040', 'pt052', 'pt034', 'pt129', 'pt055', 'pt123', 'pt097', 'pt058', 'pt144']

# subprocess.run(['cd greedy/build'])

for case in cases:
        print(case, ' ', iter)

        command = \
            './greedy -d 3 -m NCC 2x2x2 -i /home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/doublemask.nii.gz /home/valeria/MIC3/Prediction_stroke_lesion/data/Basal_to_FU1_mask/{}.nii.gz -o /home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/doublemask_df.nii.gz -n 100x50x10'.format(case, case, case)
        
        subprocess.run(['bash','-c',command])

        df_nifti = nib.funcs.as_closest_canonical(nib.load('/home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/doublemask_df.nii.gz'.format(case)))
        df = df_nifti.get_fdata()
        mask_init = nib.funcs.as_closest_canonical(nib.load('/home/valeria/MIC3/Prediction_stroke_lesion/data/Basal_to_FU1_mask/{}.nii.gz'.format(case))).get_fdata()

        print(df.dtype)
        print((mask_init.dtype))
        df[:,:,:,0,0][mask_init > 0.0] = 0.0
        df[:,:,:,0,1][mask_init > 0.0] = 0.0
        df[:,:,:,0,2][mask_init > 0.0] = 0.0

        nib.Nifti1Image(df, df_nifti.affine, df_nifti.header).to_filename('/home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/df_part.nii.gz'.format(case)) 

        command2 = \
            './greedy -d 3 -rf /home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/doublemask.nii.gz -rm /home/valeria/MIC3/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz /home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/doublemask_img.nii.gz -ri LABEL 0.2vox -r /home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/doublemask_df.nii.gz'.format(case,case,case,case)
        command3 = \
            './greedy -d 3 -rf /home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/doublemask.nii.gz -rm /home/valeria/MIC3/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz /home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/dfpart_img.nii.gz -ri LABEL 0.2vox -r /home/valeria/MIC3/Prediction_stroke_lesion/SynthesisGrowth/data/{}/df_part.nii.gz'.format(case,case,case,case)

        subprocess.run(['bash','-c',command2])

        subprocess.run(['bash','-c',command3])
