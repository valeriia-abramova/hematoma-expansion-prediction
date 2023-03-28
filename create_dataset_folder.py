import os
import nibabel as nib
import shutil

path = '/home/valeria/Prediction_stroke_lesion/HematomaTruetaV8/'

new_path = '/home/valeria/Prediction_stroke_lesion/HematomaTrueta_BasalFU/'

basal_path = '/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_V8/'
basal_mask_path = '/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_V8_mask_interp/'

cases = [f.path for f in os.scandir(path) if f.is_dir()]
cases = [case.split('/')[-1] for case in cases]
cases = [case for case in cases if os.path.isfile(path + case + '/FU1/CT_NC.nii.gz') and os.path.isfile(path + case + '/Basal/CT_NC.nii.gz')]

for case in cases:

    print(case)

    if not os.path.isdir(new_path):
                os.mkdir(new_path)

    if not os.path.isdir(new_path + case):
                os.mkdir(new_path + case)

    if not os.path.isdir(new_path + case + '/Basal/'):
                os.mkdir(new_path + case + '/Basal/')

    if not os.path.isdir(new_path + case + '/FU1/'):
                os.mkdir(new_path + case + '/FU1/')
    
    shutil.copy2(src=os.path.join(basal_path, case + '.nii.gz'), dst=os.path.join(new_path,case,'Basal/CT_SS.nii.gz'))
    shutil.copy2(src=os.path.join(basal_mask_path, case + '.nii.gz'), dst=os.path.join(new_path,case,'Basal/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'))
    shutil.copy2(src=os.path.join(path, case, 'FU1/CT_SS.nii.gz'), dst=os.path.join(new_path,case,'FU1/CT_SS.nii.gz'))
    shutil.copy2(src=os.path.join(path, case, 'FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'), dst=os.path.join(new_path,case,'FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'))
