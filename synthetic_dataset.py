import numpy as np
import nibabel as nib
import os
import subprocess

basal_path ='/home/valeria/MIC3/Prediction_stroke_lesion/data/Basal_to_FU1_V8_mask_interp/'
fu_path = '/home/valeria/MIC3/Prediction_stroke_lesion/HematomaTruetaV8/'
newDataset_path = '/home/valeria/MIC3/Prediction_stroke_lesion/data/Synthetic_full/' # this path will contain synthetically created follow-up images

cases = [f.path for f in os.scandir(newDataset_path) if f.is_dir()]
cases = [case.split('/')[-1] for case in cases]

# cases = ['pt001_33','pt001_68','pt001_100','pt037_33','pt037_68','pt037_100',]


for case in cases:
        print(case, ' ', iter)
        patient = case.split('_')[0]
        vol = case.split('_')[1]

        
        command = \
            './greedy -d 3 -m NCC 2x2x2 -i /home/valeria/MIC3/Prediction_stroke_lesion/data/Synthetic_full/{}/mask.nii.gz /home/valeria/MIC3/Prediction_stroke_lesion/HematomaTruetaV8/{}/FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz -gm /home/valeria/MIC3/Prediction_stroke_lesion/HematomaTruetaV8/{}/FU1/CT_SS.nii.gz -o /home/valeria/MIC3/Prediction_stroke_lesion/data/Synthetic_full/{}/df.nii.gz -n 100x50x10'.format(case,patient,patient,case)
        
        subprocess.run(['bash','-c',command])


        command2 = \
            './greedy -d 3 -rf /home/valeria/MIC3/Prediction_stroke_lesion/data/Synthetic_full/{}/mask.nii.gz -rm /home/valeria/MIC3/Prediction_stroke_lesion/HematomaTruetaV8/{}/FU1/CT_SS.nii.gz /home/valeria/MIC3/Prediction_stroke_lesion/data/Synthetic_full/{}/CT_SS.nii.gz -ri LINEAR -r /home/valeria/MIC3/Prediction_stroke_lesion/data/Synthetic_full/{}/df.nii.gz'.format(case,patient,case,case)
       
        subprocess.run(['bash','-c',command2])

  
