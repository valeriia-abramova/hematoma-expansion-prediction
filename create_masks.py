import os
import nibabel as nib
import numpy as np
import scipy.ndimage.morphology


def extract_dimensions(image):
    
    '''
    extract voxel spacing of an image in mm
    :param image: nib.Nifti1Image
    :return dims: a tuple of (x, y, z) voxel spacing in mm
    '''
    assert type(image) == nib.nifti1.Nifti1Image, "The input image should be a nib.nifti1.Nifti1Image object"
    header = image.header

    return (header['pixdim'][1], header['pixdim'][2], header['pixdim'][3])

def calculate_volume_ml(mask, dimensions):
    '''
    calculates the volume of a tumor in ml
    It takes a binary tumor mask and calculates the amount of non-zero voxels multiplied by voxel size
    :param mask: nib.Nifti1Image
    :param dimensions: (x, y, z) tuple which contains the voxel size in mm
    :return vol: int volume in voxels
    '''
    vol = np.count_nonzero(mask)
    vol_ml = vol*np.prod(np.array(dimensions))/1000
    return vol_ml

# creating dilated masks of basal images from Basal_to_FU1 folder (the ones having the follow up and pre-registered to them)

files_path = '/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_mask/'
results_path = '/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/data/'




cases_notUse = ['pt038', 'pt170', 'pt094', 'pt103', 'pt127', 'pt054',
            'pt095','pt057','pt089','pt168','pt113','pt043',
            'pt140','pt108','pt106','pt098','pt082','pt105', # those ones bcz we use them for testing
            'pt096', 'pt115', 'pt093', 'pt091',] # remove those cases bcz basal and its mask have inconsistent dims in axial slices

cases = [f.path for f in os.scandir(files_path) if f.is_file()]
cases = [case.split('/')[-1].split('.nii.gz')[0] for case in cases]

cases = [id for id in cases if id not in cases_notUse]
cases = [
    # 'pt007', 'pt086', 'pt011', 'pt138', 'pt065', 'pt102', 
    # 'pt116', 'pt084', 'pt124', 'pt025', 'pt051', 'pt021', 'pt003', 'pt047', 'pt131', 'pt049', 'pt015', 'pt032', 'pt141', 'pt136', 'pt041', 'pt099', 
    # 'pt111', 'pt137', 'pt069', 'pt081', 'pt062', 'pt060', 'pt110', 'pt027', 'pt092', 'pt118', 'pt020', 'pt022', 'pt143', 
    'pt139', 'pt145', 'pt166', 
    'pt031', 'pt165', 'pt064', 'pt037', 'pt017', 'pt078', 'pt019', 'pt142', 'pt033', 'pt122', 'pt088', 'pt083', 'pt012', 'pt030', 'pt061', 
    'pt152', 'pt163', 'pt048', 'pt005', 'pt044', 'pt072', 'pt076', 'pt075', 'pt039', 'pt164', 'pt132', 'pt125', 'pt073', 
    'pt066', 'pt149', 'pt002', 'pt040', 'pt052', 'pt034', 'pt129', 'pt055', 'pt123', 'pt097', 'pt058', 'pt144']

for case in cases:
    print(case)
    if not os.path.isdir(results_path + case+'/'):
        os.mkdir(results_path + case+'/')

    basal_nifti = nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_mask/{}.nii.gz'.format(case)))
    basal_mask = basal_nifti.get_fdata()

    basal_vol = np.count_nonzero(basal_mask) # the volume of initial lesion
    basal_ml = calculate_volume_ml(basal_mask,extract_dimensions(basal_nifti))

    # we want to create masks more than 33% bigger than initial one
    # I choose 100 % bcz I want visible growth for small ones
    # but I also add constraint of 6 ml, bcz I don't want big ones double in size

    vol_new = 2*basal_vol
    ml_new = basal_ml+6

    basal_imgMask = (nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz'.format(case)))).get_fdata()> 0).astype(int)

    i = 0

    while basal_vol < vol_new and basal_ml < ml_new:
        mask_new = scipy.ndimage.morphology.binary_dilation(basal_mask,iterations=1, mask = basal_imgMask)
        mask_new[basal_imgMask == 0.0] = 0

        basal_vol = np.count_nonzero(mask_new)
        basal_ml = calculate_volume_ml(mask_new,extract_dimensions(basal_nifti)) 

        i += 1

        basal_mask = mask_new
    print(i)

    nib.Nifti1Image(mask_new, basal_nifti.affine, basal_nifti.header).to_filename(
        os.path.join(os.path.join(results_path, case), 'doublemask.nii.gz'))




