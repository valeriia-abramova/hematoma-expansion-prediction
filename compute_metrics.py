import numpy as np
import nibabel as nib
import os
import csv


def save_to_csv(filepath, dict_list, append=False):
    """Saves a list of dictionaries as a .csv file.

    :param str filepath: the output filepath
    :param List[Dict] dict_list: The data to store as a list of dictionaries.
        Each dictionary will correspond to a row of the .csv file with a column for each key in the dictionaries.
    :param bool append: If True, it will append the contents to an existing file.

    :Example:

    >>> save_to_csv('data.csv', [{'id': '0', 'score': 0.5}, {'id': '1', 'score': 0.8}])
    """
    assert isinstance(dict_list, list) and all([isinstance(d, dict) for d in dict_list])
    open_mode = 'a' if append else 'w+'
    with open(filepath, mode=open_mode) as f:
        csv_writer = csv.DictWriter(f, dict_list[0].keys(), restval='', extrasaction='raise', dialect='unix')
        if not append or os.path.getsize(filepath) == 0:
            csv_writer.writeheader()
        csv_writer.writerows(dict_list)

def compute_dice_similarity_coefficient(ground_truth, prediction):
    """
    Given the ground truth and the prediction this function computes the 
    Dice Similarity coefficient (DSC).
    
    :param ground_truth: a 3D numpy array of type bool
    :param prediction: the predicted segmentation
    :return: The dice similarity coefficient is being returned.
    """

    volume_sum = ground_truth.sum() + prediction.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (np.bitwise_and(ground_truth > 0, prediction > 0)).sum()
    return np.round(2 * volume_intersect / volume_sum, 3).astype(np.float64)


def doc(ground_truth, prediction, initial_mask):
    """
    Given the ground truth, predicted mask, and baseline mask, doc computes dice score absolute change ratio
    as proposed in Xiao, T. et al. (2021). Intracerebral Haemorrhage Growth Prediction Based on Displacement Vector Field and Clinical Metadata. 
    In: , et al. Medical Image Computing and Computer Assisted Intervention - MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science(), 
    vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_71
    
    :param ground_truth: a 3D numpy array of type bool
    :param prediction: the predicted follow-up segmentation
    :param initial_mask: the baseline mask
    :return: The doc coefficient is being returned
    """
    fu_baseline_diff = np.abs(ground_truth - initial_mask)
    pred_diff = np.abs(prediction - ground_truth)
    intersection = (np.bitwise_and(pred_diff > 0, fu_baseline_diff > 0)).sum()
    volume_sum = fu_baseline_diff.sum() + pred_diff.sum()
    return np.round(intersection / volume_sum, 3).astype(np.float64)

def aev(ground_truth, prediction, header):
    """
    Given the ground truth and predicted mask, aev measures average prediction error of hematoma volume
    as proposed in Xiao, T. et al. (2021). Intracerebral Haemorrhage Growth Prediction Based on Displacement Vector Field and Clinical Metadata. 
    In: , et al. Medical Image Computing and Computer Assisted Intervention - MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science(), 
    vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_71
    
    :param ground_truth: a 3D numpy array of type bool
    :param prediction: the predicted follow-up segmentation
    :param header: header of one of the nifti images to find out voxel dims
    :return: The aev coefficient is being returned
    """
    vol_voxels = prediction.sum() - ground_truth.sum()
    vol_ml = vol_voxels*np.prod(np.array((header['pixdim'][1], header['pixdim'][2], header['pixdim'][3])))/1000
    return  vol_ml

def aaev(ground_truth, prediction, header):
    """
    Given the ground truth and predicted mask, aaev measures absolute average prediction error of hematoma volume
    as proposed in Xiao, T. et al. (2021). Intracerebral Haemorrhage Growth Prediction Based on Displacement Vector Field and Clinical Metadata. 
    In: , et al. Medical Image Computing and Computer Assisted Intervention - MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science(), 
    vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_71
    
    :param ground_truth: a 3D numpy array of type bool
    :param prediction: the predicted follow-up segmentation
    :param header: header of one of the nifti images to find out voxel dims
    :return: The aaev coefficient is being returned
    """
    vol_voxels = prediction.sum() - ground_truth.sum()
    vol_ml = vol_voxels*np.prod(np.array((header['pixdim'][1], header['pixdim'][2], header['pixdim'][3])))/1000
    return  np.abs(vol_ml)


cases = ['pt038', 'pt170', 'pt094', 'pt103', 'pt127', 'pt054', 'pt171',
        'pt095','pt077','pt043', 'pt005',
        'pt108','pt082','pt018','pt057',
        'pt040','pt098','pt207','pt208']
# cases = ['pt038', 'pt170', 'pt094', 'pt103', 'pt127', 'pt054', 'pt171',
#         'pt095','pt077','pt043', 'pt005',
#         'pt108','pt018','pt057',
#         'pt040']
# cases = ['pt057',
#         'pt040','pt098']

pred_path = '/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/046-patchBalanced80-ppi500-adam00001-bs32-l1loss-ps323232-border-mask-1000epochs-as024-onDistanceTr-2moreTestImgs/results/'
baseline_path = '/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_mask_interp/'
fu_path = '/home/valeria/Prediction_stroke_lesion/HematomaTruetaV7/'

all_measures = []

for case in cases:

    print(case)


    predicted_nifti = nib.funcs.as_closest_canonical(nib.load(os.path.join(pred_path,case+'_mask.nii.gz')))
    predicted_mask = nib.funcs.as_closest_canonical(nib.load(os.path.join(pred_path,case+'_mask.nii.gz'))).get_fdata()
    baseline_mask = nib.funcs.as_closest_canonical(nib.load(os.path.join(baseline_path,case+'.nii.gz'))).get_fdata()
    fu_mask = nib.funcs.as_closest_canonical(nib.load(os.path.join(fu_path,case,'FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'))).get_fdata()

    diff_gt = nib.funcs.as_closest_canonical(nib.load(os.path.join(fu_path,case,'FU1/diff.nii.gz'))).get_fdata()
    diff_pred = predicted_mask - baseline_mask
    diff_pred = np.round(diff_pred)

    measures = {}

    measures = { 'id': case,
        'dice': compute_dice_similarity_coefficient(fu_mask,predicted_mask),
        'doc': compute_dice_similarity_coefficient(diff_gt, diff_pred),
        'aev': aev(fu_mask,predicted_mask, predicted_nifti.header),
        'aaev': aaev(fu_mask, predicted_mask, predicted_nifti.header)
    } 
    all_measures.append(measures)

save_to_csv('/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/046-patchBalanced80-ppi500-adam00001-bs32-l1loss-ps323232-border-mask-1000epochs-as024-onDistanceTr-2moreTestImgs/results/measures_2newTest.csv',all_measures)