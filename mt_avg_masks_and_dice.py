import numpy as np
import os
import niclib as nl
import nibabel as nib
from matplotlib import pyplot as plt
from scipy import ndimage
import csv

import torch

def _convert_to_scalar(result, was_numpy):
    if was_numpy:
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy().item()
    return result


def _convert_to_torch(output, target):
    was_numpy = False
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
        was_numpy = True
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
        was_numpy = True
    return output, target, was_numpy


def dice(output, target, background_label=0):
    """Dice Similarity Coefficient. Output and target must contain integer labels."""
    output, target, was_numpy = _convert_to_torch(output, target)
    assert output.size() == target.size(), '{} != {}'.format(output.size(), target.size())

    output_mask = (output != background_label).bool()
    target_mask = (target != background_label).bool()

    result = 2.0 * torch.sum(target.bool() * output.bool()) / (2.0 * torch.sum(target.bool() * output.bool())
                                                               + torch.sum((target == 0).bool() * output.bool())
                                                               + torch.sum(target.bool() * (output == 0).bool()))

    return _convert_to_scalar(result, was_numpy)


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

dir_path = "/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/results_mt/right_kernel"

cases = [
    'pt100', 'pt169', 'pt153',   'pt171', 'pt109',  'pt161', 'pt114', 'pt133', 
    'pt134', 'pt167',  'pt130', 'pt101', 'pt119', 'pt120', 'pt135', 'pt159', 'pt121',  'pt150', 'pt128', 'pt151', 'pt138',  'pt102', 
    'pt116', 'pt124', 'pt131', 'pt141', 'pt136', 'pt111', 'pt137', 'pt110', 'pt118', 'pt143', 'pt139', 'pt145', 'pt166', 'pt165', 'pt142', 'pt122', 
    'pt152', 'pt163','pt164', 'pt132', 
    # 'pt125', # the fu1 mask is zero 
    'pt149', 'pt129',  'pt123','pt144']
    


header = 0
affine = 0

folds = [f for f in range(5)]

all_metrics = []
    
for case in cases:
    masks_folds = []
    header = 0
    affine = 0
    proto = nib.load(os.path.join(dir_path,case,"right_kernel_probs_fold0.nii.gz"))
    # proto = nib.funcs.as_closest_canonical(proto)
    proto_data = proto.get_fdata()
    result = np.zeros_like(proto_data)
    for fold in range(0,5):
            filename = os.path.join(dir_path,case,"right_kernel_probs_fold{}.nii.gz".format(fold))
            print(filename)
            image = nib.load(filename)
            # image = nib.funcs.as_closest_canonical(image)
            image_data = image.get_fdata()
            masks_folds.append(image_data)
            header = image.header
            affine = image.affine
    print("averaging masks {}".format(case))
    #    print("masks folds shape", np.shape(masks_folds))
    #    print(np.shape(result))
    result = np.round(np.average(np.array(masks_folds), axis=0))
    #    print(np.shape(np.average(masks_folds, axis=0)))
    filename_save = os.path.join(dir_path,case,"right_kernel_avg_mask.nii.gz")
    result_img = nib.Nifti1Image(result, affine, header)
    nib.nifti1.save(result_img, filename_save)
    masks_folds = []

    # gt = nib.funcs.as_closest_canonical(nib.load('/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/masks_mt/{}/right_kernel.nii.gz'.format(case))).get_fdata()
    # real_gt = nib.funcs.as_closest_canonical(nib.load('/home/valeria/Prediction_stroke_lesion/HematomaTruetaV7/{}/FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'.format(case))).get_fdata()
    gt = nib.load('/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/masks_mt/{}/right_kernel.nii.gz'.format(case)).get_fdata()
    real_gt = nib.load('/home/valeria/Prediction_stroke_lesion/HematomaTruetaV7/{}/FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'.format(case)).get_fdata()

    vol_diff = np.count_nonzero(gt)/np.count_nonzero(real_gt)

    metric_dice = dice(result, gt)

    metrics = { 'id': case,
        'dice': metric_dice,
        'volume difference': vol_diff
    } 
    all_metrics.append(metrics)


save_to_csv('/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/results_mt/right_kernel/dice.csv',all_metrics)
