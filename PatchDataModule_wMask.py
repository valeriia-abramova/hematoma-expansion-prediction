import copy
import itertools
import os
import shutil
import time

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from monai.metrics import compute_hausdorff_distance
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage.morphology import binary_dilation


def load_prepared_trueta_dataset(path:str):
    """
    Loads all the prepared data from the given path and returns a dictionary with the case names as
    keys and the image, lesion mask and header as values.
    
    :param path: the path to the directory containing the prepared dataset
    :type path: str
    :return: A dictionary with the following structure:
    {
        'case_1': {
                'basal': np.array(...),
                'fu1': np.array(...)
            }
        },
        'case_2': {
                'basal': np.array(...),
                'fu1': np.array(...)
    """

    prepared_dict = {}
    # synthetic dataset 168 patients
    cases = [f.path for f in os.scandir(path) if f.is_dir()]
    cases = [case.split('/')[-1] for case in cases]
    cases = [case for case in cases if 'pt040' not in case]
    cases = [case for case in cases if 'pt018' not in case]
    cases = sorted(cases)
    # testCases = ['pt170', 'pt094', 'pt054', 'pt098', 'pt105']
    # cases = cases + testCases
    # cases = cases[:2]
    # cases = ['pt038', 'pt170', 'pt094', 'pt103', 'pt127', 'pt054', 'pt171',
    #     'pt095','pt077','pt043', 'pt005',
    #     'pt108','pt082','pt018']
    for case in cases:
        if '_' in case:
            patient = case.split('_')[0]
            basal = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz'.format(patient)))).get_fdata()
            basalbrainMask = np.expand_dims((nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz'.format(patient)))).get_fdata() > 0).astype(np.int), axis=0)
            basal_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_mask_interp/{}.nii.gz'.format(patient)))).get_fdata()
            fu1 = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'CT_SS.nii.gz'))).get_fdata()
            fu_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'mask.nii.gz'))).get_fdata()
            borderMask = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'diff.nii.gz'))).get_fdata()
        else:
            basal = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz'.format(case)))).get_fdata()
            basalbrainMask = np.expand_dims((nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz'.format(case)))).get_fdata() > 0).astype(np.int), axis=0)
            basal_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_mask_interp/{}.nii.gz'.format(case)))).get_fdata()
            fu1 = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/HematomaTruetaV7/{}/FU1/CT_SS.nii.gz'.format(case)))).get_fdata()
            fu_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/HematomaTruetaV7/{}/FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'.format(case)))).get_fdata()
            borderMask = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/HematomaTruetaV7/{}/FU1/diff.nii.gz'.format(case)))).get_fdata()

        basal = np.stack([basal, basal_lesionMask], axis = 0)
        fu1 = np.stack([fu1, fu_lesionMask], axis = 0)
        

        prepared_dict[case] = {
            'basal': basal,
            'basalbrainMask': basalbrainMask,
            'lesionMask': borderMask,
            'fu1': fu1
            }
    return prepared_dict

def load_test_dict(path:str):
    """
    Loads all the prepared data from the given path and returns a dictionary with the case names as
    keys and the image, lesion mask and header as values.
    
    :param path: the path to the directory containing the prepared dataset
    :type path: str
    :return: A dictionary with the following structure:
    {
        'case_1': {
                'basal': np.array(...),
                'fu1': np.array(...)
            }
        },
        'case_2': {
                'basal': np.array(...),
                'fu1': np.array(...)
    """

    prepared_dict = {}
    # all patients ids
    cases = ['pt038', 'pt170', 'pt094', 'pt103', 'pt127', 'pt054', 'pt171',
        'pt095','pt077','pt043', 'pt005',
        'pt108','pt082','pt018','pt057',
        'pt040','pt098']
    # cases = ['pt057',
    #     'pt040','pt098']
    # cases = ['pt038', 'pt103', 'pt127', 'pt171',
    #     'pt095','pt057','pt077','pt043', 'pt005',
    #     'pt140','pt108','pt082']
    # cases = ['pt002', 'pt003', 'pt018', 'pt031', 'pt133']
    # cases = cases[:2]
    # for fine-tuning on pt094 val on pt170

    for case in cases:
        basal = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz'.format(case)))).get_fdata()
        basalbrainMask = np.expand_dims((nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1/{}.nii.gz'.format(case)))).get_fdata() > 0).astype(np.int), axis=0)
        basal_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_mask_interp/{}.nii.gz'.format(case)))).get_fdata()
        fu1 = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/HematomaTruetaV7/{}/FU1/CT_SS.nii.gz'.format(case)))).get_fdata()
        fu_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/HematomaTruetaV7/{}/FU1/hematoma_mask_vicorob_reviewed_reoriented.nii.gz'.format(case)))).get_fdata()
        
        basal = np.stack([basal, basal_lesionMask], axis = 0)
        fu1 = np.stack([fu1, fu_lesionMask], axis = 0)
        

        prepared_dict[case] = {
            'basal': basal,
            'basalbrainMask': basalbrainMask,
            'lesionMask': basal_lesionMask,
            'fu1': fu1
            }
    return prepared_dict

def find_normalization_parameters(image):
    """
    It takes an image and returns the mean and standard deviation of the image.
    
    :param image: the image to be normalized
    :return: The mean and standard deviation of the image.
    """

    norm_img = copy.deepcopy(image)
    # norm_img[norm_img == 0] = np.NaN
    # norm_parms = (np.nanmean(norm_img, axis=(-3, -2, -1), keepdims=True), 
    #               np.nanstd(norm_img, axis=(-3, -2, -1), keepdims=True))
    norm_parms = (np.nanmin(norm_img, axis=(-3, -2, -1), keepdims=True), 
                   np.nanmax(norm_img, axis=(-3, -2, -1), keepdims=True))

    return norm_parms 

def normalize_image(image_patch, parameters):
    """
    The function takes an image patch and a list of parameters as input, and returns the normalized
    image patch.
    
    :param image_patch: the image patch that we want to normalize
    :param parameters: [mean, std]
    :return: The normalized image patch.
    """
    if len(image_patch.shape) == 3: # 2D case
        parameters = (np.squeeze(parameters[0], axis=-1),
                      np.squeeze(parameters[1], axis=-1))

    minmax = (image_patch-parameters[0]) / (parameters[1]-parameters[0]) # [0,1]
    minusone = (image_patch - (parameters[0]+parameters[1])/2)/((parameters[1]-parameters[0])/2) # [-1,1]

    # return (image_patch - parameters[0]) / parameters[1] 
    return minmax


def generate_stroke_instructions(data: dict, patch_size: tuple, patch_step: tuple, do_data_augmentation: bool, patches_per_image: int):
    """
    For each image, find the center of each patch, and then create a dictionary of instructions for
    each patch.
    
    :param data: the dictionary of images and masks
    :param patch_size: The size of the patch to extract from the image
    :param patch_step: The step size for the patch center
    :param do_data_augmentation: Whether to perform data augmentation on the patches
    :param do_normalization: Whether to do 0 mean unit variance normalization
    :param patches_per_image: The number of patches to extract from each image
    :return: A list of dictionaries, each dictionary contains the case_id, center, patch_size,
    do_data_augmentation, and norm_params
    """

    all_instruction = []

    if patch_size[-1] == 1: # 2D case
        raise NotImplementedError('2D is still not implemented')

    else: # 3D case
        for case_id, case_dict in data.items(): # For each image (CH, X, Y, Z) and patch size
            # Calculate mean and std to perform image normalization
            norm_parms = find_normalization_parameters(case_dict['basal'][0])
            patch_instruction =[]

            # image_centers = sample_centers_uniform(np.squeeze(case_dict['basal'], axis = 0), patch_shape= patch_size, max_centers= patches_per_image,extraction_step=(16,16,16), mask=np.squeeze(case_dict['basalbrainMask'],axis=0))
            image_centers = sample_centers_balanced(label_img=case_dict['lesionMask'],patch_shape=patch_size,num_centers=patches_per_image,mask=np.squeeze(case_dict['basalbrainMask'],axis=0))
            # (np.squeeze(case_dict['basal'],axis=0), patch_shape= patch_size, max_centers= patches_per_image,extraction_step=(16,16,16), mask=np.squeeze(case_dict['basalbrainMask'],axis=0))
            patch_instruction += [{'case_id': case_id,
                                'center': center,
                                'patch_size': patch_size,
                                'do_data_augmentation': do_data_augmentation,
                                'norm_params': norm_parms} for center in image_centers]
            all_instruction+=patch_instruction

    return all_instruction


def extract_stroke_patch(instructions: dict, data: dict):     
    """
    The function takes as input a dictionary containing the instructions for extracting the patch, and
    a dictionary containing the data for the case. It then extracts the patch, normalizes it, performs
    data augmentation on it, and returns the patch as a Pytorch tensor.
    
    :param instructions: a dictionary containing the following keys:
    :param data: a dictionary containing the image and lesion mask of the case
    :return: The image_patch_torch and lesion_patch_torch are being returned.
    """

    case = data[instructions['case_id']]
    image = case['basal']
    gt = case['fu1']
    center = instructions['center']
    patch_size = instructions['patch_size']
    do_data_augmentation = instructions['do_data_augmentation']


    # Define patch slice dimensions
    patch_slice = (
    slice(None), # CH
    slice(int(center[0] - np.ceil(patch_size[0] / 2.0)), int(center[0] + np.floor(patch_size[0] / 2.0))), # X range 
    slice(int(center[1] - np.ceil(patch_size[1] / 2.0)), int(center[1] + np.floor(patch_size[1] / 2.0))), # Y range
    slice(int(center[2] - np.ceil(patch_size[2] / 2.0)), int(center[2] + np.floor(patch_size[2] / 2.0)))) # Z range


    # Create a new variable containing only the extracted patch
    image_patch = copy.deepcopy(image[patch_slice])
    gt_patch = copy.deepcopy(gt[patch_slice])

    # 2D/3D compatibility
    # image_patch = np.squeeze(image_patch, axis=tuple(ax for ax in range(-3, 0, 1) if image_patch.shape[ax] == 1))
    # lesion_patch = np.squeeze(lesion_patch, axis=tuple(ax for ax in range(-3, 0, 1) if lesion_patch.shape[ax] == 1))

    # Normalize the image_patch

    image_patch[0] = normalize_image(image_patch[0], instructions['norm_params']) # case with mask as additional input channel
    gt_patch[0] = normalize_image(gt_patch[0], instructions['norm_params'])
  
    # image_patch = normalize_image(image_patch, instructions['norm_params'])
    # gt_patch = normalize_image(gt_patch, instructions['norm_params'])

    # Transform the patch to a Pytorch tensor
    image_patch_torch = torch.tensor(np.ascontiguousarray(image_patch), dtype=torch.float)
    gt_patch_torch = torch.tensor(np.ascontiguousarray(gt_patch), dtype=torch.float)

    return image_patch_torch, gt_patch_torch

def sample_centers_uniform(vol, patch_shape, extraction_step, max_centers=None, mask=None):
    """
    This sampling is uniform, not regular! It will extract patches

    :param vol:
    :param patch_shape:
    :param extraction_step:
    :param max_centers: (O'ptional) If given, the centers will be resampled to max_len
    :param mask: (O'ptional) If given, discard centers not in foreground
    :return:
    """

    assert len(vol.shape) == len(patch_shape) == len(extraction_step), '{}, {}, {}'.format(vol.shape, patch_shape, extraction_step)
    if mask is not None:
        assert len(mask.shape) == len(vol.shape), '{}, {}'.format(mask.shape, vol.shape)
        mask = mask.astype('float16')

    # Get patch span from the center in each dimension
    span = [[int(np.ceil(dim / 2.0)), int(np.floor(dim / 2.0))] for dim in patch_shape]

    # Generate the sampling indexes for each dimension first and then get all their combinations (itertools.product)
    dim_indexes = [list(range(sp[0], vs - sp[1], step)) for sp, vs, step in zip(span, vol.shape, extraction_step)]
    centers = list(itertools.product(*dim_indexes))

    if mask is not None:
        centers = [c for c in centers if mask[c[0], c[1], c[2]] != 0.0]
    if max_centers is not None:
        centers = resample_regular(centers, max_centers)

    return centers

def sample_centers_balanced(label_img, patch_shape, num_centers, add_rand_offset=False, exclude=None, mask=None):
    """Samples centers for patch extraction from the given volume. An equal number of centers is sampled from each label.

    :param label_img: label image with dimensions (X, Y, Z) containing the label
    :param tuple patch_shape: tuple (x,y,z) shape of the patches to be extracted on returned centers
    :param int num_centers:
    :param bool add_rand_offset: if True, adds a random offset of up to half the patch size to sampled centers.
    :param list exclude: list with label ids to exclude from sampling
    :return List[tuple]: the sampled centers as a list of (x,y,z) tuples
    """

    assert len(label_img.shape) == len(patch_shape), 'len({}) ¿=? len({})'.format(label_img.shape, patch_shape)
    if mask is not None:
        # mask_brain = mask[0, :, :, :]
        mask_brain = mask
        assert len(mask_brain.shape) == len(label_img.shape), '{}, {}'.format(mask_brain.shape, label_img.shape)
        mask = mask_brain.astype('float16')

    label_ids = np.unique(label_img).tolist()

    if exclude is not None:
        label_ids = [i for i in label_ids if i not in exclude]

    centers_labels = {label_id: np.argwhere((label_img == label_id) & (mask != 0.0)) for label_id in label_ids}

    # adding some background patches to see if they will effect background of resulting deformation field
    # creating several centers from background
    # background = np.argwhere((label_img == 0.0) & (mask == 0.0))
    # background = background[:5]


    # # Resample (repeating or removing) to appropiate number

    # centers_labels = \
        # {k: resample_regular(v, num_centers // len(label_ids)) for k, v in centers_labels.items()}
    

    centers_labels[label_ids[0]] = resample_regular(centers_labels[label_ids[0]], 0.2*(num_centers))
    centers_labels[label_ids[1]] = resample_regular(centers_labels[label_ids[1]], 0.8*(num_centers))

    # change several centers from zero label to background centers
    # centers_labels[label_ids[0]] = list(centers_labels[label_ids[0]])
    # centers_labels[label_ids[0]].extend(background)

    # Add random offset of up to half the patch size
    if add_rand_offset:
        for label_centers in centers_labels.values():
            np.random.seed(0)  # Repeatability
            label_centers += np.expand_dims(np.divide(patch_shape, 2).astype(int), axis=0) * (
                    2.0 * np.random.rand(len(label_centers), len(label_centers[0])) - 1.0)

    # Clip so not out of bounds
    for k in centers_labels.keys():
        centers_labels[k] = np.clip(centers_labels[k],
                                        a_min=np.ceil(np.divide(patch_shape, 2.0)).astype(int),
                                        a_max=label_img.shape - np.floor(
                                            np.divide(patch_shape, 2.0).astype(int))).astype(int)

    # Join the centers of each label and return in appropiate format
    return [tuple(c) for c in np.concatenate(list(centers_labels.values()), axis=0)]

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
    volume_intersect = (np.bitwise_and(ground_truth, prediction)).sum()
    return np.round(2 * volume_intersect / volume_sum, 3).astype(np.float64)

def compute_sensitivity_and_specificity(ground_truth, prediction):
    """
    It takes in the ground truth and the prediction, and returns the sensitivity, specificity, and
    f-score.
    
    :param ground_truth: the ground truth labels
    :param prediction: the output of the model
    :return: The sensitivity, specificity, and f_score of the model.
    """

    ground_truth = np.squeeze(ground_truth)
    prediction = (prediction > 0.5).astype(np.int_)

    TP = np.count_nonzero(np.logical_and(ground_truth == 1, prediction == 1))
    TN = np.count_nonzero(np.logical_and(ground_truth == 0, prediction == 0))
    FN = np.count_nonzero(np.logical_and(ground_truth == 1, prediction == 0))
    FP = np.count_nonzero(np.logical_and(ground_truth == 0, prediction == 1))

    sensitivity = np.round(TP / (TP + FN), 3).astype(np.float64)
    specificity = np.round(TN / (TN + FP), 3).astype(np.float64)
    f_score = np.round((TP)/(TP + 0.5*(FP + FN)), decimals=3).astype(np.float64)
        
    return (sensitivity, specificity, f_score)
    
def ravd(result, reference):
    """
    Relative absolute volume difference.
    
    Compute the relative absolute volume difference between the (joined) binary objects
    in the two images.
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
        
    Returns
    -------
    ravd : float
        The relative absolute volume difference between the object(s) in ``result``
        and the object(s) in ``reference``. This is a percentage value in the range
        :math:`[-1.0, +inf]` for which a :math:`0` denotes an ideal score.
        
    Raises
    ------
    RuntimeError
        If the reference object is em'pty.
        
    See also
    --------
    :func:`dc`
    :func:`precision`
    :func:`recall`
    
    Notes
    -----
    This is not a real metric, as it is directed. Negative values denote a smaller
    and positive values a larger volume than the reference.
    This implementation does not check, whether the two supplied arrays are of the same
    size.
    
    Examples
    --------
    Considering the following inputs
    
    >>> import numpy
    >>> arr1 = numpy.asarray([[0,1,0],[1,1,1],[0,1,0]])
    >>> arr1
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])
    >>> arr2 = numpy.asarray([[0,1,0],[1,0,1],[0,1,0]])
    >>> arr2
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])
           
    comparing `arr1` to `arr2` we get
    
    >>> ravd(arr1, arr2)
    -0.2
    
    and reversing the inputs the directivness of the metric becomes evident
    
    >>> ravd(arr2, arr1)
    0.25
    
    It is important to keep in mind that a perfect score of `0` does not mean that the
    binary objects fit exactely, as only the volumes are compared:
    
    >>> arr1 = numpy.asarray([1,0,0])
    >>> arr2 = numpy.asarray([0,0,1])
    >>> ravd(arr1, arr2)
    0.0
    
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
        
    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)
    
    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')
    
    return (vol1 - vol2) / float(vol2)



def split_stroke_crossvalidation_folds(num_folds: int, prepared_dict: dict):
    """
    The function takes a dictionary of prepared cases and returns a dictionary of crossvalidation
    folds, where each fold is a dictionary with a list of training and test cases.
    
    :param num_folds: the number of cross-validation folds you want to split the data into
    :param prepared_dict: the dictionary of prepared data
    :return: A dictionary with the keys being the fold number and the values being a dictionary with the
    keys being train and test and the values being a list of case_ids.
    """

    crossvalidation_folds_dict = {}

    Dataset = [case_id for case_id, info in prepared_dict.items()]    
    assert len(Dataset) == len(prepared_dict.items())

    cases_per_split = int(np.ceil(len(Dataset)/num_folds))

    crossval_list = [Dataset[cases_per_split * fold_index:cases_per_split * fold_index + cases_per_split] for fold_index in range(num_folds)]

    for fold in range(num_folds):
        crossval_list_fold = crossval_list.copy()
        val_list = crossval_list_fold[fold]
        del crossval_list_fold[fold]
        train_list = crossval_list_fold
        
        crossvalidation_folds_dict[fold] = {
            'train' : sum(train_list, []),
            'test' : val_list
        }

    return crossvalidation_folds_dict 
    
def resample_regular(l: list, n: int):
    """
    Resamples a given list to have length `n`.
    
    List elements are repeated or removed at regular intervals to reach the desired length.

    :param list l: list to resample
    :param int n: desired length of resampled list
    :return list: the resampled list of length `n`

    :Example:

    >>> resample_regular([0, 1, 2, 3, 4, 5], n=3)
    [0, 2, 4]
    >>> resample_regular([0, 1, 2, 3], n=6)
    [0, 1, 2, 3, 0, 2]
    """

    n = int(n)
    if n <= 0:
        return []

    if len(l) < n:  # List is smaller than n (Repeat elements)
        resampling_idxs = list(range(len(l))) * (n // len(l))  # Full repetitions

        if len(resampling_idxs) < n:  # Partial repetitions
            resampling_idxs += np.round(np.arange(
                start=0., stop=float(len(l)) - 1., step=len(l) / float(n % len(l))), decimals=0).astype(int).tolist()

        assert len(resampling_idxs) == n
        return [l[i] for i in resampling_idxs]
    elif len(l) > n:  # List bigger than n (Subsample elements)
        resampling_idxs = np.round(np.arange(
            start=0., stop=float(len(l)) - 1., step=len(l) / float(n)), decimals=0).astype(int).tolist()

        assert len(resampling_idxs) == n
        return [l[i] for i in resampling_idxs]
    else:
        return l  

class InstructionDataset(Dataset):
    def __init__(self, instructions, data, get_item_func):
        assert callable(get_item_func)
        self.instructions = instructions
        self.data = data
        self.get_item = get_item_func
        
    def __len__(self): # Returns the number of samples in our dataset
        return len(self.instructions)
    
    def __getitem__(self, idx): # Returns a sample from the dataset at a given index
        return self.get_item(self.instructions[idx], self.data)

class PatchDataModule_wMask(pl.LightningDataModule):
    def __init__(self, prepared_data_path, test_path, patch_size, patch_step, do_skull_stripping, 
                batch_size, num_workers, patches_per_image, validation_fraction=0.2, fold_split=None, do_data_augmentation=False):
        super().__init__()
        self.prepared_data_path = prepared_data_path
        self.test_path = test_path
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.do_skull_stripping = do_skull_stripping
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.num_workers = num_workers
        self.prepared_dict = None
        self.test_dict = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.fold_index = 0
        self.do_data_augmentation = do_data_augmentation
        self.patches_per_image = patches_per_image

    def setup(self, stage='None'):
        if self.prepared_dict is None:
            self.prepared_dict = load_prepared_trueta_dataset(self.prepared_data_path)

        if self.test_dict is None:
            self.test_dict = load_test_dict(self.test_path)
         
        self.set_fold()
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) 
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_test_cases(self):
        return [os.path.join(self.prepared_data_path, case) for case in self.test_dict]
    
    def compute_image_measures(self, case_num, inference_result, header):
        inference_result = np.round(inference_result).astype('int') 
        image_measures = {}
        ground_truth = self.test_dict[case_num]['fu1']
        image_measures['DSC'] = compute_dice_similarity_coefficient(ground_truth, inference_result)
        image_measures['sensitivity'], image_measures['specificity'], image_measures['f_score'] = compute_sensitivity_and_specificity(ground_truth, 
                                                                                                           inference_result)
        image_measures['pixdim'] = tuple(np.round(header['pixdim'][1:4], 2).astype(np.float64))

        image_measures['hausdorff_distance'] = np.round(compute_hausdorff_distance(y_pred=inference_result[np.newaxis, np.newaxis, ...].astype(np.int16), 
                                                                          y=ground_truth[np.newaxis, ...].astype(np.int16), 
                                                                          percentile=95).numpy().item(), 2).astype(np.float64)
        # added another metric                                                                  
        image_measures['ravd'] = ravd(ground_truth, inference_result)
        
        return image_measures

    def set_fold(self):

        self.train_val_dict = {case_id: case_values for case_id, case_values in self.prepared_dict.items()}

        self.test_dict = {case_id: case_values for case_id, case_values in self.test_dict.items()}  

        train_dict = dict(list(self.train_val_dict.items())[:int(np.round(len(self.train_val_dict) * (1.0 - self.validation_fraction)))+1])
        val_dict = dict(list(self.train_val_dict.items())[int(np.round(len(self.train_val_dict) * (1.0 - self.validation_fraction)))+1:])         

        train_patch_instructions = generate_stroke_instructions(train_dict, self.patch_size, self.patch_step, self.do_data_augmentation, self.patches_per_image)
        val_patch_instructions = generate_stroke_instructions(val_dict, self.patch_size, self.patch_step, self.do_data_augmentation, self.patches_per_image)
        test_patch_instructions = generate_stroke_instructions(self.test_dict, self.patch_size, self.patch_step, self.do_data_augmentation, self.patches_per_image)
        self.train_dataset = InstructionDataset(train_patch_instructions, train_dict, extract_stroke_patch)   
        self.val_dataset = InstructionDataset(val_patch_instructions, val_dict, extract_stroke_patch) 
        self.test_dataset = InstructionDataset(test_patch_instructions, self.test_dict, extract_stroke_patch) 

if __name__ == "__main__":
    prepared_data_path = '/home/valeria/Master_Thesis_stroke_seg/HematomaTrueta_new'
    NUM_WORKERS = 32

    stroke_dict = load_prepared_trueta_dataset(prepared_data_path)
    StrokeDM = PatchDataModule_wMask(prepared_data_path=prepared_data_path, 
                                    patch_size=(64,64,8), patch_step=None, do_skull_stripping=False, 
                                    batch_size=16, validation_fraction=0.2, num_workers=NUM_WORKERS, patches_per_image=3000)

#    BrainDM.prepare_data()
    StrokeDM.setup()