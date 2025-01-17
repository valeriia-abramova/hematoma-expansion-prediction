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
from monai.transforms import RandAffine, RandFlip


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
    # download the real and synthetic images from the defined paths
    for p in path:
        if p == path[0]:
            cases = ['pt034','pt061','pt163','pt071','pt005','pt054',
                'pt197','pt040','pt049','pt029','pt094','pt108',
                'pt183','pt155','pt043','pt127','pt173','pt170',
                'pt207','pt208','pt179','pt018','pt056','pt098',
                'pt171','pt095','pt082','pt057','pt126','pt191',]
            for case in cases:
                basal = nib.funcs.as_closest_canonical(nib.load(os.path.join('path_to_baseline/{}.nii.gz'.format(case)))).get_fdata().astype(np.float16)
                basalbrainMask = np.expand_dims((nib.funcs.as_closest_canonical(nib.load(os.path.join('path_to_baseline/{}.nii.gz'.format(case)))).get_fdata() > 0).astype(np.int), axis=0)
                basal_lesionMask = np.round(nib.funcs.as_closest_canonical(nib.load(os.path.join('path_to_baseline_mask/{}.nii.gz'.format(case)))).get_fdata()).astype(np.uint8)
                fu1 = nib.funcs.as_closest_canonical(nib.load(os.path.join('path_to_fu/{}/FU1/CT_SS.nii.gz'.format(case)))).get_fdata().astype(np.float16)
                fu_lesionMask = np.round(nib.funcs.as_closest_canonical(nib.load(os.path.join('path_to_fu/{}/FU1/hematoma_mask.nii.gz'.format(case)))).get_fdata()).astype(np.uint8)
                borderMask = np.round(nib.funcs.as_closest_canonical(nib.load(os.path.join('path_to_fu/{}/FU1/diff.nii.gz'.format(case)))).get_fdata()).astype(np.uint8)
                case = case+'_0'
                basal_full = np.stack([basal, basal_lesionMask], axis = 0)
                fu1_full = np.stack([fu1, fu_lesionMask], axis = 0)

                del basal_lesionMask
                del fu_lesionMask
                del basal
                del fu1
                

                prepared_dict[case] = {
                    'basal': basal_full,
                    'basalbrainMask': basalbrainMask,
                    'lesionMask': borderMask,
                    'fu1': fu1_full
                    }
        else:
            pts = ['pt034','pt061','pt163','pt071','pt005','pt054',
                'pt197','pt040','pt049','pt029','pt094','pt108',
                'pt183','pt155','pt043','pt127','pt173','pt170',
                'pt207','pt208','pt179','pt018','pt056','pt098',
                'pt171','pt095','pt082','pt057','pt126','pt191',]
            cases = [f.path for f in os.scandir(path[1]) if f.is_dir()]
            cases = [case.split('/')[-1] for case in cases]

            cases = [case for case in cases if '_' in case]
            cases = [case for case in cases for pt in pts if pt in case]
            for case in cases:
                basal = nib.funcs.as_closest_canonical(nib.load(os.path.join(path[1],case+'/Basal/CT_SS.nii.gz'))).get_fdata().astype(np.float16)
                basalbrainMask = np.expand_dims((nib.funcs.as_closest_canonical(nib.load(os.path.join(path[1],case+'/Basal/CT_SS.nii.gz'))).get_fdata() > 0).astype(np.int), axis=0)
                basal_lesionMask = np.round(nib.funcs.as_closest_canonical(nib.load(os.path.join(path[1],case+'/Basal/mask.nii.gz'))).get_fdata())
                fu1 = nib.funcs.as_closest_canonical(nib.load(os.path.join('path_to_fu/{}/FU1/CT_SS.nii.gz'.format(case.split('_')[0])))).get_fdata().astype(np.float16)
                fu_lesionMask = np.round(nib.funcs.as_closest_canonical(nib.load(os.path.join('path_to_fu/{}/FU1/hematoma_mask.nii.gz'.format(case.split('_')[0])))).get_fdata())
                borderMask = fu_lesionMask - basal_lesionMask
                borderMask[borderMask == -1.0] = 0.0
                fu_lesionMask = fu_lesionMask.astype(np.uint8)
                basal_lesionMask = basal_lesionMask.astype(np.uint8)
                borderMask = borderMask.astype(np.uint8)
                # case = case+'_2'
                basal_full = np.stack([basal, basal_lesionMask], axis = 0)
                fu1_full = np.stack([fu1, fu_lesionMask], axis = 0)

                del basal_lesionMask
                del fu_lesionMask
                del basal
                del fu1
                

                prepared_dict[case] = {
                    'basal': basal_full,
                    'basalbrainMask': basalbrainMask,
                    'lesionMask': borderMask,
                    'fu1': fu1_full
                    }
    return prepared_dict



def find_normalization_parameters(image):
    """
    It takes an image and returns the mean and standard deviation of the image.
    
    :param image: the image to be normalized
    :return: The mean and standard deviation of the image.
    """

    norm_img = copy.deepcopy(image)
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


    if do_data_augmentation:
        affine = RandAffine(prob=0.75, rotate_range = (0.5, 0.5,0.2), padding_mode='zeros', as_tensor_output=False)
        flip = RandFlip(prob = 0.75, spatial_axis=1)
        concatenated_patches = np.concatenate((image_patch, gt_patch), axis=0)
        concatenated_patches = affine(concatenated_patches, mode='bilinear')
        concatenated_patches = flip(concatenated_patches)

        num_channels_gt = gt_patch.shape[0]
        image_patch = concatenated_patches[:-num_channels_gt, :, :,:].numpy()
        gt_patch = concatenated_patches[-num_channels_gt:, :, :,:].numpy()

        # Round bilinearly interpolated labels
        gt_patch[1] = (gt_patch[1] > 0.5).astype(int)
        image_patch[1] = (image_patch[1] > 0.5).astype(int)


    # Normalize the image_patch

    image_patch[0] = normalize_image(image_patch[0], instructions['norm_params']) # case with mask as additional input channel
    gt_patch[0] = normalize_image(gt_patch[0], instructions['norm_params'])


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

    centers_labels[label_ids[0]] = resample_regular(centers_labels[label_ids[0]], 0.2*(num_centers))
    centers_labels[label_ids[1]] = resample_regular(centers_labels[label_ids[1]], 0.8*(num_centers))

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

    # when training on real cases with baseline vars
    Dataset = [case_id.split('_')[0] for case_id, info in prepared_dict.items()]
    Dataset = list(set(Dataset))    
    assert 6*len(Dataset) == len(prepared_dict.items())

    cases_per_split = int(np.ceil(len(Dataset)/num_folds))

    crossval_list = [Dataset[cases_per_split * fold_index:cases_per_split * fold_index + cases_per_split] for fold_index in range(num_folds)]

    crossval_list = sorted(crossval_list)

    for fold in range(num_folds):
        crossval_list_fold = crossval_list.copy()
        val_list = crossval_list_fold[fold]
        del crossval_list_fold[fold]
        train_list = crossval_list_fold
        for i in range(6):
            train_list += [[case[0]+'_'+str(i)] for case in train_list]

        crossvalidation_folds_dict[fold] = {
            'train' : sum(train_list, []),
            'test' : val_list
        }
    
    new_dict = {}

    for i in range(len(list(crossvalidation_folds_dict.values()))):
        new_dict[i] = list(crossvalidation_folds_dict.values())[i]

    return new_dict 
    
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

class PatchDataModule_wMask_crossval(pl.LightningDataModule):
    def __init__(self, prepared_data_path, test_path, patch_size, patch_step, do_skull_stripping, 
                batch_size, num_workers, patches_per_image, validation_fraction=0.2, num_folds=5, fold_split=None, do_data_augmentation=False):
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
        self.num_folds = num_folds
        self.do_data_augmentation = do_data_augmentation
        self.patches_per_image = patches_per_image

        if fold_split is not None:
            assert num_folds == len(fold_split)
        self.fold_split = fold_split


    def setup(self, stage='None'):
        if self.prepared_dict is None:
            self.prepared_dict = load_prepared_trueta_dataset(self.prepared_data_path)

        if self.fold_split is None:
            # Split cases manually
           self.fold_split = split_stroke_crossvalidation_folds(self.num_folds, self.prepared_dict)             

        self.set_fold()
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) 
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_test_cases(self):
        return [os.path.join(self.test_path, case) for case in self.test_dict]
    
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

        self.train_val_dict = {case_id: case_values for case_id, case_values in self.prepared_dict.items() 
                                if case_id in self.fold_split[self.fold_index]['train'][:][:]}

        self.test_dict = {case_id.split('_')[0]: case_values for case_id, case_values in self.prepared_dict.items() 
                                if case_id.split('_')[0] in self.fold_split[self.fold_index]['test']}
        print('Test case in this fold ', self.test_dict.keys())

        train_dict = dict(list(self.train_val_dict.items())[:int(np.round(len(self.train_val_dict) * (1.0 - self.validation_fraction)))])
        val_dict = dict(list(self.train_val_dict.items())[int(np.round(len(self.train_val_dict) * (1.0 - self.validation_fraction))):])         

        train_patch_instructions = generate_stroke_instructions(train_dict, self.patch_size, self.patch_step, self.do_data_augmentation, self.patches_per_image)
        val_patch_instructions = generate_stroke_instructions(val_dict, self.patch_size, self.patch_step, self.do_data_augmentation, self.patches_per_image)
        test_patch_instructions = generate_stroke_instructions(self.test_dict, self.patch_size, self.patch_step, self.do_data_augmentation, self.patches_per_image)
        self.train_dataset = InstructionDataset(train_patch_instructions, train_dict, extract_stroke_patch)   
        self.val_dataset = InstructionDataset(val_patch_instructions, val_dict, extract_stroke_patch) 
        self.test_dataset = InstructionDataset(test_patch_instructions, self.test_dict, extract_stroke_patch) 

if __name__ == "__main__":
    prepared_data_path = 'path_to_files/'
    NUM_WORKERS = 32

    stroke_dict = load_prepared_trueta_dataset(prepared_data_path)
    StrokeDM = PatchDataModule_wMask_crossval(prepared_data_path=prepared_data_path, 
                                    patch_size=(64,64,8), patch_step=None, do_skull_stripping=False, 
                                    batch_size=16, validation_fraction=0.2, num_workers=NUM_WORKERS, patches_per_image=3000)

#    BrainDM.prepare_data()
    StrokeDM.setup()
