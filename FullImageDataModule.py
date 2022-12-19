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
import monai.transforms


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
    # synthetic dataset 174 patients
    cases = [f.path for f in os.scandir(path) if f.is_dir()]
    cases = [case.split('/')[-1] for case in cases]
    cases = [case for case in cases if 'pt080' not in case]
    cases = sorted(cases)
    for case in cases:
        patient = case.split('_')[0]
        basal = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'Basal/basal_augm.nii.gz'))).get_fdata()
        # basalbrainMask = np.expand_dims((nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'Basal/basal_resized.nii.gz'))).get_fdata() > 0).astype(np.int), axis=0)
        # basal_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'Basal/basalMask_resized.nii.gz'))).get_fdata()
        fu1 = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'FU1/fu_augm.nii.gz'))).get_fdata()
        # fu_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'FU1/fuMask_resized.nii.gz'))).get_fdata()
        basal = np.stack([basal], axis = 0)
        fu1 = np.stack([fu1], axis = 0)
        

        prepared_dict[case] = {
            'basal': basal,
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
    cases = ['pt038', 'pt170', 'pt094', 'pt103', 'pt127', 'pt054',
            'pt095','pt057','pt089','pt168','pt113','pt043',
            'pt140','pt108','pt106','pt098','pt082','pt105']
    # cases = cases[:5]
    for case in cases:
        basal = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'Basal/basal_resized.nii.gz'))).get_fdata()
        # basalbrainMask = np.expand_dims((nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'Basal/basal_resized.nii.gz'))).get_fdata() > 0).astype(np.int), axis=0)
        # basal_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'Basal/basalMask_resized.nii.gz'))).get_fdata()
        fu1 = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'FU1/fuMask_resized.nii.gz'))).get_fdata()
        # fu_lesionMask = nib.funcs.as_closest_canonical(nib.load(os.path.join(path,case,'FU1/fuMask_resized.nii.gz'))).get_fdata()
        basal = np.stack([basal], axis = 0)
        fu1 = np.stack([fu1], axis = 0)
        

        prepared_dict[case] = {
            'basal': basal,
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
    return minusone



def generate_stroke_instructions(data: dict, do_data_augmentation: bool):
    """
    For each image a dictionary of instructions
    
    :param data: the dictionary of images and masks
    :param do_data_augmentation: Whether to perform data augmentation on the patches
    :return: A list of dictionaries, each dictionary contains the case_id, do_data_augmentation, and norm_params
    """

    all_instruction = []

    # 3D case
    for case_id, case_dict in data.items(): # For each image (CH, X, Y, Z)

        # Calculate mean and std to perform image normalization
        # norm_parms = find_normalization_parameters(case_dict['basal'][0]) for case with mask
        norm_parms = find_normalization_parameters(case_dict['basal'])
        patch_instruction =[]

        
        patch_instruction += [{'case_id': case_id,
                            'norm_params': norm_parms,
                            'do_data_augmentation': do_data_augmentation}]
        all_instruction+=patch_instruction

    return all_instruction


def extract_stroke_patch(instructions: dict, data: dict):     
    """
    The function takes as input a dictionary containing the instructions for each image, and
    a dictionary containing the data for the case. It then takes the image, normalizes it, performs
    data augmentation on it, and returns the image as a Pytorch tensor.
    
    :param instructions: a dictionary containing the following keys:
    :param data: a dictionary containing the image and lesion mask of the case
    :return: The image_torch and lesion_torch are being returned.
    """

    case = data[instructions['case_id']]
    image = case['basal']
    fu = case['fu1']
    do_data_augmentation = instructions['do_data_augmentation']


    # Create a new variable containing only the extracted image
    image_patch = copy.deepcopy(image)
    fu_patch = copy.deepcopy(fu)

    # perform data augmentation
    if do_data_augmentation:
        affine = monai.transforms.Compose([
            monai.transforms.RandShiftIntensity(offsets=(20), prob=0.5),
            monai.transforms.RandRotate(range_x=[0.0,5.0], range_y=[0.0,5.0], range_z=[0.0,60.0], prob=0.5)

        ])
        concatenated_patches = np.concatenate((image_patch, fu_patch), axis=0)
        concatenated_patches = affine(concatenated_patches)

        num_channels_gt = fu_patch.shape[0]
        image_patch = concatenated_patches[:-num_channels_gt, :, :]
        fu_patch = concatenated_patches[-num_channels_gt:, :, :]


    # Normalize the image_patch
    # image_patch[0] = normalize_image(image_patch[0], instructions['norm_params']) case with mask as additional input channel
    # fu_patch[0] = normalize_image(fu_patch[0], instructions['norm_params'])
    image_patch = normalize_image(image_patch, instructions['norm_params'])
    fu_patch = normalize_image(fu_patch, instructions['norm_params'])
    # Transform the image to a Pytorch tensor
    image_torch = torch.tensor(np.ascontiguousarray(image_patch),dtype=torch.float)
    fu_torch = torch.tensor(np.ascontiguousarray(fu_patch),dtype=torch.float)

    return image_torch, fu_torch


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
        If the reference object is empty.
        
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

class FullImageDataModule(pl.LightningDataModule):
    def __init__(self, prepared_data_path, test_path,do_skull_stripping, 
                batch_size, num_workers,validation_fraction=0.2, num_folds=5, fold_split=None, do_data_augmentation=False):
        super().__init__()
        
        self.prepared_data_path = prepared_data_path
        self.test_path = test_path
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

        if fold_split is not None:
            assert num_folds == len(fold_split)
        self.fold_split = fold_split


    def setup(self, stage='None'):
        if self.prepared_dict is None:
            self.prepared_dict = load_prepared_trueta_dataset(self.prepared_data_path)

        if self.test_dict is None:
            self.test_dict = load_test_dict(self.test_path)
         
        self.set_fold()
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) 
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_test_cases(self):
        return [os.path.join(self.prepared_data_path, case) for case in self.test_dict]
    
    def compute_image_measures(self, case_num, inference_result, header):
        inference_result = np.round(inference_result).astype('int') 
        image_measures = {}
        ground_truth = self.test_dict[case_num]['lesionMask']
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

        # self.train_val_dict = {case_id: case_values for case_id, case_values in self.prepared_dict.items() 
        #                         if case_id in self.fold_split[self.fold_index]['train'][:][:]}

        # self.test_dict = {case_id: case_values for case_id, case_values in self.prepared_dict.items() 
        #                         if case_id in self.fold_split[self.fold_index]['test']}  

        # train_dict = dict(list(self.train_val_dict.items())[:int(np.round(len(self.train_val_dict) * (1.0 - self.validation_fraction)))])
        # val_dict = dict(list(self.train_val_dict.items())[int(np.round(len(self.train_val_dict) * (1.0 - self.validation_fraction))):])         

        self.train_val_dict = {case_id: case_values for case_id, case_values in self.prepared_dict.items()}

        self.test_dict = {case_id: case_values for case_id, case_values in self.test_dict.items()}  

        train_dict = dict(list(self.train_val_dict.items())[:int(np.round(len(self.train_val_dict) * (1.0 - self.validation_fraction)))+2])
        val_dict = dict(list(self.train_val_dict.items())[int(np.round(len(self.train_val_dict) * (1.0 - self.validation_fraction)))+2:])         

        train_patch_instructions = generate_stroke_instructions(train_dict,  self.do_data_augmentation)
        val_patch_instructions = generate_stroke_instructions(val_dict,  self.do_data_augmentation)
        test_patch_instructions = generate_stroke_instructions(self.test_dict,  self.do_data_augmentation)
        self.train_dataset = InstructionDataset(train_patch_instructions, train_dict, extract_stroke_patch)   
        self.val_dataset = InstructionDataset(val_patch_instructions, val_dict, extract_stroke_patch) 
        self.test_dataset = InstructionDataset(test_patch_instructions, self.test_dict, extract_stroke_patch) 

if __name__ == "__main__":
    prepared_data_path = '/home/valeria/MIC3/Prediction_stroke_lesion/HematomaTruetaV7/'
    NUM_WORKERS = 32

    stroke_dict = load_prepared_trueta_dataset(prepared_data_path)
    StrokeDM = FullImageDataModule(original_data_dict=stroke_dict, prepared_data_path=prepared_data_path, 
                                    do_skull_stripping=False, batch_size=8, validation_fraction=0.2, num_workers=NUM_WORKERS)

#    BrainDM.prepare_data()
    StrokeDM.setup()