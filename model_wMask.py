import json
import os

import SimpleITK as sitk
import nibabel as nib
import niclib as nl
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PatchDataModule_wMask import *
from monai.inferers import sliding_window_inference
from torch.nn import functional as F
from torch.utils.data import DataLoader


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(Conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.conv(x)

        return x



class MICCAI_model(pl.LightningModule):
    """
    The encoder-decoder model for hematoma growth prediction based on paper from miccai 2021
    but without the clinical information fusion model in the bottleneck, instead there will be
    ordinary Unet bottleneck

    Xiao, T. et al. (2021). Intracerebral Haemorrhage Growth Prediction Based on Displacement Vector Field and Clinical Metadata. 
    In: , et al. Medical Image Computing and Computer Assisted Intervention - MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science(), 
    vol 12905. Springer, Cham. https://doi.org/10.1007/978-3-030-87240-3_71

    output_channels = 3

    """

    def __init__(self, input_channels, output_channels, loss, return_activated_output=False):
        # output channels should be three
        # output without activation function!

        # super(MICCAI_model, self).__init__()
        # self.loss = loss
        # self.return_activated_output = return_activated_output
        # self.activation = nn.Sigmoid()

        # self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)

        # self.Conv1 = Conv_block(in_channels=input_channels, out_channels=64)
        # self.Conv2 = Conv_block(in_channels=64, out_channels=128)
        # self.Conv3 = Conv_block(in_channels=128, out_channels=256)
        # self.Conv4 = Conv_block(in_channels=256, out_channels=512)

        # self.Bottleneck = Conv_block(in_channels=512, out_channels=1024)

        # self.Up1 = nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        # self.UpConv1 = Conv_block(in_channels=1024, out_channels=512)

        # self.Up2 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # self.UpConv2 = Conv_block(in_channels=512, out_channels=256)

        # self.Up3 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        # self.UpConv3 = Conv_block(in_channels=256, out_channels=128)

        # self.Up4 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        # self.UpConv4 = Conv_block(in_channels=128, out_channels=64)

        # self.Conv5 = nn.Conv3d(in_channels=64, out_channels=output_channels, kernel_size=1)

        super(MICCAI_model, self).__init__()
        self.loss = loss
        self.return_activated_output = return_activated_output
        # self.activation = nn.Sigmoid()

        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = Conv_block(in_channels=input_channels, out_channels=64)
        self.Conv2 = Conv_block(in_channels=64, out_channels=128)
        self.Conv3 = Conv_block(in_channels=128, out_channels=256)


        self.Bottleneck = Conv_block(in_channels=256, out_channels=512)

        self.Up1 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.UpConv1 = Conv_block(in_channels=512, out_channels=256)

        self.Up2 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.UpConv2 = Conv_block(in_channels=256, out_channels=128)

        self.Up3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.UpConv3 = Conv_block(in_channels=128, out_channels=64)

        self.Conv5 = nn.Conv3d(in_channels=64, out_channels=output_channels, kernel_size=1)


    def forward(self, x):
        # encoder
        # x = x.float()
        # x1 = self.Conv1(x)

        # x2 = self.MaxPool(x1)
        # x2 = self.Conv2(x2)

        # x3 = self.MaxPool(x2)
        # x3 = self.Conv3(x3)

        # x4 = self.MaxPool(x3)
        # x4 = self.Conv4(x4)

        # x5 = self.MaxPool(x4)
        # x5 = self.Bottleneck(x5)

        # up1 = self.Up1(x5)
        # up1 = torch.cat((x4,up1),dim=1)
        # up1 = self.UpConv1(up1)

        # up2 = self.Up2(up1)
        # up2 = torch.cat((x3,up2),dim=1)
        # up2 = self.UpConv2(up2)

        # up3 = self.Up3(up2)
        # up3 = torch.cat((x2,up3),dim=1)
        # up3 = self.UpConv3(up3)

        # up4 = self.Up4(up3)
        # up4 = torch.cat((x1,up4),dim=1)
        # up4 = self.UpConv4(up4)

        # out = self.Conv5(up4)

        x = x.float()
        x1 = self.Conv1(x)

        x2 = self.MaxPool(x1)
        x2 = self.Conv2(x2)

        x3 = self.MaxPool(x2)
        x3 = self.Conv3(x3)

        x4 = self.MaxPool(x3)
        x4 = self.Bottleneck(x4)

        up1 = self.Up1(x4)
        up1 = torch.cat((x3,up1),dim=1)
        up1 = self.UpConv1(up1)

        up2 = self.Up2(up1)
        up2 = torch.cat((x2,up2),dim=1)
        up2 = self.UpConv2(up2)

        up3 = self.Up3(up2)
        up3 = torch.cat((x1,up3),dim=1)
        up3 = self.UpConv3(up3)

        out = self.Conv5(up3)

        if self.return_activated_output:
            out = self.activation(out)

        return out

class Regressor(pl.LightningModule):
    """
    The localization network - the first stepp of STN
    It regresses the maps obtained from unet (the dvf) to obtain transformation parameters
    """
    def __init__(self, input_channels, output_channels):
        super(Regressor,self).__init__()

        self.conv = nn.Conv3d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1)


        self.max_pool = nn.MaxPool3d(kernel_size=2, stride = 2)

        # self.fc1 = nn.Linear(3*8*8*8, 12) # multiply channels by the output dimensions
        # self.fc1 = nn.Linear(3*32*32*2, 12) # for input size 128,128,8
        self.fc1 = nn.Linear(3*16*16*16, 12) # for input size 64,64,64
        self.fc2 = nn.Linear(12,output_channels)

        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):

        # x1 = self.conv(x) # the input size is [3,32,32,32] 
        # x2 = self.max_pool(x1)
        # x3 = self.conv(x2)
        # x4 = self.max_pool(x3) # the ouput size is [3,8,8,8]

        # x1 = self.max_pool(x) # [3,48,48,48] [3,24,24,24]
        # x2 = self.max_pool(x1)# [3,24,24,24] [3,12,12,12]
        # x3 = self.max_pool(x2)#[3,12,12,12] [3,6,6,6]

        x1 = self.max_pool(x)# 
        x2 = self.max_pool(x1)

        # x4 = x2.view(-1, 3*8*8*8)
        x4 = x2.view(-1, 3*16*16*16)

        x5 = self.fc1(x4)
        x6 = self.fc2(x5)

        return x6

class FullModel(pl.LightningModule):
    """
    The full framework of predicting the deformation field and warping input to get output image
    It contains MICCAI_model - encoder-decoder model to predict the deformation field
    Regressor to get transformation parameters theta
    Spatial transform network consisting of grid generator and sampler

    """

    def __init__(self,
                    in_channels,
                    out_channels,
                    dvf_loss,
                    sim_loss,
                    seg_loss):
        super(FullModel, self).__init__()
        self.dvf_loss = dvf_loss
        self.sim_loss = sim_loss
        self.seg_loss = seg_loss
        self.unet = MICCAI_model(in_channels, out_channels, self.dvf_loss)
        # Regressor to get theta parameters as 3*4 affine matrix (3*4 because we work in 3D and affine matrix in this case is N*N+1 -> 3*4)
        self.regressor = Regressor(input_channels=out_channels, output_channels=12)
        # Initialize the weights/bias with identity transformation
        # self.regressor.weight.data.zero_()
        # self.regressor.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))


    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.unet(x)
        theta = self.regressor(xs)
        theta = theta.view(-1, 3, 4)

        x = x.type(torch.float32)
        grid = F.affine_grid(theta, x.size(),align_corners=True)
        fin = F.grid_sample(x, grid,align_corners=True)

        return [xs, fin]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def training_step(self, batch, batch_idx):
        # x = image_patch, y = lesion_patch
        x, y = batch 
        y_hat = self.forward(x)
        mask_target = torch.split(y,1,dim=1)[1] #case with mask
        img_target = torch.split(y,1,dim=1)[0]
        mask_output = torch.split(y_hat[1],1,dim=1)[1]
        img_output = torch.split(y_hat[1],1,dim=1)[0]
        dvf_loss = self.dvf_loss(y_hat[0])
        sim_loss = self.sim_loss(img_output, img_target)
        seg_loss = self.seg_loss(mask_output, mask_target)
        loss = dvf_loss + sim_loss + seg_loss
        # if batch_idx == 1:
        #     img_output = img_output.detach().cpu().numpy() # BS, CH, X, Y, Z
        #     mask_output = mask_output.detach().cpu().numpy() # BS, CH, X, Y, Z
        #     columns = []
        #     for xi, yi in zip(mask_output, img_output):
        #         columns.append(np.concatenate([xij for xij in xi] + [yi[0]], axis=0))
        #     grid_arr = np.concatenate(columns, axis=1)
        #     nib.Nifti1Image(grid_arr, np.eye(4)).to_filename('/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/018-patchBalanced80-ppi500-adam00001-bs32-l1loss-ps323232-border-mask/results/batches/batch1_epoch{}.nii.gz'.format(self.current_epoch))
        # dvf_loss = self.dvf_loss(y_hat[0])
        # sim_loss = self.sim_loss(y_hat[1], y)
        # loss = 0.01*dvf_loss + sim_loss
        self.log_dict({'train_loss': loss, 'dvf_loss' : dvf_loss, 'sim_loss' : sim_loss, 'seg_loss': seg_loss}, on_epoch=True) 
                
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x) 
        mask_target = torch.split(y,1,dim=1)[1] #case with mask
        img_target = torch.split(y,1,dim=1)[0]
        mask_output = torch.split(y_hat[1],1,dim=1)[1]
        img_output = torch.split(y_hat[1],1,dim=1)[0]
        dvf_loss = self.dvf_loss(y_hat[0])
        sim_loss = self.sim_loss(img_output, img_target)
        seg_loss = self.seg_loss(mask_output, mask_target)
        loss = dvf_loss + sim_loss + seg_loss
        # dvf_loss = self.dvf_loss(y_hat[0])
        # sim_loss = self.sim_loss(y_hat[1], y)
        # loss = 0.01*dvf_loss + sim_loss 
        
        # self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({'val_loss': loss, 'val_dvf_loss' : dvf_loss, 'val_sim_loss' : sim_loss, 'val_seg_loss': seg_loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'val_loss': loss}  
                    
    def infer_test_images(self, test_cases, Stroke_DM, filepath_out):
        os.makedirs(filepath_out, exist_ok=True)
        image_measures = {}
        for n, case_folder in enumerate(test_cases):
            case_num = case_folder.split('/')[-1]
            print(" " * 50, end='\r') # Erase the last line
            print(f"> Segmenting test case {case_num} ({n+1}/{len(test_cases)})")
            

            img = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_V8/{}.nii.gz'.format(case_num)))).get_fdata().astype(np.float16)
            # img = (img - np.min(img))/(np.max(img) - np.min(img))
            mask = np.round(nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_V8_mask_interp/{}.nii.gz'.format(case_num)))).get_fdata()).astype(np.uint8)
            # img = np.expand_dims(img, axis=0)

            norm_params = find_normalization_parameters(img)
            norm_img = normalize_image(img, norm_params)

            norm_img = np.stack([norm_img, mask], axis = 0)
            norm_img = np.expand_dims(norm_img, axis=0)

            device = torch.device('cuda')

            self.eval()
            self.cuda()
            with torch.no_grad():
                prediction = sliding_window_inference(inputs=torch.tensor(norm_img).to(device), 
                                                    roi_size=Stroke_DM.patch_size, 
                                                    sw_batch_size=Stroke_DM.batch_size,
                                                    # device=torch.cuda._device,
                                                    # sw_device=torch.device('cuda'),
                                                    predictor=self,
                                                    overlap=0.6, progress=True, mode='gaussian') # Self calls the forward method of the model


            # Convert to numpy and Round to save disk space
            # prediction1 = np.round(prediction[0, 1, :, :, :].detach().cpu().numpy(), decimals=4)
            prediction1 = prediction[1][0,0,:,:,:].detach().cpu().numpy() # the final prediction
            prediction_mask = prediction[1][0,1,:,:,:].detach().cpu().numpy()
            prediction_mask = np.where(prediction_mask > 0.5, 1, 0)
            prediction2 = prediction[0].permute(2,3,4,0,1).detach().cpu().numpy() # the dvf
            prediction_mask[img == 0.0] = 0.0
            prediction1[img == 0.0] = 0.0
     

            # prediction1 = (prediction1 - np.min(prediction1))/(np.max(prediction1) - np.min(prediction1))
            # prediction2 = sitk.GetImageFromArray(prediction2)
            
            nifti_img = nib.funcs.as_closest_canonical(nib.load(os.path.join('/home/valeria/Prediction_stroke_lesion/data/Basal_to_FU1_V8/{}.nii.gz'.format(case_num))))
            nib.Nifti1Image(prediction1, nifti_img.affine, nifti_img.header).to_filename(
                os.path.join(os.path.join(filepath_out, case_num) + '_prob.nii.gz'))
            

            nib.Nifti1Image(prediction2, nifti_img.affine, nifti_img.header).to_filename(
                os.path.join(os.path.join(filepath_out, case_num) + '_dvf.nii.gz'))

            nib.Nifti1Image(prediction_mask, nifti_img.affine, nifti_img.header).to_filename(
                os.path.join(os.path.join(filepath_out, case_num) + '_mask.nii.gz'))

            
            # image_measures[case_num] = Stroke_DM.compute_image_measures(case_num=case_num, inference_result=np.argmax(prediction[0,:,:,:,:].numpy(),axis=0), 
            #                                                                  header=nifti_img.header)
            
        # print("")
        # with open(os.path.join(filepath_out, 'measures.json'), 'w') as f:
        #     json.dump(image_measures, f, indent=2)

        return image_measures