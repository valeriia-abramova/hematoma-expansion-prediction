import json

import numpy as np
from PatchDataModule import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# from Prediction_stroke_lesion.GrowthPrediction.model import FullModel
import torch
from model1 import *

torch.cuda.empty_cache()

prepared_data_path = '/home/valeria/Prediction_stroke_lesion/data/Synthetic/'
test_path = '/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/data/'
results_path = '/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/004-patchBalanced80-ppi200-adam0001-bs16-l1loss-ps1281288/results/'
experiment_path = '/home/valeria/Prediction_stroke_lesion/SynthesisGrowth/004-patchBalanced80-ppi200-adam0001-bs16-l1loss-ps1281288/experiment/'
MAX_EPOCHS = 100
PATIENCE = 20
subfolder = ['lightning_logs','Model_checkpoints']
for subf in subfolder:
    if not os.path.isdir(experiment_path + subf):
        os.mkdir(experiment_path + subf)

features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


StrokeDM = PatchDataModule(prepared_data_path=prepared_data_path, test_path=test_path,
                                patch_size=(128,128,8), patch_step=(16,16,16), do_skull_stripping=False, 
                                batch_size=16, validation_fraction=0.2, num_workers=12, 
                                do_data_augmentation=False, patches_per_image=200)

# loss function

def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3],
                                     labels.size()[4]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class SimLoss(nn.Module):
    '''
    Intensity mean squared error loss
    '''
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean((target - output) ** 2)

        return loss

class SmoothLoss(nn.Module):
    '''
    Smooth DVF regularization to avoid the unrealistic image generation
    Usually it is a spatial gradient of the DVF
    '''
    """
    N-D gradient loss.
    """

    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, output):
        dy = torch.abs(output[:, :, 1:, :, :] - output[:, :, :-1, :, :])
        dx = torch.abs(output[:, :, :, 1:, :] - output[:, :, :, :-1, :])
        dz = torch.abs(output[:, :, :, :, 1:] - output[:, :, :, :, :-1])
        # Return tensors with same size as original image by concatenating zeros.
        dy = torch.cat((dy,torch.zeros(output.size(dim=0),output.size(dim=1),1,output.size(dim=3),output.size(dim=4)).to(device='cuda')),dim=2)
        dx = torch.cat((dx,torch.zeros(output.size(dim=0),output.size(dim=1),output.size(dim=2),1,output.size(dim=4)).to(device='cuda')),dim=3)
        dz = torch.cat((dz,torch.zeros(output.size(dim=0),output.size(dim=1),output.size(dim=2),output.size(dim=3),1).to(device='cuda')),dim=4)


        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        return torch.tensor(grad)

my_loss = torch.nn.L1Loss()
smooth_loss = SmoothLoss()
my_dvfLoss = lambda output: smooth_loss(output)
my_simloss = lambda output, target: my_loss(output,target)



image_measures = {}


# StrokeDM.set_fold()
StrokeDM.setup()

logger = TensorBoardLogger(experiment_path + 'lightning_logs/', log_graph=True )

early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            patience=PATIENCE,
                                            min_delta=1e-7,
                                            verbose=True,
                                            mode='min')

checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(experiment_path,'Model_checkpoints'),
                                    filename='trueta' + '-{epoch:02d}',
                                    monitor='val_loss',
                                    mode='min',
                                    verbose=False)

pl.seed_everything(0, workers=True)

model = FullModel(1,3,my_dvfLoss,my_simloss)

trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                    accelerator='gpu',
                    devices=[0], 
                    callbacks=[early_stopping_callback, checkpoint_callback],
                    deterministic=False,
                    fast_dev_run=False, 
                    enable_model_summary=False,
                    logger=logger)

trainer.fit(model, StrokeDM)

# model = FullModel.load_from_checkpoint('/home/valeria/Prediction_stroke_lesion/GrowthPrediction/DVFexperiment/Model_checkpoints/fold0/trueta-epoch=00.ckpt',in_channels = 1, out_channels = 3, unet_loss = my_CombinedLoss)
# StrokeDM.setup(stage='test')
# model.unet.register_forward_hook(get_features('unet'))

test_cases = StrokeDM.get_test_cases()
model.return_activated_output = False
image_measures.update(
    model.infer_test_images(test_cases=test_cases, 
                            Stroke_DM=StrokeDM, 
                            filepath_out=results_path))
# output = model(test_cases[0])
# print(features['unet'].shape)


with open(os.path.join(results_path, 'image_measures.json'), 'w') as f:
        json.dump(image_measures, f, indent=2)