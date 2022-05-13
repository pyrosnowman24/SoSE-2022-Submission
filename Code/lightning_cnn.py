from json import encoder
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch import nn
import torch
import numpy as np
from torchmetrics import Accuracy
from image_datamodule import RSUIntersectionDataModule
from torch.optim import Adam
from torchmetrics.functional import pairwise_euclidean_distance

class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnv = nn.Conv2d(5,64,3,padding = 'same')
        self.rel = nn.ReLU()
        self.bn = nn.BatchNorm2d(64)
        self.mxpool = nn.MaxPool2d(4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(496000,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,2)
        self.softmax = nn.Softmax()

    def forward(self,x):
        out = self.bn(self.rel(self.cnv(x)))
        out = self.flat(self.mxpool(out))
        out = self.rel(self.fc1(out))
        out = self.rel(self.fc2(out))
        out = self.fc3(out)
        return out


class RSU_Placement_Model(LightningModule):
    def __init__(self,network):
        super().__init__()
        self.net = network.double()

    def forward(self,x):
        out = self.net(x)
        return out

    def loss_fn(self,out,target):
        loss = nn.functional.mse_loss(out,target)
        return loss
    
    def configure_optimizers(self):
        LR = 1e-3
        optimizer = Adam(self.parameters(),lr=LR)
        return optimizer

    def training_step(self,batch,batch_id):
        x_act = batch["solution1"]
        y_act = batch["solution2"]
        img_1 = np.asarray(batch["map"])
        img_2 = np.expand_dims(batch["building"],axis = 3)
        img_3 = np.expand_dims(batch["road"], axis = 3)
        
        coords_act = np.vstack((x_act,y_act))
        coords_act = torch.tensor(np.reshape(coords_act,(coords_act.shape[1],coords_act.shape[0])))

        model_input = np.concatenate((img_1,img_2,img_3),axis=3)
        model_input = np.reshape(model_input,(model_input.shape[0],model_input.shape[-1],model_input.shape[1],model_input.shape[2]))
        model_input = torch.tensor(model_input)

        out = self(model_input)
        loss = self.loss_fn(out,coords_act)
        self.log('train_loss', loss)
        return loss       

    def test_step(self,batch,batch_id):
        x_act = batch["solution1"]
        y_act = batch["solution2"]
        img_1 = np.asarray(batch["map"])
        img_2 = np.expand_dims(batch["building"],axis = 3)
        img_3 = np.expand_dims(batch["road"], axis = 3)

        coords_act = np.vstack((x_act,y_act))
        coords_act = torch.tensor(np.reshape(coords_act,(coords_act.shape[1],coords_act.shape[0])))
        
        model_input = np.concatenate((img_1,img_2,img_3),axis=3)
        model_input = np.reshape(model_input,(model_input.shape[0],model_input.shape[-1],model_input.shape[1],model_input.shape[2]))
        model_input = torch.tensor(model_input)

        out = self(model_input)
        loss = self.loss_fn(out,coords_act)
        accuracy = pairwise_euclidean_distance(out,coords_act)

        metrics = {"test_acc": accuracy,"test_loss": loss}
        self.log("test_acc", accuracy)
        self.log("test_loss", loss)
        return metrics

rsu_cnn_model = RSU_Placement_Model(network=CnnModel())
trainer = Trainer(max_epochs=10,default_root_dir="/home/acelab/Dissertation/Map_dataset_script/checkpoints")
rsu_data = RSUIntersectionDataModule(csv_file = "/home/acelab/Dissertation/Map_dataset_script/Datasets/Austin_downtown/data.csv",
                                     root_dir = "/home/acelab/Dissertation/Map_dataset_script/Datasets/Austin_downtown/",
                                     batch_size = 5)
trainer.fit(rsu_cnn_model,datamodule = rsu_data)

trainer.test(rsu_cnn_model,datamodule=rsu_data)