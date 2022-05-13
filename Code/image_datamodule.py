import csv
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
from PIL import Image, ImageOps
import numpy as np
from dataset_coordinate_transform import Dataset_Transformation

import matplotlib.pyplot as plt

class RSUIntersectionDataset(Dataset):
    def __init__(self,csv_file,root_dir):
        self.rsu_intersections = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.road_dir = os.path.join(root_dir,"Images/road_images/")
        self.building_dir = os.path.join(root_dir,"Images/building_images/")
        self.map_dir = os.path.join(root_dir,"Images/map_images/")

        variables_csv = os.path.join(root_dir,"variables.csv")
        variables = pd.read_csv(variables_csv).to_numpy()
        self.data_size = variables[0][5:7].astype(int)
        self.global_bounds = variables[0][1:5]
        world_img = Image.open(os.path.join(root_dir,'Map'))
        self.transforms = Dataset_Transformation(self.global_bounds,world_img.size,self.data_size)

    def __len__(self):
        return len(self.rsu_intersections)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.rsu_intersections.iloc[idx]
        data[:-3] = self.preprocess_data(data[:-3])
        map_image_name = os.path.join(self.map_dir,data['map_image'])
        map_image = Image.open(map_image_name)

        road_image_name = os.path.join(self.road_dir,data['road_image'])
        road_image = Image.open(road_image_name)

        building_image_name = os.path.join(self.building_dir,data['building_image'])
        building_image = Image.open(building_image_name)

        data = data[1:].to_dict()
        data["map"] = self.preprocess_image(map_image,rgb=True)
        data["building"] = self.preprocess_image(building_image,gray_scale=True)
        data["road"] = self.preprocess_image(road_image,gray_scale=True)

        return data

    def preprocess_image(self, image, gray_scale = False, rgb = False):
        """Preprocesses images before they are used to train the model.

        Parameters
        ----------
        imgage : Pillow Image
            Image to be preprocessed.
        gray_scale : bool, optional
            Flag to determine if the image should be converted to greyscale, by default False
        rgb : bool, optional
            Flag to determine if the image should be converted to RGB, by default False

        Returns
        -------
         : PIL.image.image
            Preprocessed image.
        """
        im = image.resize((self.data_size[0], self.data_size[1]))
        if rgb : im = im.convert('RGB')
        if gray_scale: im = ImageOps.grayscale(im)
        im = np.array(im) / 255.0
        return im

    def preprocess_data(self,dataset):
        return self.transforms.prepare_data(dataset)

class RSUIntersectionDataModule(pl.LightningDataModule):
    def __init__(self,csv_file: str = "/home/acelab/Dissertation/Map_dataset_script/Datasets/Austin_downtown/data.csv", root_dir: str = "/home/acelab/Dissertation/Map_dataset_script/Datasets/Austin_downtown/", batch_size: int = 25):
        super().__init__()
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_test_split = .7
        self.setup()

    def setup(self):
        self.rsu_database = RSUIntersectionDataset(csv_file = self.csv_file, root_dir = self.root_dir)
        train_size = int(self.train_test_split*len(self.rsu_database))
        test_size = len(self.rsu_database) - train_size
        self.train_dataset,self.test_dataset = torch.utils.data.random_split(self.rsu_database,[train_size,test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size)
