import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from dataset_coordinate_transform import Dataset_Transformation
from sklearn.model_selection import train_test_split

class StreetImageDataModule(pl.LightningDataModule):
    def __init__(self,global_bounding_box, sample_bounding_box, batch_size = 64, data_dir: str="./"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.global_bounding_box = global_bounding_box
        self.sample_bounding_box = sample_bounding_box
    
    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.
        print("hi")

    def setup(self,stage=None):
        # There are also data operations you might want to perform on every GPU. Includes train/test/val splits and transformations
        print("hi")

    def train_dataloader(self):
        print("hi")

    def val_dataloader(self):
        print("hi")

    def test_dataloader(self):
        print("hi")