import numpy as np
import pandas as pd
from PIL import Image
from dataset_coordinate_transform import Dataset_Transformation
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os
import glob
from sklearn import metrics




class Model_Test():
    def __init__(self,path):
        self.model = keras.models.load_model(path)
        self.import_files()
        
    def __call__(self):
        test_idx = p = np.random.permutation(len(self.df))
        test_batch_size = len(test_idx)
        test_generator = self.generate_images(test_idx, batch_size=test_batch_size, is_training=False)
        x_pred, y_pred = self.model.predict(test_generator, steps=len(test_idx)//test_batch_size)
        test_generator = self.generate_images(test_idx, batch_size=test_batch_size, is_training=False)
        images, x_true, y_true = [], [], []
        for test_batch in test_generator:
            image = test_batch[0]
            labels = test_batch[1]
            images.extend(image)
            x_true.extend(labels[0])
            y_true.extend(labels[1])
        x_explained_variance = metrics.explained_variance_score(x_true,x_pred) # 1 is best, lower is worse
        y_explained_variance = metrics.explained_variance_score(y_true,y_pred)
        x_mae = metrics.mean_absolute_error(x_true,x_pred) # Lower is better
        y_mae = metrics.mean_absolute_error(y_true,y_pred)
        x_r2 = metrics.r2_score(x_true,x_pred) # 0 is the best
        y_r2 = metrics.r2_score(y_true,y_pred)
        print(x_explained_variance,y_explained_variance)
        print(x_mae,y_mae)
        print(x_r2,y_r2)

    def import_files(self):
        current_path = pathlib.Path().resolve()
        folder_path = 'Map_Dataset_Generator/Datasets'
        dataset_name = 'Thu 05 Aug 2021 03:08:35 PM '
        map_name = 'Map'
        file_path = os.path.join(current_path,folder_path)
        self.dataset_path = os.path.join(file_path,dataset_name)
        map_path = os.path.join(self.dataset_path,map_name)
        data_name = "data.csv"
        data_file_path = os.path.join(self.dataset_path,data_name)
        images_folder = 'Images/samples'
        images_path = os.path.join(self.dataset_path,images_folder)
        self.data = pd.read_csv(data_file_path)
        self.data_array = self.data.to_numpy()
        self.data_array = self.transforms.prepare_dataset(self.data_array)
        self.df = pd.DataFrame(self.data_array[:,-2:],columns = ['x','y'])
        files = glob.glob(os.path.join(images_path, "*.%s" % 'png'))
        self.df['file'] = files

    def generate_images(self, image_idx, batch_size=32, is_training=True):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        # arrays to store our batched data
        images, x_array, y_array, = [], [], []
        while True:
            for idx in image_idx:
                sample = self.df.iloc[idx]
                
                x_coord = sample['x']
                y_coord = sample['y']
                file = sample['file']
                im = self.preprocess_image(file)
                
                x_array.append(x_coord)
                y_array.append(y_coord)
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(x_array), np.array(y_array)]
                    images, x_array, y_array, = [], [], []
            if not is_training:
                break

test_bed = Model_Test()
test_bed()