import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo
from tensorflow.python.autograph.utils.ag_logging import _output_to_stdout
from dataset_coordinate_transform import Dataset_Transformation
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os
import glob
from sklearn import metrics
import re




class Model_Test():
    def __init__(self,dataset_name,model_folder):
        self.import_files(dataset_name,model_folder)
        
    def __call__(self):
        self.plot_solution()

    def import_files(self,dataset_name,model_folder):
        current_path = pathlib.Path().resolve()
        folder_path = 'Datasets'
        map_name = 'Map'
        
        # Create necessary paths for imports
        file_path = os.path.join(current_path,folder_path)
        self.dataset_path = os.path.join(file_path,dataset_name)
        map_path = os.path.join(self.dataset_path,map_name)
        data_name = "data.csv"
        data_file_path = os.path.join(self.dataset_path,data_name)
        images_folder = 'Images/samples'
        images_path = os.path.join(self.dataset_path,images_folder)

        # Create transforms class
        self.world_img = Image.open(map_path)
        self.bbox =  -98.5149, 29.4441, -98.4734, 29.3876 # San Antonio Downtown
        self.data_size = [250,500]
        self.transforms = Dataset_Transformation(self.bbox,self.world_img.size,self.data_size)

        # Import data
        self.data = pd.read_csv(data_file_path)
        self.data_array = self.data.to_numpy()
        self.data_array = self.transforms.prepare_dataset(self.data_array)
        self.df = pd.DataFrame(self.data_array[:,-2:],columns = ['x','y'])
        files = glob.glob(os.path.join(images_path, "*.%s" % 'png'))
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.df['file'] = files

        # Import Model
        model_path = os.path.join(self.dataset_path, model_folder)
        self.model = keras.models.load_model(model_path)

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

    def preprocess_image(self, img_path):
        im = Image.open(img_path)
        im = im.resize((self.data_size[0], self.data_size[1]))
        im = np.array(im) / 255.0
        return im

    def score_model(self):
        test_idx  = np.random.permutation(len(self.df))[0:3000]
        test_batch_size = 300
        test_generator = self.generate_images(test_idx, batch_size=test_batch_size, is_training=True)
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

    def plot_solution(self):
        test_idx = np.random.permutation(len(self.df))[0:2]
        test_batch_size = len(test_idx)
        test_generator = self.generate_images(test_idx, batch_size=test_batch_size, is_training=True)
        x_pred,y_pred = self.model.predict(test_generator,steps=len(test_idx)//test_batch_size)
        test_generator = self.generate_images(test_idx, batch_size=test_batch_size, is_training=False)
        images, x_true, y_true = [], [], []
        for test_batch in test_generator:
            image = test_batch[0]
            labels = test_batch[1]
            images.extend(image)
            x_true.extend(labels[0])
            y_true.extend(labels[1])
        x_true = np.reshape(x_true,(len(x_true),1))
        y_true = np.reshape(y_true,(len(y_true),1))
        pred_output = np.hstack((x_pred,y_pred))
        true_output = np.hstack((x_true,y_true))

        pred_samples = self.transforms.output_to_sample(np.copy(pred_output))
        true_samples = self.transforms.output_to_sample(np.copy(true_output))

        for i in range(test_batch_size):
            boundry_coordinates = self.data_array[test_idx[i],1:9]
            boundry_coordinates = np.reshape(boundry_coordinates,(4,2))
            angle = self.data_array[test_idx[i],11]  
            coordinates = self.data_array[test_idx[i],-2:]
            coordinate_map = self.transforms.output_to_map(coordinates,boundry_coordinates,angle)

            pred_map = self.transforms.sample_to_map(pred_samples[i,:],boundry_coordinates,angle)
            fig,(ax1,ax2) = plt.subplots(1,2)
            ax1.scatter(coordinate_map[:,0],coordinate_map[:,1],color = 'b')
            ax1.scatter(pred_map[0][0],pred_map[0][1],color = 'r')
            ax1.scatter(boundry_coordinates[:,0],boundry_coordinates[:,1])
            ax1.imshow(self.world_img,origin = 'lower')

            ax2.imshow(images[i],origin = 'lower')
            ax2.scatter(pred_samples[i,0],pred_samples[i,1],color = 'r')
            ax2.scatter(true_samples[i,0],true_samples[i,1],color = 'b')
            plt.show()

dataset_name = 'Thu 26 Aug 2021 03:29:43 PM '
model_folder = '32_batch'
test_bed = Model_Test(dataset_name,model_folder)
test_bed()