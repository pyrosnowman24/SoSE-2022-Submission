import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.sparse import coo
from tensorflow.python.autograph.utils.ag_logging import _output_to_stdout
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_coordinate_transform import Dataset_Transformation
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import glob
from sklearn import metrics
import re

class Model_Test():
    def __init__(self,dataset_name,model_folder):
        self.import_files(dataset_name,model_folder)
        
    def __call__(self,number_tests = 2):
        self.plot_solution(num_tests=number_tests)

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
        map_images_folder = 'Images/map_images'
        roads_images_folder = 'Images/map_images'
        buildings_images_folder = 'Images/map_images'
        map_images_path = os.path.join(self.dataset_path,map_images_folder)
        roads_images_path = os.path.join(self.dataset_path,roads_images_folder)
        buildings_images_path = os.path.join(self.dataset_path,buildings_images_folder)

        # Create transforms class
        self.world_img = Image.open(map_path)
        self.bbox =  -97.7907, 30.2330, -97.6664, 30.3338 # Austin Downtown
        self.data_size = [250,500]
        self.transforms = Dataset_Transformation(self.bbox,self.world_img.size,self.data_size)

        # Import data
        self.data = pd.read_csv(data_file_path)
        self.data_array = self.data.to_numpy()
        self.data_array = self.transforms.prepare_dataset(self.data_array)
        self.df = pd.DataFrame(self.data_array[:,-2:],columns = ['x','y'])

        map_files = glob.glob(os.path.join(map_images_path, "*.%s" % 'png'))
        map_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.df['map_file'] = map_files

        roads_files = glob.glob(os.path.join(roads_images_path, "*.%s" % 'png'))
        roads_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.df['road_file'] = map_files

        buildings_files = glob.glob(os.path.join(buildings_images_path, "*.%s" % 'png'))
        buildings_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.df['building_file'] = map_files

        # Import Model
        model_path = os.path.join(self.dataset_path, model_folder)
        self.model = keras.models.load_model(model_path)

    def generate_images(self, image_idx, batch_size=32, is_training=True):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        # arrays to store our batched data
        combined, images, buildings, roads, x_array, y_array, = [], [], [], [], [], []
        while True:
            for idx in image_idx:
                sample = self.df.iloc[idx]
                
                x_coord = sample['x']
                y_coord = sample['y']
                map = sample['map_file']
                im_map = self.preprocess_image(map,rgb=True)
                building = sample['building_file']
                im_building = self.preprocess_image(building,gray_scale=True)
                im_building = np.reshape(im_building,(self.data_size[1],self.data_size[0],1))
                road = sample['road_file']
                im_roads = self.preprocess_image(road,gray_scale=True)
                im_roads = np.reshape(im_roads,(self.data_size[1],self.data_size[0],1))
                
                x_array.append(x_coord)
                y_array.append(y_coord)
                images.append(im_map)
                buildings.append(im_building)
                roads.append(im_roads)
                combined = np.concatenate((images,buildings,roads),axis = 3)
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(combined), [np.array(x_array), np.array(y_array)]
                    combined, images, buildings, roads, x_array, y_array, = [], [], [], [], [], []
            if not is_training:
                break

    def preprocess_image(self, img_path, gray_scale = False, rgb = False):
        im = Image.open(img_path)
        im = im.resize((self.data_size[0], self.data_size[1]))
        if rgb : im = im.convert('RGB')
        if gray_scale: im = ImageOps.grayscale(im)
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

    def plot_solution(self,num_tests = 2):
        test_idx = np.random.permutation(len(self.df))[0:num_tests]
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

        print("output coords:")
        print(pred_output)
        print(true_output)

        pred_samples = self.transforms.output_to_sample(np.copy(pred_output))
        true_samples = self.transforms.output_to_sample(np.copy(true_output))

        print("sample coords:")
        print(pred_samples)
        print(true_samples)

        for i in range(test_batch_size):
            boundry_coordinates = self.data_array[test_idx[i],1:9]
            boundry_coordinates = np.reshape(boundry_coordinates,(4,2))
            angle = self.data_array[test_idx[i],11]  
            coordinates = self.data_array[test_idx[i],-2:]
            print("Correct Solution (Output coordinate):")
            print(coordinates)
            coordinate_map = self.transforms.output_to_map(coordinates,boundry_coordinates,angle)
            print("Correct Solution (Map Coordinate)")
            print(coordinate_map)

            coordinate_solution = self.transforms.map_to_sample(coordinate_map,boundry_coordinates,angle)
            print(coordinate_solution)

            pred_map = self.transforms.sample_to_map(pred_samples[i,:],boundry_coordinates,angle)
            fig,(ax1,ax2,ax3) = plt.subplots(1,3)
            ax1.scatter(coordinate_map[:,0],coordinate_map[:,1],color = 'b',label = 'Solution')
            # ax1.scatter(pred_map[0][0],pred_map[0][1],color = 'r',label = 'Prediction')
            ax1.scatter(boundry_coordinates[:,0],boundry_coordinates[:,1])
            ax1.imshow(self.world_img,origin = 'lower')

            ax2.imshow(images[i][:,:,0:3],origin = 'upper')
            # ax2.scatter(pred_samples[i,0],pred_samples[i,1],color = 'r',label = 'Prediction')
            ax2.scatter(true_samples[i,0],true_samples[i,1],color = 'b',label = 'Solution')
            ax2.set_title(angle * 180 / np.pi)
            ax2.invert_yaxis()
            ax2.invert_xaxis()

            ax3.imshow(images[i][:,:,0:3],origin = 'upper')
            # ax2.scatter(pred_samples[i,0],pred_samples[i,1],color = 'r',label = 'Prediction')
            ax3.scatter(true_samples[i,0],true_samples[i,1],color = 'b',label = 'Solution')
            ax3.set_title(angle * 180 / np.pi)
            ax3.invert_yaxis()

            ax1.legend()
            plt.show()

dataset_name = 'Tue 26 Oct 2021 03:35:35 PM '
model_folder = 'batch_128'
test_bed = Model_Test(dataset_name,model_folder)
test_bed(1)