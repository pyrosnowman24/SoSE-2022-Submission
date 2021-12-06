from operator import index
from PIL.Image import FLIP_TOP_BOTTOM
from PIL import Image
from numpy.core.fromnumeric import diagonal
from numpy.core.numeric import NaN
import numpy as np
import matplotlib.pyplot as plt
import geotiler
import pandas as pd
import pathlib
import os
import glob
from dataset_coordinate_transform import Dataset_Transformation

class Data_Plotter():
    def __init__(self,dataset_name,bbox,data_size):
        self.bbox = bbox
        self.data_size = data_size
        self.import_files(dataset_name)
    
    def __call__(self,index_array):
        images = self.generate_images(index_array)
        for i in range(len(index_array)):
            index = index_array[i]
            image = images[i]
            data = self.df.iloc[index]
            coordinates = data[:8].to_numpy()
            coordinates = np.reshape(coordinates,(4,2))
            solution = data[-3:-1].to_numpy()
            angle = data[-4]
            self.plot_data(image,coordinates,solution,angle)
        return None

    def import_files(self,dataset_name):
        # Set all file paths
        current_path = pathlib.Path().resolve()
        folder_path = 'Map_Dataset_Generator/Datasets'
        map_name = 'Map'
        file_path = os.path.join(current_path,folder_path)
        self.dataset_path = os.path.join(file_path,dataset_name)
        map_path = os.path.join(self.dataset_path,map_name)
        data_name = "data.csv"
        data_file_path = os.path.join(self.dataset_path,data_name)
        images_folder = 'Images/samples'
        images_path = os.path.join(self.dataset_path,images_folder)

        # Import world map
        self.world_img = Image.open(map_path)

        # Create transforms class
        self.transforms = Dataset_Transformation(self.bbox,self.world_img.size,self.data_size)

        # Import data.csv file
        self.data = pd.read_csv(data_file_path)
        self.data_array = self.data.to_numpy()
        self.data_array = self.transforms.prepare_dataset(self.data_array)
        self.df = pd.DataFrame(self.data_array[:,1:],columns=["cord1","cord2","cord3","cord4","cord5","cord6","cord7","cord8","center1","center2","angle","solution1","solution2"])

        # Import images
        
        files = glob.glob(os.path.join(images_path, "*.%s" % 'png'))
        self.df['file'] = files

    def plot_data(self,image,coords,solution,angle): # plots the map, the sample area, the intersections, and the solution
            fig,ax = plt.subplots(1)
            solution = self.transforms.output_to_map(solution,coords,angle)
            ax.scatter(coords[:, 0], coords[:, 1], marker='x',c = 'r')
            ax.scatter(solution[0],solution[1],marker='o',color = 'g',zorder = 1)
            ax.imshow(self.world_img,origin = 'lower')
            ax.set_title('Intersections in San Antonio')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.axis('equal')
            plt.show()

    def preprocess_image(self, img_path):
            im = Image.open(img_path)
            im = im.resize((self.data_size[0], self.data_size[1]))
            return im

    def generate_images(self, image_idx):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        # arrays to store our batched data
        images = []
        while True:
            for idx in image_idx:
                sample = self.df.iloc[idx]
                
                file = sample['file']
                im = self.preprocess_image(file)
                
                images.append(im)

            return images

dataset_name = 'Thu 26 Aug 2021 03:29:43 PM '
bbox =  -98.5149, 29.4441, -98.4734, 29.3876 # San Antonio Downtown
data_size = [250,500]
plotter = Data_Plotter(dataset_name,bbox,data_size)
index_array = np.array((1684,3999))
plotter(index_array)