import numpy as np
import pandas as pd
import os
from PIL import Image
from dataset_coordinate_transform import Dataset_Transformation

# Import world map
path = '/home/acelab/Scripts/Map_dataset_script/Datasets/Mon Jul 19 17:23:22 2021'
name = 'Map'
file_path = os.path.join(path,name)
world_img = Image.open(file_path)

# Create transforms class
bbox =  -98.5149, 29.4441, -98.4734, 29.3876 # San Antonio Downtown
data_size = [250,500]
transforms = Dataset_Transformation(bbox,world_img.size,data_size)

# Import data.csv file
data_name = "data.csv"
data_file_path = os.path.join(path,data_name)
data = pd.read_csv(data_file_path)
data_array = data.to_numpy()
