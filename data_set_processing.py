import numpy as np
import pandas as pd
import os
from PIL import Image
from dataset_coordinate_transform import Dataset_Transformation
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib

# Import world map
current_path = pathlib.Path().resolve()
folder_path = 'Map_Dataset_Generator/Datasets'
dataset_name = 'Thu 05 Aug 2021 03:08:35 PM '
map_name = 'Map'
file_path = os.path.join(current_path,folder_path)
dataset_path = os.path.join(file_path,dataset_name)

map_path = os.path.join(dataset_path,map_name)
world_img = Image.open(map_path)

# Create transforms class
bbox =  -98.5149, 29.4441, -98.4734, 29.3876 # San Antonio Downtown
data_size = [250,500]
transforms = Dataset_Transformation(bbox,world_img.size,data_size)

# Import data.csv file
data_name = "data.csv"
data_file_path = os.path.join(dataset_path,data_name)
data = pd.read_csv(data_file_path)
data_array = data.to_numpy()

data_array = transforms.prepare_dataset(data_array)

# Import images
images_folder = 'Images'
images_path = os.path.join(dataset_path,images_folder)
datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_it = datagen.flow_from_directory(images_path, classes = None, color_mode='rgb', target_size=data_size, batch_size=64,save_format="png")

# Convert coordinates to correct format

# define cnn model
def define_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(*data_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    # compile model
    opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    return model

model = define_model()
history = model.fit_generator(train_it,data_array[:,12:],epochs=50,verbose=1)