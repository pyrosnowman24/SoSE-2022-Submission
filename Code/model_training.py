import re
from re import X
import numpy as np
import pandas as pd
import os, sys
from PIL import Image, ImageOps
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_coordinate_transform import Dataset_Transformation
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import glob
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint

class RSU_Placement_CNN():
    """Class to train a CNN to guess the optimal intersection to place an RSU in a area.
    """
    def __init__(self,database_name):
        """Creates a RSU_Placement_CNN class.

        Parameters
        ----------
        database_name : string
            Name of the database that will be used to train the CNN.
        train_test_split : float, optional
            States the train test split for data, by default .7
        """
        self.import_files(database_name)
        # building = self.df.iloc[1]['building']

    def __call__(self, epochs, train_test_split = .7,batch_size = 64, validation_batch_size = 64, init_lr = 1e-4):
        """Trains a CNN model using the dataset based on the provided parameters. The trained model is saved in the dataset's directory along with a training history.

        Parameters
        ----------
        epochs : int
            The number of epochs that the CNN model should be trained for.
        train_test_split : float, optional
            The percentage of the dataset that should be used for training, by default .7
        batch_size : int, optional
            The batch size to use with the training data, by default 64
        validation_batch_size : int, optional
            The batch size to use with the validation data, by default 64
        init_lr : float, optional
            The constant used to determine how quickly the model changes based on each epoch, by default 1e-4
        """
        
        model = self.assemble_full_model(*self.data_size)

        train_idx, valid_idx, test_idx = self.generate_split_indexes(train_test_split) 
        train_gen = self.generate_images(train_idx, batch_size=batch_size)
        valid_gen = self.generate_images(valid_idx, batch_size=validation_batch_size)



        opt = keras.optimizers.Adam(lr=init_lr, decay=init_lr / epochs)
        model.compile(optimizer=opt, 
                    loss={
                        'x_output': 'mse', 
                        'y_output': 'mse'},
                    metrics={
                        'x_output': 'mae', 
                        'y_output': 'mae'})

        model_path = os.path.join(self.dataset_path,"model_checkpoint")

        callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss')
        ]
        history = model.fit(train_gen,
                            steps_per_epoch=len(train_idx)//batch_size,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=valid_gen,
                            validation_steps=len(valid_idx)//validation_batch_size)
        self.plot_history(history,fig_path = model_path)
        var_path = self.create_var_file(model_path)
        self.save_variable_file(var_path,epochs,init_lr,batch_size,validation_batch_size)
        plt.show()

    def import_files(self, database_name):
        """Function that pulls all relevent information from the specified database for the Class to use. The functions imports the images for each sample, imports the .csv file for sample data, imports the map used for the database, and sets up the transform class.

        Parameters
        ----------
        database_name : string
            The name of the database that will be used to train the CNN.
        """
        # Set all file paths
        current_path = pathlib.Path().resolve()
        folder_path = 'Datasets'
        map_name = 'Map'
        file_path = os.path.join(current_path,folder_path)
        self.dataset_path = os.path.join(file_path,database_name)
        map_path = os.path.join(self.dataset_path,map_name)
        data_name = "data.csv"
        variable_name = "variables.csv"
        data_file_path = os.path.join(self.dataset_path,data_name)
        variable_file = os.path.join(self.dataset_path,variable_name)
        images_folder = 'Images/map_images'
        buildings_folder = 'Images/building_images'
        roads_folder = 'Images/road_images'
        images_path = os.path.join(self.dataset_path,images_folder)
        building_path = os.path.join(self.dataset_path,buildings_folder)
        road_path = os.path.join(self.dataset_path,roads_folder)

        # Import Database Variables
        variables = pd.read_csv(variable_file).to_numpy()
        self.bbox = variables[0][1:5]
        self.data_size = variables[0][5:7].astype(int)

        # Import world map
        self.world_img = Image.open(map_path)

        # Create transforms class
        self.transforms = Dataset_Transformation(self.bbox,self.world_img.size,self.data_size)

        # Import data.csv file
        self.data = pd.read_csv(data_file_path)
        data_array = self.data.to_numpy()
        data_array = self.transforms.prepare_dataset(data_array)
        self.df = pd.DataFrame(data_array[:,-2:],columns = ['x','y'])

        # Import images
        files_map = glob.glob(os.path.join(images_path, "*.%s" % 'png'))
        files_map.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.df['map'] = files_map

        files_building = glob.glob(os.path.join(building_path, "*.%s" % 'png'))
        files_building.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.df['building'] = files_building

        files_road = glob.glob(os.path.join(road_path, "*.%s" % 'png'))
        files_road.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.df['road'] = files_road

    def create_var_file(self,folder_path):
        """Creates a .csv to store the variables used to train the CNN

        Parameters
        ----------
        folder_path : string
            Path to the directory the .csv will be stored in.

        Returns
        -------
         : string
            The path to the new .csv file.
        """
        var_file = os.path.join(folder_path, "cnn_variables.csv")
        datas = open(var_file,"w+")
        datas.close()
        return var_file

    def save_variable_file(self,var_file,num_epochs,lr,batch,val_batch):
        """Saves the variables used to train the model to the variable.csv file.

        Parameters
        ----------
        var_file : string
            Path to the variable.csv file.
        num_epochs : int
            The number of epochs used to train the model.
        lr : float
            The learning rate of the model.
        batch : int
            The batch size used to train the model.
        val_batch : int
            The validation batch size used to train the model.
        """
        vars = pd.DataFrame(None,columns=["num_epochs","lr","batch","val_batch"])
        vars.loc[0] = np.array((num_epochs,lr,batch,val_batch)).tolist()
        vars.to_csv(var_file)

    def plot_history(self,history,fig_path = None):
        """Plots the training history of the model and can save it.

        Parameters
        ----------
        history : ndarray
            Array containing the model history.
        fig_path : string, optional
            Path to the directory that the figure should be saved, by default None
        """
        fig, (ax1,ax2) = plt.subplots(2,figsize=(15,10))
        ax1.plot(history.history['x_output_mae'])
        ax1.plot(history.history['val_x_output_mae'])
        ax1.legend(["Train","Validate"])
        ax1.set_ylabel("Mean Absolute Error")
        ax1.set_xlabel("Epochs")
        ax1.set_title("X Position")

        ax2.plot(history.history['y_output_mae'])
        ax2.plot(history.history['val_y_output_mae'])
        ax2.legend(["Train","Validate"])
        ax2.set_ylabel("Mean Absolute Error")
        ax2.set_xlabel("Epochs")
        ax2.set_title("Y Position")
        
        if fig_path is not None:
            fig_name = os.path.join(fig_path,"train_history.png")
            plt.savefig(fig_name)

        plt.draw()

    def generate_split_indexes(self,train_test_split):
        """Separates the database into training, validation, and testing datasets.

        Parameters
        ----------
        train_test_split : float
            The percentage of the database that should be used to train the model.

        Returns
        -------
        train_idx : ndarray
            An array of indexes for the samples that are in the training dataset.
        valid_idx : ndarray
            An array of indexes for the samples that are in the validation dataset.
        test_idx : ndarray
            An array of indexes for the samples that are in the testing dataset.
        """
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * train_test_split)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]
        train_up_to = int(train_up_to * train_test_split)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        return train_idx, valid_idx, test_idx

    def preprocess_image(self, img_path, gray_scale = False, rgb = False):
        """Preprocesses images before they are used to train the model.

        Parameters
        ----------
        img_path : string
            Path to a specific image.
        gray_scale : bool, optional
            Flag to determine if the image should be converted to greyscale, by default False
        rgb : bool, optional
            Flag to determine if the image should be converted to RGB, by default False

        Returns
        -------
         : PIL.image.image
            Preprocessed image.
        """
        im = Image.open(img_path)
        im = im.resize((self.data_size[0], self.data_size[1]))
        if rgb : im = im.convert('RGB')
        if gray_scale: im = ImageOps.grayscale(im)
        im = np.array(im) / 255.0
        return im

    def generate_images(self, image_idx, batch_size=32, is_training=True):
        """Used to generate batches of images.

        Parameters
        ----------
        image_idx : ndarray
            Array of sample ids that batches will be generated from.
        batch_size : int, optional
            Size of the batches that will be generated, by default 32
        is_training : bool, optional
            Flag that is used to determine if a python generator should be made of if the images should be returned, by default True

        Yields
        -------
        combined : ndarray
            Array of images from the map, roads, and buildings.
          : ndarray
            Array of the x and y coordinates of the solution to the sample.
        """
        # arrays to store our batched data
        combined, images, buildings, roads, x_array, y_array, = [], [], [], [], [], []
        while True:
            for idx in image_idx:
                sample = self.df.iloc[idx]
                
                x_coord = sample['x']
                y_coord = sample['y']
                map = sample['map']
                im_map = self.preprocess_image(map,rgb=True)
                building = sample['building']
                im_building = self.preprocess_image(building,gray_scale=True)
                im_building = np.reshape(im_building,(self.data_size[1],self.data_size[0],1))
                road = sample['road']
                im_roads = self.preprocess_image(road,gray_scale=True)
                im_roads = np.reshape(im_roads,(self.data_size[1],self.data_size[0],1))
                
                x_array.append(x_coord)
                y_array.append(y_coord)
                images.append(im_map)
                buildings.append(im_building)
                roads.append(im_roads)
                
                # yielding condition
                if len(images) >= batch_size:
                    combined = np.concatenate((images,buildings,roads),axis = 3)
                    yield np.array(combined), [np.array(x_array), np.array(y_array)]
                    combined, images, buildings, roads, x_array, y_array, = [], [], [], [], [], []
            if not is_training:
                break

    def make_default_hidden_layers(self, inputs):
        """Creates the hidden layers that are the same between the two outputs.

        Parameters
        ----------
        inputs : ndarray
            The input to the model, which is a array of the input images.

        Returns
        -------
        x : tensorflow.python.keras.engine.functional.Functional
            The model with the hidden layers added.
        """
        x = layers.Conv2D(16, (3, 3), padding="same")(inputs)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Dropout(0.25)(x)

        # x = layers.Conv2D(32, (3, 3), padding="same")(x)
        # x = layers.Activation("relu")(x)
        # x = layers.BatchNormalization(axis=-1)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        # x = layers.Dropout(0.25)(x)

        # x = layers.Conv2D(32, (3, 3), padding="same")(x)
        # x = layers.Activation("relu")(x)
        # x = layers.BatchNormalization(axis=-1)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        # x = layers.Dropout(0.25)(x)
        return x

    def build_x_branch(self, inputs):
        """Creates the model that will process the x coordinate of the solution.

        Parameters
        ----------
        inputs : ndarray
            The input images for the model.

        Returns
        -------
        x : tensorflow.python.keras.engine.functional.Functional
            The model for the x coordinate.
        """
        x = self.make_default_hidden_layers(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(256)(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1)(x)
        x = layers.Activation("tanh", name="x_output")(x)
        return x

    def build_y_branch(self, inputs):
        """Creates the model that will process the y coordinate of the solution.

        Parameters
        ----------
        inputs : ndarray
            The input images for the model.

        Returns
        -------
        x : tensorflow.python.keras.engine.functional.Functional
            The model for the y coordinate.
        """
        x = self.make_default_hidden_layers(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(256)(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1)(x)
        x = layers.Activation("tanh", name="y_output")(x)
        return x

    def assemble_full_model(self, width, height):
        """Creates the CNN model that will be trained.

        Parameters
        ----------
        width : int
            The width of the input image.
        height : int
            The height of the input image.

        Returns
        -------
        model : tensorflow.python.keras.engine.functional.Functional
            The keras CNN model.
        """
        input_shape = (height, width, 5)
        inputs = layers.Input(shape=input_shape)
        x_branch = self.build_x_branch(inputs)
        y_branch = self.build_y_branch(inputs)
        model = keras.models.Model(inputs=inputs,
                     outputs = [x_branch, y_branch],
                     name="rsu_placement_net")
        return model

init_lr = 1e-4

database_name = "Austin_downtown"
cnn = RSU_Placement_CNN(database_name)
cnn(200,init_lr = init_lr,batch_size=64)