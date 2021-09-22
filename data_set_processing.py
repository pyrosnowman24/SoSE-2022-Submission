from re import X
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
import glob
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint


class RSU_Placement_CNN():
    def __init__(self,train_test_split = .7):
        self.train_test_split = train_test_split
        self.import_files()

    def __call__(self, epochs, init_lr = 1e-4):
        model = self.assemble_full_model(*self.data_size,*self.data_size)

        train_idx, valid_idx, test_idx = self.generate_split_indexes() 
        batch_size = 150
        valid_batch_size = 150
        train_gen = self.generate_images(train_idx, batch_size=batch_size)
        valid_gen = self.generate_images(valid_idx, batch_size=valid_batch_size)
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
                            validation_steps=len(valid_idx)//valid_batch_size)
        self.plot_history(history,fig_path = model_path)
        var_path = self.create_var_file(model_path)
        self.save_variable_file(var_path,epochs,init_lr,batch_size,valid_batch_size)
        self.evaluate_model(model,test_idx)
        plt.show()

    def import_files(self):
        # Set all file paths
        current_path = pathlib.Path().resolve()
        folder_path = 'Datasets'
        dataset_name = 'Thu 26 Aug 2021 03:29:43 PM '
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
        self.bbox =  -98.5149, 29.4441, -98.4734, 29.3876 # San Antonio Downtown
        self.data_size = [250,500]
        self.transforms = Dataset_Transformation(self.bbox,self.world_img.size,self.data_size)

        # Import data.csv file
        self.data = pd.read_csv(data_file_path)
        self.data_array = self.data.to_numpy()
        self.data_array = self.transforms.prepare_dataset(self.data_array)
        self.df = pd.DataFrame(self.data_array[:,-2:],columns = ['x','y'])

        # Import images
        
        files = glob.glob(os.path.join(images_path, "*.%s" % 'png'))
        self.df['file'] = files

    def create_var_file(self,folder_path):
        var_file = os.path.join(folder_path, "variables.csv")
        datas = open(var_file,"w+")
        datas.close()
        return var_file

    def save_variable_file(self,var_file,num_epochs,lr,batch,val_batch):
        vars = pd.DataFrame(None,columns=["num_epochs","lr","batch","val_batch"])
        vars.loc[0] = np.array((num_epochs,lr,batch,val_batch)).tolist()
        vars.to_csv(var_file)

    def evaluate_model(self,model,test_idx):
        test_batch_size = len(test_idx)/10
        test_generator = self.generate_images(test_idx, batch_size=test_batch_size, is_training=True)
        x_pred, y_pred = model.predict(test_generator, steps=len(test_idx)//test_batch_size)
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

    def plot_history(self,history,fig_path = None):
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

    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * self.train_test_split)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]
        train_up_to = int(train_up_to * self.train_test_split)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        return train_idx, valid_idx, test_idx

    def preprocess_image(self, img_path):
        im = Image.open(img_path)
        im = im.resize((self.data_size[0], self.data_size[1]))
        im = np.array(im) / 255.0
        return im

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

    def make_default_hidden_layers(self, inputs):
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

        x = layers.Conv2D(32, (3, 3), padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)
        return x

    def build_x_branch(self, inputs, num_x):
        x = self.make_default_hidden_layers(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1)(x)
        x = layers.Activation("tanh", name="x_output")(x)
        return x

    def build_y_branch(self, inputs, num_y):
        x = self.make_default_hidden_layers(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1)(x)
        x = layers.Activation("tanh", name="y_output")(x)
        return x

    def assemble_full_model(self, width, height, num_x,num_y):
        input_shape = (height, width, 4)
        inputs = layers.Input(shape=input_shape)
        x_branch = self.build_x_branch(inputs, num_x)
        y_branch = self.build_y_branch(inputs, num_y)
        model = keras.models.Model(inputs=inputs,
                     outputs = [x_branch, y_branch],
                     name="rsu_placement_net")
        return model

init_lr = 1e-4

cnn = RSU_Placement_CNN()
cnn(200,init_lr = init_lr)