import numpy as np
import matplotlib.pyplot as plt
import pyrosm
import os
import pathlib
from datetime import datetime

class Dataset_Generator():
    def __init__(self):
        self.osm = pyrosm.OSM("/home/ace/Desktop/NNWork/Map_Dataset_Generator/osmium_dataset_gen/sa_downtown.pbf")

    def __call__(self,plot=False,save_data = True,file_name = None):
        if save_data: folder,data_file,image_folder = self.create_files(file_name)
        my_filter = {"building": ["residential", "retail"]}
        self.buildings = self.osm.get_buildings()
        ax = self.buildings.plot(column="building", cmap="RdBu", legend=False)
        plt.draw()
        self.roads = self.osm.get_network("driving+service")
        ax = self.roads.plot(column="highway", legend=False)
        plt.show()



    def create_files(self,named_file):
        if self.new_file:
            now = datetime.now()
            current_time = now.strftime("%c")
            folder_name = str(current_time)
        else:
            folder_name = named_file
        current_path = pathlib.Path().resolve()
        folder_path = '../Datasets'
        path = os.path.join(current_path,folder_path)
        folder = os.path.join(path, folder_name)
        image_folder = os.path.join(folder, "Images/samples")
        data_file = os.path.join(folder, "data.csv")
        if self.new_file:
            os.makedirs(folder)
            os.makedirs(image_folder)
            datas = open(data_file,"w+")
            datas.close()
        return folder,data_file,image_folder

gen = Dataset_Generator()
gen(save_data = False)