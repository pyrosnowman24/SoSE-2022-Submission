from PIL.Image import FLIP_TOP_BOTTOM
from numpy.core.fromnumeric import diagonal
from numpy.core.numeric import NaN
import requests
import numpy as np
import matplotlib.pyplot as plt
import geotiler
import os
from datetime import datetime
import pandas as pd
import pathlib
import time
from dataset_coordinate_transform import Dataset_Transformation

class Data_Generator():
    """
    Generates a dataset for training a CNN how to place RSUs in a city. 
    The user specifies an area to gather samples from, the number of samples to gather, and the size each sample should be.
    The generator will create a folder named after the time that the script was run, which contains the following files:
        A folder called Images containing the .png images of each sampled area. These size of the images may differ slightly, so trimming them back to the desired size is necessary.
        A csv file called data.csv which contains 13 attributes: cord1, cord2, cord3, cord4, cord5, cord6, cord7, cord8, center1, center2, angle, solution1, solution2
        Cord1-8 are the four corner coordinates for the sample in lon,lat. center1-2 is the center of the sample region in lon,lat. The angle is the amount in radians that the sample is rotated counter clock wise. solution1-2 are the coordinates of the intersection that the RSU should be placed on.
        An image called Map.png that contains the image of the global region specified by the user.
    """
    def __init__(self,bbox,zoom=17,data_size = [250,500]):
        self.global_bbox = bbox
        self.data_size = data_size
        self.map  = geotiler.Map(extent=self.global_bbox, zoom = zoom)
        self.image = geotiler.render_map(self.map).transpose(FLIP_TOP_BOTTOM)
        self.transforms = Dataset_Transformation(bbox,self.image.size,data_size)
        self.overpass_url = "https://overpass.kumi.systems/api/interpreter"
        self.df_data = pd.DataFrame(data=None,columns=["cord1","cord2","cord3","cord4","cord5","cord6","cord7","cord8","center1","center2","angle","solution1","solution2"])


    def __call__(self,data_set_size,plot=False,save_data = True,named_file = None):
        self.plot = plot
        if named_file is None: self.new_file = True
        else: self.new_file = False
        if save_data: folder,data_file,image_folder = self.create_files(named_file)
        if not self.new_file:
            i = self.get_index(data_file)
        else: i = 0
        self.i_init = i
        if save_data: self.save_image(folder,self.image,0,assigned_name = "Map")

        while i < data_set_size:
            coordinates,center,angle,diagonal = self.generate_area()
            data = self.find_constrained_intersections(coordinates)
            intersections = self.transforms.data_to_coordinate(data)
            if intersections.shape[0] == 0:
                print("Zero detected")
                continue
            else:
                solution = self.choose_intersection(intersections,center)
                cropped_img,transformed_solution = self.crop_image(coordinates,angle,solution)
                if plot:
                    self.plot_map(cropped_img,transformed_solution)
                    self.plot_data(data,coordinates,solution = solution)
                    plt.show()
                self.add_data(coordinates,center,angle,solution)
                if save_data: self.save_image(image_folder,cropped_img,i)
                if save_data: self.df_to_csv(i,data_file)
                print("Completed ",i)
                i = i+1
                time.sleep(1)

    def get_index(self,data_file):
        file = open(data_file, "r")
        line_count = 0
        for line in file:
            if line != "\n":
                line_count += 1
        file.close()
        return line_count-1

    def create_files(self,named_file):
        if self.new_file:
            now = datetime.now()
            current_time = now.strftime("%c")
            folder_name = str(current_time)
        else:
            folder_name = named_file
        current_path = pathlib.Path().resolve()
        folder_path = 'Map_Dataset_Generator/Datasets'
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
 
    def add_data(self,coordinates,center,angle,solution):
        data_array = np.array((*coordinates.flatten(),*center.flatten(),angle,*solution.flatten()))
        self.df_data.loc[len(self.df_data)] = data_array.tolist()

    def save_image(self,folder,image,index,assigned_name=None):
        name = "image_"+str(index)+".png"
        if assigned_name is not None:
            name = assigned_name
        image_name = os.path.join(folder, name)
        image.save(image_name,"PNG")

    def df_to_csv(self,index,data_file):
        self.df_data.index = np.arange(self.i_init,index+1)
        if index == 0 and self.new_file:
            self.df_data[index:index+1].to_csv(data_file, mode='a')
        else:
            self.df_data[-1:].to_csv(data_file, mode='a', header=False)
        

    def generate_area(self): # Creates a tilted area that will be used for the sample. 
        rectangle_coordinates = np.zeros((4,2))
        diagonal = np.sqrt(np.sum(np.power(self.data_size,2)))*.5 # calculates half the diagonal of the rectangle
        angle = np.random.uniform(low = 0, high = 2*np.pi) # Generates random angle for the area the sample will be pulled from. 
        start_point = np.array((np.random.uniform(low = diagonal,high = self.image.size[0]-diagonal),np.random.uniform(low = diagonal, high = self.image.size[1]-diagonal)))
        rectangle_coordinates = self.get_rectangle_corners(start_point,angle)
        rectangle_coordinates = self.transforms.map_to_coordinate(rectangle_coordinates)
        start_point = self.transforms.map_to_coordinate(start_point)
        return rectangle_coordinates,start_point,angle,diagonal # lon,lat

    def get_rectangle_corners(self,center,angle): # finds the corners of a tilted rectangle
        v1 = np.array((np.cos(angle),np.sin(angle)))
        v2 = np.array((-v1[1],v1[0]))
        v1*=self.data_size[0]/2
        v2*=self.data_size[1]/2
        return np.array(((center + v1 + v2),(center - v1 + v2),(center - v1 - v2),(center + v1 - v2))) # map coordinates

# Crop image, getBounds,getCenter,getBoundsCenter,recenter, and rotate are all based on the diagonal-crop python library https://github.com/jobevers/diagonal-crop

    def crop_image(self,coordinates,angle,solution):
        coordinates = np.vstack((coordinates,solution))
        coordinates = self.transforms.coordinate_to_map(coordinates)
        bounds = self.getBounds(coordinates)
        img2 = self.image.crop(bounds.astype(int))
        bound_center = self.getBoundsCenter(bounds)
        crop_center = self.getCenter(img2)
        crop_points = np.apply_along_axis(self.recenter,1,coordinates,bound_center,crop_center)
        rotated_points = np.apply_along_axis(self.rotate,1,crop_points,crop_center,angle)
        img3 = img2.rotate(angle * 180 / np.pi, expand=True)
        im3_center = self.getCenter(img3)
        rotated_points = self.recenter(rotated_points,0,im3_center)
        img4 = img3.crop(self.getBounds(rotated_points).astype(int))
        im4_center = self.getCenter(img4)
        final_coords = self.recenter(rotated_points,im3_center,im4_center)

        if self.plot:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3)
            ax1.imshow(img2,origin = 'lower')
            ax1.scatter(crop_points[-1,0],crop_points[-1,1],color = 'k')
            ax1.scatter(crop_points[:-1,0],crop_points[:-1,1],marker='x',color = 'k')
            ax1.scatter(crop_center[0],crop_center[1],marker='x',color='r')
            ax2.imshow(img3,origin = 'lower')
            ax2.scatter(rotated_points[-1,0],rotated_points[-1,1],color = 'k')
            ax2.scatter(rotated_points[:-1,0],rotated_points[:-1,1],marker='x',color = 'k')
            ax2.scatter(im3_center[0],im3_center[1],marker='x',color='r')
            ax3.imshow(img4,origin = 'lower')
            ax3.scatter(final_coords[-1,0],final_coords[-1,1],color = 'k')
            ax3.scatter(final_coords[:-1,0],final_coords[:-1,1],marker='x',color = 'k')
            ax2.set_title(angle)
            ax3.set_title(angle * 180 / np.pi)
            plt.draw()

        return img4, final_coords[-1,:].astype(int)

    def getBounds(self,points):
        xs, ys = zip(*points)
        # left, upper, right, lower using the usual image coordinate system
        # where top-left of the image is 0, 0
        return np.array((min(xs), min(ys), max(xs), max(ys)))

    def getCenter(self,im):
        return np.divide(im.size,2)

    def getBoundsCenter(self,bounds):
        return np.array(((bounds[2] - bounds[0]) / 2 + bounds[0],(bounds[3] - bounds[1]) / 2 + bounds[1]))
        
    def recenter(self, coord, old_center, new_center):
        return np.add(coord,np.subtract(new_center,old_center))

    def rotate(self, coord, center, angle):
        # angle should be in radians
        rotation_matrix = np.array(((np.cos(angle),-np.sin(angle)),(np.sin(angle),np.cos(angle))))
        coord = self.recenter(coord,center,0)
        rotated_coords = np.matmul(coord,rotation_matrix)
        return rotated_coords

    def find_constrained_intersections(self,coordinates): # Finds the intersections in a rectangular area.
        cord1 = coordinates[0,:]
        cord2 = coordinates[1,:]
        cord3 = coordinates[2,:]
        cord4 = coordinates[3,:]
        # Code for query can be developed in Overpass Turbo
        overpass_query = f"""
        [out:json];
        way(poly:"{cord1[1]} {cord1[0]} {cord2[1]} {cord2[0]} {cord3[1]} {cord3[0]} {cord4[1]} {cord4[0]}");
        //way[highway~"^(unclassified|residential|living_street|service|motorway|trunk|primary|secondary|tertiary|(motorway|trunk|primary|secondary)_link)$"]
        way._[highway~"^(motorway|trunk|primary|secondary|tertiary|residential)$"]
        [~"^(name|ref)$"~"."] -> .allways;
        foreach.allways -> .currentway(
            (.allways; - .currentway;)->.otherways_unfiltered; // Calculates the difference between all roads in the area and the current one being looked at
            way.otherways_unfiltered(if:t["name"] != currentway.u(t["name"])) -> .otherways; // Removes any ways that were selected that have the same name as the current road
            node(w.currentway)->.currentwaynodes;
            node(w.otherways)->.otherwaynodes;
            node.currentwaynodes.otherwaynodes;   // intersection of the current roadway and other roadways
            node._(poly:"{cord1[1]} {cord1[0]} {cord2[1]} {cord2[0]} {cord3[1]} {cord3[0]} {cord4[1]} {cord4[0]}"); // The intersections returned before may still contain nodes outside the area, this removes them
            out;
        );
        """
        while True: # Handles exceptions from the query to OpenStreetMaps, they can time out if too many are made.
            try:
                response = requests.get(self.overpass_url, 
                                        params={'data': overpass_query})
                data = response.json()
                break
            except:
                time.sleep(1)
                continue
        return data

    def plot_map(self,map,solution = NaN): # plots a map and the coordinates of the sample area in it.
        fig,ax = plt.subplots(1)
        ax.imshow(map, origin = 'lower')
        if not np.isnan(solution).all(): ax.scatter(solution[0],solution[1],marker='o',color = 'g',zorder = 1)
        ax.set_title('Intersections in San Antonio')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.axis('equal')
        plt.draw()

    def plot_data(self,data,coordinates,solution = NaN): # plots the map, the sample area, the intersections, and the solution
        coords = self.transforms.data_to_coordinate(data)
        X = self.transforms.coordinate_to_map(coords)
        if not np.isnan(solution).all(): solution = self.transforms.coordinate_to_map(solution)
        coordinates = self.transforms.coordinate_to_map(coordinates)
        fig,ax = plt.subplots(1)
        ax.scatter(coordinates[:,0],coordinates[:,1],marker='o',color='k')
        ax.scatter(X[:, 0], X[:, 1], marker='x',c = 'r')
        if not np.isnan(solution).all(): ax.scatter(solution[:,0],solution[:,1],marker='o',color = 'g',zorder = 1)
        ax.imshow(self.image,origin = 'lower')
        ax.set_title('Intersections in San Antonio')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.axis('equal')
        plt.draw()

    def choose_intersection(self,intersections,center,number_solution=1): # chooses an intersection to be the solution to the problem
        # intersections = self.transforms.coordinate_to_map(intersections)
        nearest = np.zeros((intersections.shape[0],3))
        nearest[:,:2] = intersections
        nearest[:,2] = np.linalg.norm(np.subtract(nearest[:,:2],center),axis = 1)
        return np.reshape(nearest[nearest[:,2] == nearest[:,2].min(),:2][0,:],(number_solution,2))


bbox =  -98.5149, 29.4441, -98.4734, 29.3876 # San Antonio Downtown
data_size = [250,500]
data_gen = Data_Generator(bbox,data_size = data_size)
data_gen(10000,plot=False,save_data=True,named_file= "Thu 26 Aug 2021 03:29:43 PM ")
