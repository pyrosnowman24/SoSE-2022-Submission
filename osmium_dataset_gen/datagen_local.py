import numpy as np
import matplotlib.pyplot as plt
import pyrosm
import os, sys, io
import pathlib
from datetime import datetime
import geotiler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_coordinate_transform import Dataset_Transformation
from PIL.Image import FLIP_TOP_BOTTOM
from PIL import Image
import pandas as pd
from shapely.geometry import Polygon
import geopandas

class Dataset_Generator():
    def __init__(self,global_bounding_box,sample_bounding_box,zoom=17):
        self.global_osm = pyrosm.OSM("/home/ace/Desktop/NNWork/Map_Dataset_Generator/osmium_dataset_gen/austin_downtown.pbf")
        # self.global_osm = pyrosm.OSM("/home/ace/Desktop/NNWork/Map_Dataset_Generator/osmium_dataset_gen/sa_downtown.pbf")
        # print(self.global_osm.conf.tags.highway)
        self.global_bounding_box = global_bounding_box
        self.sample_bounding_box = sample_bounding_box
        self.global_map  = geotiler.Map(extent=self.global_bounding_box, zoom = zoom)
        self.global_image = geotiler.render_map(self.global_map).transpose(FLIP_TOP_BOTTOM)
        self.transforms = Dataset_Transformation(global_bounding_box,self.global_image.size,sample_bounding_box)
        self.df_data = pd.DataFrame(data=None,columns=["cord1","cord2","cord3","cord4","cord5","cord6","cord7","cord8","center1","center2","angle","solution1","solution2"])


    def __call__(self,number_of_samples,plot=False,save_data = True,file_name = None):
        self.plot = plot
        if file_name is None: self.new_file = True
        else: self.new_file = False
        if save_data: folder,data_file,image_folder, road_folder, building_folder = self.create_files(file_name)
        if not self.new_file:
            i = self.get_index(data_file)
        else: i = 0
        self.i_init = i
        if save_data: self.save_image(folder,self.global_image,0,assigned_name = "Map")

        while i < number_of_samples:
            coordinates,center,angle,diagonal = self.generate_area()
            intersections, road_ways, buildings = self.find_constrained_intersections(coordinates)
            if intersections is None or intersections.shape[0] == 0 or buildings is None or road_ways is None:
                print("Zero detected")
                continue
            else:
                solution = self.choose_intersection(intersections,center)
                cropped_img,transformed_solution = self.crop_image(coordinates,angle,solution)
                building_image = self.crop_OSM_image(coordinates,angle,buildings)
                road_image = self.crop_OSM_image(coordinates,angle,road_ways)
                # if plot:
                #     self.plot_map(cropped_img,transformed_solution)
                #     self.plot_data(data,coordinates,solution = solution)
                #     plt.show()
                self.add_data(coordinates,center,angle,solution)
                if save_data: self.df_to_csv(i,data_file)
                if save_data: self.save_image(image_folder,cropped_img,i)
                if save_data: self.save_image(building_folder,building_image,i)
                if save_data: self.save_image(road_folder,road_image,i)
                print("Completed ",i)
                i = i+1
                plt.show()

    def create_files(self,named_file = None):
        if self.new_file:
            now = datetime.now()
            current_time = now.strftime("%c")
            folder_name = str(current_time)
        else:
            folder_name = named_file
        current_path = pathlib.Path().resolve()
        folder_path = 'Datasets'
        path = os.path.join(current_path,folder_path)
        folder = os.path.join(path, folder_name)
        image_folder = os.path.join(folder, "Images/map_images")
        road_folder = os.path.join(folder, "Images/road_images")
        building_folder = os.path.join(folder, "Images/building_images")
        data_file = os.path.join(folder, "data.csv")
        if self.new_file:
            os.makedirs(folder)
            os.makedirs(image_folder)
            os.makedirs(building_folder)
            os.makedirs(road_folder)
            datas = open(data_file,"w+")
            datas.close()
        return folder,data_file,image_folder, road_folder, building_folder

    def get_index(self,data_file):
            file = open(data_file, "r")
            line_count = 0
            for line in file:
                if line != "\n":
                    line_count += 1
            file.close()
            return line_count-1

    def generate_area(self):
        rectangle_coordinates = np.zeros((4,2))
        diagonal = np.sqrt(np.sum(np.power(self.sample_bounding_box,2)))*.5 # calculates half the diagonal of the rectangle
        angle = np.random.uniform(low = 0, high = 2*np.pi) # Generates random angle for the area the sample will be pulled from. 
        start_point = np.array((np.random.uniform(low = diagonal,high = self.global_image.size[0]-diagonal),np.random.uniform(low = diagonal, high = self.global_image.size[1]-diagonal)))
        rectangle_coordinates = self.get_rectangle_corners(start_point,angle)
        rectangle_coordinates = self.transforms.map_to_coordinate(rectangle_coordinates)
        start_point = self.transforms.map_to_coordinate(start_point)
        return rectangle_coordinates,start_point,angle,diagonal # lon,lat

    def get_rectangle_corners(self,center,angle):
        v1 = np.array((np.cos(angle),np.sin(angle)))
        v2 = np.array((-v1[1],v1[0]))
        v1*=self.sample_bounding_box[0]/2
        v2*=self.sample_bounding_box[1]/2
        return np.array(((center + v1 + v2),(center - v1 + v2),(center - v1 - v2),(center + v1 - v2))) # map coordinates

    def plot_data(self,coordinates,map,roads,buildings):
        cords_map = self.transforms.coordinate_to_map(coordinates)
        poly = Polygon(coordinates)
        poly2 = Polygon(cords_map)
        poly_gdf = geopandas.GeoDataFrame([1], geometry=[poly])

        fig,(ax0,ax1) = plt.subplots(2)
        ax0.imshow(self.global_image, zorder=1)
        # poly_gdf.boundary.plot(ax=ax0, color="red", zorder=2)
        ax0.scatter(cords_map[:,0],cords_map[:,1],zorder = 2)
        ax1.imshow(map)
        plt.draw()

        fig,(ax2,ax3) = plt.subplots(2)
        poly_gdf.boundary.plot(ax=ax2, color="red")
        buildings.plot(column="building", cmap="RdBu", legend=False, ax = ax2, aspect = None)
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        poly_gdf.boundary.plot(ax=ax3, color="red")
        roads.plot(column="highway", legend=False, ax = ax3, aspect = None)

        plt.show()

    def find_constrained_intersections(self,coordinates):
        poly = Polygon(coordinates)
        # This needs to be changed in the future, it would only work on this computer and it cant automatically change between cities
        osm = pyrosm.OSM("/home/ace/Desktop/NNWork/Map_Dataset_Generator/osmium_dataset_gen/austin_downtown.pbf",bounding_box=poly)
        # osm = pyrosm.OSM("/home/ace/Desktop/NNWork/Map_Dataset_Generator/osmium_dataset_gen/sa_downtown.pbf",bounding_box=poly)
        buildings = osm.get_buildings()
        try: roads = osm.get_network("driving+service",nodes=True)
        except ValueError: return None, None, None
        road_nodes = roads[0]
        road_ways = roads[1]
        intersections = None
        if road_nodes is not None:
            lon = np.array(road_nodes["lon"])
            lon = np.reshape(lon,(len(lon),1))
            lat = np.array(road_nodes["lat"])
            lat = np.reshape(lat,(len(lat),1))
            intersections = np.hstack((lon,lat))
        else:
            coordinates = None

        # if self.plot:
        #     map_intersections = self.transforms.coordinate_to_map(intersections)
        #     map_coordinates = self.transforms.coordinate_to_map(coordinates)
        #     fig,ax = plt.subplots(1)
        #     ax.imshow(self.global_image,origin = "lower")
        #     ax.scatter(map_intersections[:,0],map_intersections[:,1],marker = 'x')
        #     ax.scatter(map_coordinates[:,0],map_coordinates[:,1])
        #     plt.draw()


        return intersections, road_ways, buildings

    def save_image(self,folder,image,index,assigned_name=None):
        name = "image_"+str(index)+".png"
        if assigned_name is not None:
            name = assigned_name
        image_name = os.path.join(folder, name)
        image.save(image_name,"PNG")

    def df_to_csv(self,index,data_file):
        print(self.df_data.shape," ", index+1 - self.i_init)
        self.df_data.index = np.arange(self.i_init,index+1)
        if index == 0 and self.new_file:
            self.df_data[index:index+1].to_csv(data_file, mode='a')
        else:
            self.df_data[-1:].to_csv(data_file, mode='a', header=False)

    def choose_intersection(self,intersections,center,number_solution=1): # chooses an intersection to be the solution to the problem
        nearest = np.zeros((intersections.shape[0],3))
        nearest[:,:2] = np.copy(intersections)
        nearest[:,2] = np.linalg.norm(np.subtract(nearest[:,:2],center),axis = 1)
        return np.reshape(nearest[nearest[:,2] == nearest[:,2].min(),:2][0,:],(number_solution,2))

    def add_data(self,coordinates,center,angle,solution):
        data_array = np.array((*coordinates.flatten(),*center.flatten(),angle,*solution.flatten()))
        exit_var = True
        while exit_var:
            self.df_data.loc[len(self.df_data)] = data_array.tolist()
            if self.df_data.shape[0] != 0 and self.df_data.iloc[len(self.df_data)-1]['solution2'] != data_array[-1]:
                print("Error with adding value, retrying")
            else:
                exit_var = False

    def OSM_fig_to_img(self,OSM_data,coordinates):
        poly = Polygon(coordinates)
        poly_gdf = geopandas.GeoDataFrame([1], geometry=[poly])
        # convert matplotlib figure into PIL image without borders
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(1)
        poly_gdf.boundary.plot(ax=ax, color="red")
        OSM_data.plot( color='k', legend=False, ax = ax, aspect = None)
        ax.invert_yaxis()
        plt.axis('off')
        plt.margins(x=0)
        plt.margins(y=0)
        img_buf = io.BytesIO()
        plt.savefig(img_buf,bbox_inches='tight',transparent=True, pad_inches=0)
        plt.close()
        img = Image.open(img_buf)
        img = img.resize([int(self.osm_image_size[2]),int(self.osm_image_size[3])])
        pixels = img.load()
        for y in range(img.size[1]): 
            for x in range(img.size[0]): 
                if pixels[x,y][3] < 255:    # check alpha
                    pixels[x,y] = (255, 255, 255, 255)
        return img

    def crop_image(self,coordinates,angle,solution):
        coordinates = np.vstack((coordinates,solution))
        coordinates = self.transforms.coordinate_to_map(coordinates)
        bounds = self.getBounds(coordinates)
        img2 = self.global_image.crop(bounds.astype(int))
        bound_center = self.getBoundsCenter(bounds)
        crop_center = self.getCenter(img2)
        crop_points = np.apply_along_axis(self.recenter,1,coordinates,bound_center,crop_center)
        # In order for the osm data to be scalled correctly, it must be fit to this size. By pulling the information from here,
        # it makes it so this information dosnt need to be calculated unnecessarily again
        self.osm_image_size = self.getBounds(crop_points) 
        rotated_points = np.apply_along_axis(self.rotate,1,crop_points,crop_center,-angle)
        img3 = img2.rotate(-angle * 180 / np.pi, expand=True)
        im3_center = self.getCenter(img3)
        rotated_points = self.recenter(rotated_points,0,im3_center)
        img4 = img3.crop(self.getBounds(rotated_points).astype(int))
        im4_center = self.getCenter(img4)
        final_coords = self.recenter(rotated_points,im3_center,im4_center)

        if self.plot:
            fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
            ax0.imshow(self.global_image,origin = 'lower')
            ax0.scatter(coordinates[-1,0],coordinates[-1,1],color = 'r')
            ax0.scatter(coordinates[:-1,0],coordinates[:-1,1],marker='x',color = 'k')
            ax1.imshow(img2,origin = 'lower')
            ax1.scatter(crop_points[-1,0],crop_points[-1,1],color = 'r')
            ax1.scatter(crop_points[:-1,0],crop_points[:-1,1],marker='x',color = 'k')
            ax1.scatter(crop_center[0],crop_center[1],marker='x',color='g')
            ax2.imshow(img3,origin = 'lower')
            ax2.scatter(rotated_points[-1,0],rotated_points[-1,1],color = 'r')
            ax2.scatter(rotated_points[:-1,0],rotated_points[:-1,1],marker='x',color = 'k')
            ax2.scatter(im3_center[0],im3_center[1],marker='x',color='g')
            ax3.imshow(img4,origin = 'lower')
            ax3.scatter(final_coords[-1,0],final_coords[-1,1],color = 'r')
            ax3.scatter(final_coords[:-1,0],final_coords[:-1,1],marker='x',color = 'k')
            ax2.set_title(angle)
            ax3.set_title(angle * 180 / np.pi)
            ax3.set_xlim(0,self.sample_bounding_box[0])
            ax3.set_ylim(0,self.sample_bounding_box[1])
            plt.draw()

        return img4, final_coords[-1,:].astype(int)

    def crop_OSM_image(self,coordinates,angle,OSM_data):
        bounds = self.getBounds(coordinates)
        img2 = self.OSM_fig_to_img(OSM_data,coordinates)
        crop_points = self.coordinate_transform(coordinates,bounds,img2.size)
        bounds = self.getBounds(crop_points)
        crop_center = self.getCenter(img2)
        rotated_points = np.apply_along_axis(self.rotate,1,crop_points,crop_center,-angle)
        img3 = img2.rotate(-angle * 180 / np.pi, expand=True)
        im3_center = self.getCenter(img3)
        rotated_points = self.recenter(rotated_points,0,im3_center)
        img4 = img3.crop(self.getBounds(rotated_points).astype(int))
        im4_center = self.getCenter(img4)
        final_coords = self.recenter(rotated_points,im3_center,im4_center)

        if self.plot:
            fig,(ax1,ax2,ax3) = plt.subplots(3,1)
            ax1.imshow(img2,origin = 'lower')
            ax1.scatter(crop_points[:,0],crop_points[:,1],marker='x',color = 'k')
            ax1.scatter(crop_center[0],crop_center[1],marker='x',color='g')
            ax2.imshow(img3,origin = 'lower')
            ax2.scatter(rotated_points[:,0],rotated_points[:,1],marker='x',color = 'k')
            ax2.scatter(im3_center[0],im3_center[1],marker='x',color='g')
            ax3.imshow(img4,origin = 'lower')
            ax3.scatter(final_coords[:,0],final_coords[:,1],marker='x',color = 'k')
            ax2.set_title(angle)
            ax3.set_title(angle * 180 / np.pi)
            ax3.set_xlim(0,self.sample_bounding_box[0])
            ax3.set_ylim(0,self.sample_bounding_box[1])
            plt.draw()

        return img4

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

    def coordinate_transform(self,coordinates,old_range,new_range): # This will ONLY work for self.getBounds --> Image size
        new_coords = np.zeros(coordinates.shape)
        new_coords[:,0] = coordinates[:,0]-np.min((old_range[0],old_range[2]))
        new_coords[:,1] = coordinates[:,1]-np.min((old_range[1],old_range[3]))

        new_coords[:,0] = new_coords[:,0]/(np.max((old_range[0],old_range[2]))-np.min((old_range[0],old_range[2])))
        new_coords[:,1] = new_coords[:,1]/(np.max((old_range[1],old_range[3]))-np.min((old_range[1],old_range[3])))

        new_coords[:,0] = new_coords[:,0]*new_range[0]
        new_coords[:,1] = new_coords[:,1]*new_range[1]

        return new_coords

# bbox =  -98.5149, 29.4441, -98.4734, 29.3876 # San Antonio Downtown
bbox = -97.7907, 30.2330, -97.6664, 30.3338 # Austin Downtown
data_size = [250,500]
gen = Dataset_Generator(bbox,data_size)
# file_name = 'Tue 26 Oct 2021 03:35:35 PM '
file_name = None
gen(500,save_data = True,plot=False,file_name=file_name)