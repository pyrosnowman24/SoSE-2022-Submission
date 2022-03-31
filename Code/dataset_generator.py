import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
import pyrosm
import os, sys, io
import pathlib
import geotiler
from dataset_coordinate_transform import Dataset_Transformation
from PIL.Image import FLIP_TOP_BOTTOM
from PIL import Image
import pandas as pd
from shapely.geometry import Polygon
import geopandas

class Dataset_Generator():
    """Class used to generate image data to train a CNN for the RSU placement problem. The class can create a new dataset or add additional samples to an existing dataset.
    """
    def __init__(self,global_bounding_box,sample_bounding_box,folder_name,OSM_file,zoom=17,map_image=None):
        """Creates a Dataset_Generator class.

        Parameters
        ----------
        global_bounding_box : array
            The four (lat,lon) coordinates used to form the perimiter of the map that the sample data will be generated from.
        sample_bounding_box : array
            The width and height of a sample in pixels. 
        folder_name : string
            The name of the folder that should be created or expanded.
        OSM_file : string
            The path to the OSM file that the sample data will be generated from.
        zoom : int, optional
            A parameter used by geotiller when creating an image of the map, by default 17
        map_image : string
            Provides a path to a premade .png for the map.
            
        """
        self.OSM_file = OSM_file
        self.global_osm = pyrosm.OSM(OSM_file)
        self.global_bounding_box = global_bounding_box
        self.sample_bounding_box = sample_bounding_box
        self.folder_name = folder_name
        if map_image is None:
            self.global_map  = geotiler.Map(extent=self.global_bounding_box, zoom = zoom)
            self.global_image = geotiler.render_map(self.global_map).transpose(FLIP_TOP_BOTTOM)
        else:
            self.global_image = Image.open(map_image)
        self.transforms = Dataset_Transformation(global_bounding_box,self.global_image.size,sample_bounding_box)
        self.df_data = pd.DataFrame(data=None,columns=["cord1","cord2","cord3","cord4","cord5","cord6","cord7","cord8","center1","center2","angle","solution1","solution2"])

    def __call__(self,number_of_samples,plot=False,save_data = True):
        """Creates the requested number of samples and adds them to the database.

        Parameters
        ----------
        number_of_samples : int
            Specifies the number of samples that the dataset should contain. If the dataset contains less then this value then more will be added until it reaches this value.
        plot : bool, optional
            Boolean determining if plots should be shown, by default False
        save_data : bool, optional
            Boolean determining if data should be saved, by default True
        """

        self.plot = plot
        if save_data: folder, data_file, variable_file, image_folder, road_folder, building_folder, i = self.create_files()
        if not save_data: i = 0 
        self.i_init = i
        if save_data: self.save_image(folder,self.global_image,0,assigned_name = "Map")
        if save_data: self.save_variables(variable_file)

        while i < number_of_samples:
            coordinates,center,angle = self.generate_area()
            intersections, road_ways, buildings = self.find_constrained_intersections(coordinates)
            if intersections is None or intersections.shape[0] == 0 or buildings is None or road_ways is None:
                print("Zero detected, generating new sample.")
                continue
            else:
                solution = self.choose_intersection(intersections,center)
                cropped_img,transformed_solution = self.crop_image(coordinates,angle,intersections,solution)
                building_image = self.crop_OSM_image(coordinates,angle,buildings)
                road_image = self.crop_OSM_image(coordinates,angle,road_ways)
                df_add_fail_check = self.add_data(coordinates,center,angle,solution)

                if df_add_fail_check:
                    print("Failed to add sample to df, generating new sample.")
                    continue
                else:
                    if save_data: self.df_to_csv(i,data_file)
                    if save_data: self.save_image(image_folder,cropped_img,index = i)
                    if save_data: self.save_image(building_folder,building_image,index = i)
                    if save_data: self.save_image(road_folder,road_image,index = i)
                    print("Completed ",i)
                    i = i+1
                    plt.show()

    def create_files(self):
        """Determines if the provided database already exists, and if not creates a new one using the provided name. Returns the paths for the database directory, the data.csv file, and each image folder for the samples.

        Returns
        -------
        folder : string
            Path of the database directory.
        data_file : string
            Path of the data.csv for the database.
        image_folder : string
            Path of the directory containing the map images for each sample.
        road_folder : string
            Path of the directory containing the road images for each sample.
        building_folder : string
            Path of the directory containing the building trace images for each sample.
        index : int
            Index that the generated data should begin at in the .csv file.
        """
        current_path = pathlib.Path.cwd()
        folder_path = 'Datasets'
        path = os.path.join(current_path,folder_path)
        folder = os.path.join(path, self.folder_name)
        image_folder = os.path.join(folder, "Images/map_images")
        road_folder = os.path.join(folder, "Images/road_images")
        building_folder = os.path.join(folder, "Images/building_images")
        data_file = os.path.join(folder, "data.csv")
        variable_file = os.path.join(folder,"variables.csv")

        if os.path.isdir(folder):
            print("Directory already exists.")
            index = self.get_index(data_file)
        else:
            print("Creating new database directory.")
            os.makedirs(folder)
            os.makedirs(image_folder)
            os.makedirs(building_folder)
            os.makedirs(road_folder)
            datas = open(data_file,"w+")
            datas.close()
            vars = open(variable_file,"w+")
            vars.close()
            index = 0
        return folder, data_file, variable_file, image_folder, road_folder, building_folder, index

    def save_variables(self,variable_file):
        df_vars = pd.DataFrame(data = None, columns=["cord1","cord2","cord3","cord4","sample1","sample2"])
        data = [self.global_bounding_box[0],
                self.global_bounding_box[1],
                self.global_bounding_box[2],
                self.global_bounding_box[3],
                self.sample_bounding_box[0],
                self.sample_bounding_box[1]]

        df_vars.loc[0] = data
        df_vars["OSM_File"] = self.OSM_file
        df_vars.to_csv(variable_file)

    def get_index(self,data_file):
        """Generates the index that new samples should begin at in the .csv file.

        Parameters
        ----------
        data_file : string
            Path to the .csv file of the sample data.

        Returns
        -------
         : int
            Index that new samples should begin at.
        """
        file = open(data_file, "r")
        line_count = 0
        for line in file:
            if line != "\n":
                line_count += 1
        file.close()
        return line_count

    def save_image(self,folder,image,index = 0,assigned_name=None):
        """Saves an image to a designated directory path.

        Parameters
        ----------
        folder : string
            Path to directory image should be saved to.
        image : PIL.image.image
            Image that will be saved.
        index : int, optional
            Index of the image in the dataset, by default 0
        assigned_name : string, optional
            Designated name of the file if it is different then "image_index.png", by default None
        """
        name = "image_"+str(index)+".png"
        if assigned_name is not None:
            name = assigned_name
        
        image_name = os.path.join(folder, name)
        image.save(image_name,"PNG")

    def generate_area(self):
        """Generates a random sample from the map.

        Returns
        -------
        rectangle_coordinates : ndarray
            The (lon,lat) coordinates of the sample.
        start_point : ndarray
            The center of the sample.
        angle : int
            The angle that the sample is rotated clockwise in radians.
        
        """
        rectangle_coordinates = np.zeros((4,2))
        diagonal = np.sqrt(np.sum(np.power(self.sample_bounding_box,2)))*.5 # calculates half the diagonal of the rectangle
        angle = np.random.uniform(low = 0, high = 2*np.pi) # Generates random angle for the area the sample will be pulled from. 
        start_point = np.array((np.random.uniform(low = diagonal,high = self.global_image.size[0]-diagonal),np.random.uniform(low = diagonal, high = self.global_image.size[1]-diagonal)))
        rectangle_coordinates = self.get_rectangle_corners(start_point,angle)
        rectangle_coordinates = self.transforms.map_to_coordinate(rectangle_coordinates)
        start_point = self.transforms.map_to_coordinate(start_point)
        return rectangle_coordinates,start_point,angle # lon,lat

    def get_rectangle_corners(self,center,angle):
        """Generates the coordinates of a tilted rectangle.

        Parameters
        ----------
        center : ndarray
            The coordinates of the center of the rectangle.
        angle : int
            Angle the rectangle should be rotated clockwise in radians.

        Returns
        -------
         : array
            The coordinates of the four corners of the rectangle.
        """
        v1 = np.array((np.cos(angle),np.sin(angle)))
        v2 = np.array((-v1[1],v1[0]))
        v1*=self.sample_bounding_box[0]/2
        v2*=self.sample_bounding_box[1]/2
        return np.array(((center + v1 + v2),(center - v1 + v2),(center - v1 - v2),(center + v1 - v2))) # map coordinates

    def find_constrained_intersections(self,coordinates):
        """Finds all intersections inside of a polygon.

        Parameters
        ----------
        coordinates : ndarray
            Coordinates of the polygon region.

        Returns
        -------
        intersections : ndarray
            Coordinates of the intersections contained in the polygon.
        road_ways : GeoDataFrame
            Geopandas geodataframe containing the information of the roads within the polygon.
        buildings : GeoDataFrame
            Geopandas geodataframe containing the information of the building traces within the polygon.
        """
        poly = Polygon(coordinates)
        osm = pyrosm.OSM(self.OSM_file,bounding_box=poly)
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

    def choose_intersection(self,intersections,center,number_solution=1):
        """Chooses the intersection that an RSU should be placed at. Currently this is the closest intersection to the center of the sample.

        Parameters
        ----------
        intersections : ndarray
            Array of the candidate intersections.
        center : ndarray
            Coordinates of the center of the sample.
        number_solution : int, optional
            The number of intersections that should be chosen, by default 1

        Returns
        -------
         : ndarray
            The coordinates of the intersection that an RSU should be placed at.
        """
        # fig,ax = plt.subplots(1)  % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ax.scatter(intersections[:,0],intersections[:,1])
        # ax.scatter(center[0],center[1],color='g')
        # plt.draw()

        nearest = np.zeros((intersections.shape[0],3))
        nearest[:,:2] = np.copy(intersections)
        nearest[:,2] = np.linalg.norm(np.subtract(nearest[:,:2],center),axis = 1)
        return np.reshape(nearest[nearest[:,2] == nearest[:,2].min(),:2][0,:],(number_solution,2))

    def crop_image(self,coordinates,angle,intersections,solution):
        """Creates the image of the sample area by cropping it out of the map. 

        Parameters
        ----------
        coordinates : ndarray
            Coordinates of the sampled area that will be cropped from the map.
        angle : int
            Angle that the sampled area is rotated clockwise in radians.
        solution : ndarray
            Coordinates of the solution intersection.

        Returns
        -------
        img4 : PIL.image.image
            Image of the map of the sampled area.
         : ndarray
            Coordinates of the solution intersection, but transformed to the coordinate system of the new image of the sampled area.
        """
        coordinates = np.vstack((coordinates,intersections))
        coordinates = np.vstack((coordinates,solution))

        coordinates_map = self.transforms.coordinate_to_map(coordinates)
        bounds = self.getBounds(coordinates_map)
        img2 = self.global_image.crop(bounds.astype(int))

        bound_center = self.getBoundsCenter(bounds)
        crop_center = self.getCenter(img2)
        crop_points = np.apply_along_axis(self.recenter,1,coordinates_map,bound_center,crop_center)
        # In order for the osm data to be scalled correctly, it must be fit to this size. By pulling the information from here,
        # it makes it so this information dosnt need to be calculated unnecessarily again
        self.osm_image_size = self.getBounds(crop_points) 
        rotated_points = np.apply_along_axis(self.rotate,1,crop_points,crop_center,angle)
        img3 = img2.rotate(angle * 180 / np.pi, expand=True)

        im3_center = self.getCenter(img3)
        rotated_points = self.recenter(rotated_points,0,im3_center)
        img4 = img3.crop(self.getBounds(rotated_points).astype(int))
        im4_center = self.getCenter(img4)
        final_coords = self.recenter(rotated_points,im3_center,im4_center)

        if self.plot:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3)
            # fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
            # ax0.imshow(self.global_image,origin = 'lower')
            # ax0.scatter(coordinates_map[-1,0],coordinates_map[-1,1],color = 'r')
            # ax0.scatter(coordinates_map[:-1,0],coordinates_map[:-1,1],marker='x',color = 'k')
            ax1.imshow(img2,origin = 'lower')
            ax1.scatter(crop_points[-1,0],crop_points[-1,1],color = 'r') # Solution
            ax1.scatter(crop_points[:4,0],crop_points[:4,1],marker='x',color = 'b') # Corners
            ax1.scatter(crop_points[4:-1,0],crop_points[4:-1,1],marker='x',color = 'k') # Intersections
            ax1.scatter(crop_center[0],crop_center[1],marker='x',color='g') # Center
            ax2.imshow(img3,origin = 'lower')
            ax2.scatter(rotated_points[-1,0],rotated_points[-1,1],color = 'r') # Solution
            ax2.scatter(rotated_points[:4,0],rotated_points[:4,1],marker='x',color = 'b') # Corners
            ax2.scatter(rotated_points[4:-1,0],rotated_points[4:-1,1],marker='x',color = 'k') # Intersections
            ax2.scatter(im3_center[0],im3_center[1],marker='x',color='g') # Center
            ax3.imshow(img4,origin = 'lower')
            ax3.scatter(final_coords[-1,0],final_coords[-1,1],color = 'r') # Solution
            ax3.scatter(final_coords[:4,0],final_coords[:4,1],marker='x',color = 'b') # Corners
            ax3.scatter(final_coords[4:-1,0],final_coords[4:-1,1],marker='x',color = 'k') # Intersections
            ax3.scatter(im4_center[0],im4_center[1],marker='x',color='g') # Center
            ax2.set_title(angle)
            ax3.set_title(angle * 180 / np.pi)
            ax3.set_xlim(0,self.sample_bounding_box[0])
            ax3.set_ylim(0,self.sample_bounding_box[1])
            plt.draw()

        return img4, final_coords[-1,:].astype(int)

    def crop_OSM_image(self,coordinates,angle,OSM_data):
        """Creates an image of the OSM data for the samples area.

        Parameters
        ----------
        coordinates : ndarray
            Coordinates of the sampled area that the OSM data is from.
        angle : int
            Angle that the sampled area is rotated clockwise in radians.
        OSM_data : GeoDataFrame
            Geopandas GeoDataFrame containing OSM data that will be plotted for the image.

        Returns
        -------
        img4 : PIL.image.image
            Image of the OSM data for the sample.
        """
        bounds = self.getBounds(coordinates)
        img2 = self.OSM_fig_to_img(OSM_data,coordinates)
        crop_points = self.coordinate_transform(coordinates,bounds,img2.size)
        bounds = self.getBounds(crop_points)
        crop_center = self.getCenter(img2)
        rotated_points = np.apply_along_axis(self.rotate,1,crop_points,crop_center,angle)
        img3 = img2.rotate(angle * 180 / np.pi, expand=True)
        im3_center = self.getCenter(img3)
        rotated_points = self.recenter(rotated_points,0,im3_center)
        img4 = img3.crop(self.getBounds(rotated_points).astype(int))
        im4_center = self.getCenter(img4)
        final_coords = self.recenter(rotated_points,im3_center,im4_center)

        # if self.plot:
        #     fig,(ax1,ax2,ax3) = plt.subplots(3,1)
        #     ax1.imshow(img2,origin = 'lower')
        #     ax1.scatter(crop_points[:,0],crop_points[:,1],marker='x',color = 'k')
        #     ax1.scatter(crop_center[0],crop_center[1],marker='x',color='g')
        #     ax2.imshow(img3,origin = 'lower')
        #     ax2.scatter(rotated_points[:,0],rotated_points[:,1],marker='x',color = 'k')
        #     ax2.scatter(im3_center[0],im3_center[1],marker='x',color='g')
        #     ax3.imshow(img4,origin = 'lower')
        #     ax3.scatter(final_coords[:,0],final_coords[:,1],marker='x',color = 'k')
        #     ax2.set_title(angle)
        #     ax3.set_title(angle * 180 / np.pi)
        #     ax3.set_xlim(0,self.sample_bounding_box[0])
        #     ax3.set_ylim(0,self.sample_bounding_box[1])
        #     plt.draw()

        return img4

    def add_data(self,coordinates,center,angle,solution):
        """Adds the data of the current sample to the database of sample data.

        Parameters
        ----------
        coordinates : ndarray
            Coordinates of the corners of the sample.
        center : ndarray
            Coordinate of the center of the sample.
        angle : int
            Angle that the sampled area is rotated clockwise in radians.
        solution : ndarray
            Coordinates of the solution intersection for the sample.
        """
        data_array = pd.Series([*coordinates.flatten(),*center.flatten(),angle,*solution.flatten()],self.df_data.columns)
        
        self.df_data = self.df_data.append(data_array,ignore_index=True) 
        
        # self.df_data = pd.concat((self.df_data,data_array),ignore_index=True,axis=0, join='outer')

        # self.df_data.loc[len(self.df_data)] = data_array.tolist()
        if self.df_data.shape[0] != 0 and self.df_data.iloc[len(self.df_data)-1]['solution2'] != data_array[-1]:
            return True
        else:
            return False

    def df_to_csv(self,index,data_file):
        """Adds the data in the dataframe to the specified .csv file.

        Parameters
        ----------
        index : int
            The index that the new data should begin at in the .csv.
        data_file : string
            The file path of the .csv file that the new data should be added to.
        """
        self.df_data.index = np.arange(self.i_init,index+1)
        if index == 0:
            self.df_data[index:index+1].to_csv(data_file, mode='a')
        else:
            self.df_data[-1:].to_csv(data_file, mode='a', header=False)

    def OSM_fig_to_img(self,OSM_data,coordinates):
        """Plots OSM data as a figure then converts it to a PIL image.

        Parameters
        ----------
        OSM_data : GeoDataFrame
            Geopandas GeoDataFrame containing the OSM data.
        coordinates : ndarray
            The coordinates that bound the OSM data that should be plotted.

        Returns
        -------
         : PIL.image.image
            A PIL image of the OSM data.
        """
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

    def getBounds(self,points):
        """Finds the bounds of a set of points.

        Parameters
        ----------
        points : ndarray
            A set of coordinates for each point of data.

        Returns
        -------
         : ndarray
            Array of the bounds of the points in the form (left, upper, right, lower).
        """
        xs, ys = zip(*points)
        # left, upper, right, lower using the usual image coordinate system
        # where top-left of the image is 0, 0
        return np.array((min(xs), min(ys), max(xs), max(ys)))

    def getCenter(self,im):
        """Finds the coordinates of the center of an image.

        Parameters
        ----------
        im : PIL.image.image
            A PIL image that you want to find the center of.

        Returns
        -------
         : ndarray
            The coordinates of the center of the image.
        """
        return np.divide(im.size,2)

    def getBoundsCenter(self,bounds):
        """Finds the coordinates of the center of the bounds.

        Parameters
        ----------
        bounds : ndarray
            An array of the bounds in the form (left, upper, right, lower).

        Returns
        -------
         : ndarray
            The coordinates of the center of the bounds.
        """
        return np.array(((bounds[2] - bounds[0]) / 2 + bounds[0],(bounds[3] - bounds[1]) / 2 + bounds[1]))
        
    def recenter(self, coord, old_center, new_center):
        """Translates a set of coordinates based on the difference between the old and new centers.

        Parameters
        ----------
        coord : ndarray
            The set of coordinates that will be translated.
        old_center : ndarray
            The coordinates of the old center.
        new_center : ndarray
            The coordinates of the new center.

        Returns
        -------
         : ndarray
            The new set of coordinates that have been translated.
        """
        return np.add(coord,np.subtract(new_center,old_center))

    def rotate(self, coord, center, angle):
        """Rotates a set of coordinates around a center.

        Parameters
        ----------
        coord : ndarray
            Array containing the coordinates that will be rotated.
        center : ndarray
            Coordinates of the point to be rotated around.
        angle : int
            Angle that the coordinates will be rotated clockwise in radians.

        Returns
        -------
         : ndarray
            Array of rotated coordinates
        """
        rotation_matrix = np.array(((np.cos(angle),-np.sin(angle)),(np.sin(angle),np.cos(angle))))
        coord = self.recenter(coord,center,0)
        rotated_coords = np.matmul(coord,rotation_matrix)
        return rotated_coords

    def coordinate_transform(self,coordinates,old_range,new_range): # This will ONLY work for self.getBounds --> Image size
        """Scales a set of coordinates between two ranges.

        Parameters
        ----------
        coordinates : ndarray
            Array of coordinates that will be scaled.
        old_range : ndarray
            The range that the coordinates are currently scaled to.
        new_range : ndarray
            The range that the coordinates will be converted to.

        Returns
        -------
         : ndarray
            The array of coordinates scaled to the new range.
        """
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

current_path = str(pathlib.Path.cwd())
folder_path = 'Code'
if folder_path in current_path:
    OSM_file = os.path.join(current_path,"austin_downtown.pbf")
else:
    path = os.path.join(current_path,folder_path)
    OSM_file = os.path.join(path,"austin_downtown.pbf")
# map_image = os.path.join(path,"Map")
folder_name = "test_data"
data_generator = Dataset_Generator(bbox,data_size,folder_name,OSM_file)
number_of_samples = 1

data_generator(number_of_samples,save_data = False,plot=True)