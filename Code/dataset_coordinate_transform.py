import numpy as np
import matplotlib.pyplot as plt
from numpy import core


class Dataset_Transformation():
    """
    Class containing transformations needed to convert between the following coordinate systems:
        Coordinates = (lon,lat)
        Global Map Array = Global map image size
        Sampled Map Array = Data image size
        CNN Output = (0,1)
    Examples of these functions being used can be found in the Dataset_Generator class.

    Parameters
         ----------
         map_boundry : ndarray
            Two corners used to bound the global map in the form of (lon,lat).

         image_size : array
            Width and height of the global map image
        
         data_size : array
            Width and height of the samples recorded from the global map.
    """


    def __init__(self,map_boundry,image_size,data_size):
        self.boundry = map_boundry
        self.data_size = data_size
        self.image_size = image_size

    def map_to_coordinate(self,old_coords,method = "unity scale"): 
        """
         Converts coordinates from global map coordinates to lat,long.
         Parameters
         ----------
         old_coords : ndarray
            A column array of coordinates in the global image coordinate system.
        
         Returns
         -------
         new_coords : ndarray
            A column array of longitudes and latitudes.
        """
        new_coords = np.zeros(old_coords.shape)
        if len(new_coords.shape) == 1:
            new_coords = np.reshape(new_coords, (1,len(new_coords)))
            old_coords = np.reshape(old_coords, (1,len(old_coords)))
        if method == "unity scale":

            new_coords[:,0] = old_coords[:,0]/self.image_size[0]
            new_coords[:,1] = old_coords[:,1]/self.image_size[1]

            new_coords[:,0] = new_coords[:,0]*(np.max((self.boundry[0],self.boundry[2]))-np.min((self.boundry[0],self.boundry[2])))
            new_coords[:,1] = new_coords[:,1]*(np.max((self.boundry[1],self.boundry[3]))-np.min((self.boundry[1],self.boundry[3])))

            new_coords[:,0] = new_coords[:,0]+np.min((self.boundry[0],self.boundry[2]))
            new_coords[:,1] = new_coords[:,1]+np.min((self.boundry[1],self.boundry[3]))

        return new_coords
 
    def coordinate_to_map(self,old_coords,method = "unity scale"): 
        """
         Converts coordinates from lat,long to global map coordinates

         Parameters
         ----------
         old_coords : ndarray
            A column array of longitudes and latitudes.
        

         Returns
         -------
         new_coords : ndarray
            A column array of coordinates in the global image coordinate system.
        """
        new_coords = np.zeros(old_coords.shape)
        if len(new_coords.shape) == 1:
            new_coords = np.reshape(new_coords, (1,len(new_coords)))
            old_coords = np.reshape(old_coords, (1,len(old_coords)))
        if method == "interpolation":
            lon_array1 = np.linspace(start = self.boundry[0],stop = self.boundry[2],num = self.image_size[0])
            lon_array2 = np.arange(self.image_size[0])
            new_coords[:,0] = np.interp(old_coords[:,0],lon_array1,lon_array2)

            lat_array1 = np.linspace(start = self.boundry[3],stop = self.boundry[1],num = self.image_size[1])
            lat_array2 = np.arange(self.image_size[1])
            new_coords[:,1] = np.interp(old_coords[:,1],lat_array1,lat_array2)
        elif method == "unity scale":
            new_coords[:,0] = old_coords[:,0]-np.min((self.boundry[0],self.boundry[2]))
            new_coords[:,1] = old_coords[:,1]-np.min((self.boundry[1],self.boundry[3]))

            new_coords[:,0] = new_coords[:,0]/(np.max((self.boundry[0],self.boundry[2]))-np.min((self.boundry[0],self.boundry[2])))
            new_coords[:,1] = new_coords[:,1]/(np.max((self.boundry[1],self.boundry[3]))-np.min((self.boundry[1],self.boundry[3])))

            new_coords[:,0] = new_coords[:,0]*self.image_size[0]
            new_coords[:,1] = new_coords[:,1]*self.image_size[1]

        return new_coords

    def map_to_sample(self,coordinates,boundry_coordinates,angle): 
        """
         Converts coordinates from global map coordinates to sampled map coordinates

         Parameters
         ----------
         coordinates : ndarray
            Coordinates that are to be converted from the global map coordinates to the sampled map coordinates.

         boundry_coordinates : ndarray
            Coordinates for the four corners of the sample. These points should be in the global map coordinate system. 

         angle : float
            Angle in radians that the sample is rotated. The rotation is clockwise. 

         Returns
         -------
         final_coords : ndarray
            Coordinates in the sampled map coordinate system. It will have the same size as the coordinates input.
        """
        if len(coordinates.shape) == 1:
            coordinates = np.reshape(coordinates,(1,len(coordinates)))
        coordinates = np.vstack((boundry_coordinates,coordinates))
        bounds = self.getBounds(coordinates)
        bound_center = self.getBoundsCenter(bounds)
        crop_center = np.subtract(bound_center,[bounds[0],bounds[1]])
        crop_points = np.apply_along_axis(self.recenter,1,coordinates,bound_center,crop_center)
        rotated_points = np.apply_along_axis(self.rotate,1,crop_points,crop_center,angle)
        final_coords = self.recenter(rotated_points,0,[self.data_size[0]/2,self.data_size[1]/2])
        return final_coords[4:,:]

    def sample_to_map(self,coordinates,boundry_coordinates,angle): 
        """
         Converts coordinates from sampled map coordinates to global map coordinates

         Parameters
         ----------
         coordinates : ndarray
            Coordinates that are to be converted from the sampled map coordinates to the global map coordinates.

         boundry_coordinates : ndarray
            Coordinates for the four corners of the sample. These points should be in the global map coordinate system. 

         angle : float
            Angle in radians that the sample is rotated. The rotation is clockwise. 

         Returns
         -------
         final_coords : ndarray
            Coordinates in the global map coordinate system. It will have the same size as the coordinates input.
        """
        if len(coordinates.shape) == 1:
            coordinates = np.reshape(coordinates,(1,2))
        bound_coords = np.array(((250,0),(0,500),(0,0),(250,500)))
        coordinates = np.vstack((bound_coords,coordinates))
        bounds_coords = self.getBounds(boundry_coordinates)
        bounds_center = self.getBoundsCenter(bounds_coords)
        center = np.array(( self.data_size[0]/2 , self.data_size[1]/2 ))
        rotated_points = np.apply_along_axis(self.rotate,1,coordinates,center,-angle)
        bounds2 = self.getBounds(rotated_points)
        center2 = self.getBoundsCenter(bounds2)
        final_coords = self.recenter(rotated_points,center2,bounds_center)

        return final_coords[4:,:]

    def data_to_coordinate(self,data): 
        """
         Converts the data points retrieved from the Overpass API for OpenStreetMap to lon,lat.

         Parameters
         ----------
         
         data : dict
            Data returned from an Overpass API query. 

         Returns
         -------
         coords : ndarray
            Coordinates in lon,lat of nodes in the data. These are the lon,lat coordinates of the intersections in the sample area. 
         
        """
        coords = []
        for element in data['elements']:
            if element['type'] == 'node':
                lon = element['lon']
                lat = element['lat']
                coords.append((lon, lat))
            elif 'center' in element:
                lon = element['center']['lon']
                lat = element['center']['lat']
                coords.append((lon, lat))
        coords = np.array(coords)
        return coords

    def sample_to_output(self,coordinates): 
        """
         Converts coordinates from sampled map coordinates to CNN output. 

         Parameters
         ----------
         coordinates : ndarray
            Coordinates in the sampled map coordinate system.

         Returns
         -------
         coordinates : ndarray
            Coordinates with values scaled between 0 and 1. Scaling is done based on the size of the sampled region.
         
        """

        if len(coordinates.shape) == 1:
            coordinates = np.reshape(coordinates,(2,1))
        coordinates[:,0] = np.divide(coordinates[:,0],self.data_size[0])
        coordinates[:,1] = np.divide(coordinates[:,1],self.data_size[1])
        return coordinates

    def output_to_sample(self,coordinates): 
        """
         Converts coordinates from CNN output to sampled map coordinates. 

         Parameters
         ----------
         coordinates : ndarray
            Coordinates with values scaled between 0 and 1. Scaling is done based on the size of the sampled region.

         Returns
         -------
         coordinates : ndarray
            Coordinates in the sampled map coordinate system.
         
        """
        if len(coordinates.shape) == 1:
            coordinates = np.reshape(coordinates,(1,2))
        coordinates[:,0] = np.multiply(coordinates[:,0],self.data_size[0])
        coordinates[:,1] = np.multiply(coordinates[:,1],self.data_size[1])
        return coordinates

    def output_to_map(self,coordinates,boundry_coordinates,angle):
        """
         Converts coordinates from CNN output to global map coordinates. 

         Parameters
         ----------
         coordinates : ndarray
            Coordinates with values scaled between 0 and 1. Scaling is done based on the size of the sampled region.
         
         boundry_coordinates : ndarray
            Coordinates for the four corners of the sample. These points should be in the global map coordinate system. 

         angle : float
            Angle in radians that the sample is rotated. The rotation is clockwise. 

         Returns
         -------
         map_coordinates : ndarray
            Coordinates in the global map coordinate system.
         
        """
        if len(coordinates.shape) == 1:
            coordinates = np.reshape(coordinates,(1,2))
        sample_coordinates = self.output_to_sample(coordinates)
        bound_coords = np.array(((250,0),(0,500),(0,0),(250,500)))
        sample_coordinates = np.vstack((bound_coords,sample_coordinates))
        map_coordinates = self.sample_to_map(sample_coordinates,boundry_coordinates,angle)
        return map_coordinates[4:,:]
    
    def map_to_output(self,coordinates,boundry_coordinates,angle):
        """
         Converts coordinates from global map coordinates to CNN output. 

         Parameters
         ----------
         coordinates : ndarray
            Coordinates in the global map coordinate system.   

         boundry_coordinates : ndarray
            Coordinates for the four corners of the sample. These points should be in the global map coordinate system. 

         angle : float
            Angle in radians that the sample is rotated. The rotation is clockwise. 
            
         Returns
         -------
         output_coordinates : ndarray
            Coordinates with values scaled between 0 and 1. Scaling is done based on the size of the sampled region.
         
        """
        sample_coordinates = self.map_to_sample(coordinates,boundry_coordinates,angle)
        output_coordinates = self.sample_to_output(sample_coordinates)
        return output_coordinates

    def prepare_dataset(self,dataset):
        """
         Prepares the numpy array from the .csv for use with a CNN. Converts all the coordinates to the map coordinate system, then converts the solution to the output coordinate system.

         Parameters
         ----------
         dataset : ndarray
            Dataset from the .csv.   

         Returns
         -------
         output_coordinates : ndarray
            Dataset with the coordinates transformed. 
         
        """
        dataset[:,1:3] = self.coordinate_to_map(dataset[:,1:3])
        dataset[:,3:5] = self.coordinate_to_map(dataset[:,3:5])
        dataset[:,5:7] = self.coordinate_to_map(dataset[:,5:7])
        dataset[:,7:9] = self.coordinate_to_map(dataset[:,7:9])
        dataset[:,9:11] = self.coordinate_to_map(dataset[:,9:11])
        dataset[:,12:] = self.coordinate_to_map(dataset[:,12:])
        for i in range(dataset.shape[0]):
            boundry_coordinates = dataset[i,1:9]
            dataset[i,12:] = self.map_to_output(dataset[i,12:],boundry_coordinates=np.reshape(boundry_coordinates,(4,2)),angle=dataset[i,11])
        return dataset

    def prepare_data(self,data):
        """
         Prepares the numpy array from the .csv for use with a CNN. Converts all the coordinates to the map coordinate system, then converts the solution to the output coordinate system.

         Parameters
         ----------
         dataset : ndarray
            Dataset from the .csv.   

         Returns
         -------
         output_coordinates : ndarray
            Dataset with the coordinates transformed. 
         
        """
        data_numpy = data.to_numpy()
        
        data_numpy[1:3] = self.coordinate_to_map( data_numpy[1:3])
        data_numpy[3:5] = self.coordinate_to_map( data_numpy[3:5])
        data_numpy[5:7] = self.coordinate_to_map( data_numpy[5:7])
        data_numpy[7:9] = self.coordinate_to_map( data_numpy[7:9])
        data_numpy[9:11] = self.coordinate_to_map( data_numpy[9:11])
        data_numpy[12:] = self.coordinate_to_map( data_numpy[12:])
        
        boundry_coordinates =  data_numpy[1:9]
        data_numpy[12:] = self.map_to_output(data_numpy[12:],boundry_coordinates=np.reshape(boundry_coordinates,(4,2)),angle= data_numpy[11])
        data[:] = data_numpy
        return data
# The below functions are for the transformations above

    def getBounds(self,points):
        xs, ys = zip(*points)
        # left, upper, right, lower using the usual image coordinate system
        # where top-left of the image is 0, 0
        return np.array((min(xs), min(ys), max(xs), max(ys)))

    def getBoundsCenter(self,bounds):
        return np.array(((bounds[2] - bounds[0]) / 2 + bounds[0],(bounds[3] - bounds[1]) / 2 + bounds[1]))

    def getRelativeCenter(self,bounds):
        return np.array((np.abs(bounds[0]-bounds[2])/2,np.abs(bounds[1]-bounds[3])/2))

    def getBoundsLocalCoordinates(self,bounds):
        width = bounds[2]-bounds[1]
        height = bounds[1]-bounds[3]
        return np.array(((0,0),(0,height),(width,0),(width,height)))
        
    def recenter(self, coord, old_center, new_center):
        return np.add(coord,np.subtract(new_center,old_center))

    def rotate(self, coord, center, angle):
        # angle should be in radians
        rotation_matrix = np.array(((np.cos(angle),-np.sin(angle)),(np.sin(angle),np.cos(angle))))
        coord = self.recenter(coord,center,0)
        rotated_coords = np.matmul(coord,rotation_matrix)
        return rotated_coords