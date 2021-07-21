import numpy as np
import matplotlib.pyplot as plt


class Dataset_Transformation():
    """
    Class containintg transformations needed to covert between the following coordinate systems:
        Coordinates = (lon,lat)
        Global Map Array = Global map image size
        Sampled Map Array = Data image size
        CNN Output = (0,1)
    Examples of these functions being used can be found in the Dataset_Generator class.
    """


    def __init__(self,map_boundry,image_size,data_size):
        self.boundry = map_boundry
        self.data_size = data_size
        self.image_size = image_size

    def map_to_coordinate(self,old_coords): 
        """
         Converts coordinates from global map coordinates to lat,long

         Parameters
         ----------
         old_coords : ndarray
            A column array of longitudes and latitudes.

         Returns
         -------
         new_coords : ndarray
            A column array of coordinates scaled to the global image size. It should be the same size as the input.
        """

        new_coords = np.zeros(old_coords.shape)
        if len(new_coords.shape) == 1:
            new_coords = np.reshape(new_coords, (1,len(new_coords)))
            old_coords = np.reshape(old_coords, (1,len(old_coords)))
        lon_array1 = np.linspace(start = self.boundry[0],stop = self.boundry[2],num = self.image_size[0])
        lon_array2 = np.arange(self.image_size[0])
        new_coords[:,0] = np.interp(old_coords[:,0],lon_array2,lon_array1)

        lat_array1 = np.linspace(start = self.boundry[3],stop = self.boundry[1],num = self.image_size[1])
        lat_array2 = np.arange(self.image_size[1])
        new_coords[:,1] = np.interp(old_coords[:,1],lat_array2,lat_array1)

        return new_coords
 
    def coordinate_to_map(self,old_coords): 
        """
         Converts coordinates from lat,long to global map coordinates

         Parameters
         ----------
         new_coords : ndarray
            A column array of coordinates in the global image coordinate system.
        

         Returns
         -------
         old_coords : ndarray
            A column array of longitudes and latitudes.
        """

        new_coords = np.zeros(old_coords.shape)
        if len(new_coords.shape) == 1:
            new_coords = np.reshape(new_coords, (1,len(new_coords)))
            old_coords = np.reshape(old_coords, (1,len(old_coords)))
        lon_array1 = np.linspace(start = self.boundry[0],stop = self.boundry[2],num = self.image_size[0])
        lon_array2 = np.arange(self.image_size[0])
        new_coords[:,0] = np.interp(old_coords[:,0],lon_array1,lon_array2)

        lat_array1 = np.linspace(start = self.boundry[3],stop = self.boundry[1],num = self.image_size[1])
        lat_array2 = np.arange(self.image_size[1])
        new_coords[:,1] = np.interp(old_coords[:,1],lat_array1,lat_array2)

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
            Angle in radians that the sample is rotated. The rotation is counter-clockwise. 
         Returns
         -------
         final_coords : ndarray
            Coordinates in the sampled map coordinate system. It will have the same size as the coordinates input.
        """
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
            Angle in radians that the sample is rotated. The rotation is counter-clockwise. 
         Returns
         -------
         final_coords : ndarray
            Coordinates in the global map coordinate system. It will have the same size as the coordinates input.
        """
        if len(coordinates.shape) == 1:
            coordinates = np.reshape(coordinates,(2,1))
        bounds_coords = self.getBounds(boundry_coordinates)
        bounds_center = self.getBoundsCenter(bounds_coords)
        center = np.array(( self.data_size[0]/2 , self.data_size[1]/2 ))
        rotated_points = np.apply_along_axis(self.rotate,1,coordinates,center,-angle)
        bounds2 = self.getBounds(rotated_points)
        center2 = self.getBoundsCenter(bounds2)
        final_coords = self.recenter(rotated_points,center2,bounds_center)
        return final_coords

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
            coordinates = np.reshape(coordinates,(2,1))
        coordinates[:,0] = np.multiply(coordinates[:,0],self.data_size[0])
        coordinates[:,1] = np.multiply(coordinates[:,1],self.data_size[1])
        return coordinates

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