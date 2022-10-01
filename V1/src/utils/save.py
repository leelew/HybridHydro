
from osgeo import gdal
import os

class Matrix2Raster():

    # read raster file
    def read_img(self, filename):
        dataset = gdal.Open(filename)

        im_width = dataset.RasterXSize  # cols
        im_height = dataset.RasterYSize  # ros

        im_geotrans = dataset.GetGeoTransform()  # transform_array
        im_proj = dataset.GetProjection()  # projection
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # im_data

        del dataset
        return im_proj, im_geotrans, im_data

    # write raster file
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # Datatype
        # gdal.GDT_Byte,
        #gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        #gdal.GDT_Float32, gdal.GDT_Float64

        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # array dimensions
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # create file
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])

        del dataset

    # write raster file
    def write_img_without_projection(self, filename, im_data):

        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # array dimensions
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # create file
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            filename, im_width, im_height, im_bands, datatype)

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])

        del dataset


class SaveImage():
    def __init__(self):
        pass

    def __call__(self, X, *args: Any, **kwds: Any) -> Any:
        

