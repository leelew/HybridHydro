import shapefile
import shapely.geometry as geometry
import numpy as np
import netCDF4 as nc


path = '/tera03/lilu/work/CLFS/outputs/'

f = nc.Dataset(path + 'SMAP_L4_SSM_20150531.nc')
lat_smap = np.array(f['latitude'][:])
lon_smap = np.array(f['longitude'][:])
print("lat shape is {} and lon shape is {}".format(lat_smap.shape, lon_smap.shape))

in_china_shape = np.zeros((len(lat_smap), len(lon_smap)))
sz_shp = shapefile.Reader('china-shapefiles/china_country')
for city_rcd in sz_shp.shapeRecords():   
    if city_rcd.record[5] == 'CHINA':  
        sz_shp = city_rcd.shape 

grid_lon, grid_lat = np.meshgrid(lon_smap, lat_smap)  
flat_lon = grid_lon.flatten()  
flat_lat = grid_lat.flatten()
flat_points = np.column_stack((flat_lon, flat_lat))
in_shape_points = []

for pt in flat_points:
        if geometry.Point(pt).within(geometry.shape(sz_shp)):
            in_shape_points.append(pt)
            lo_index, la_index = list(lon_smap).index(pt[0]),list(lat_smap).index(pt[1])
            in_china_shape[la_index,lo_index] = 1
            print("The point is in CHINA------------------------------------------------{}".format(pt))
        else:
            print("The point is not in CHINA")
np.save("mask_in_china.npy",in_china_shape)
