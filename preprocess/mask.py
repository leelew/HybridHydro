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
for city_rcd in sz_shp.shapeRecords():   # 遍历每一条shaperecord
    if city_rcd.record[5] == 'CHINA':  # 遍历时，record字段是地区的信息（由字符串表示）
        sz_shp = city_rcd.shape# 遍历时，shape字段是shape——形状（由点组成）

grid_lon, grid_lat = np.meshgrid(lon_smap, lat_smap)  # 构成了一个坐标矩阵，实际上也就是一个网格，两者是同型矩阵
flat_lon = grid_lon.flatten()  # 将坐标展成一维
flat_lat = grid_lat.flatten()
flat_points = np.column_stack((flat_lon, flat_lat))# np.column_stack((a,b)):   向矩阵a增加列，b是增加的部分，将1维数组转换成2维，这样flat的每个点对应上面xi,yi的所有点
in_shape_points = []

for pt in flat_points:# 此处第一维度是经度，第二维是维度
        if geometry.Point(pt).within(geometry.shape(sz_shp)):
            in_shape_points.append(pt)
            lo_index, la_index = list(lon_smap).index(pt[0]),list(lat_smap).index(pt[1])
            in_china_shape[la_index,lo_index] = 1
            print("The point is in CHINA------------------------------------------------{}".format(pt))
        else:
            print("The point is not in CHINA")
np.save("mask_in_china.npy",in_china_shape)
