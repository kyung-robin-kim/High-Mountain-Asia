#Import functions
import pandas as pd
import xarray as xr
import geopandas as gpd
import numpy as np
import glob


#POINT ANALYSIS
#https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx


#Sets for validation:

# SET A. Monthly & resampled to Annual (Surface & Air): \validation_monthly
# 1. AIRS DownScaled 
# 2. MODMYD11 (filtered)
# 3. MYD11-AIRS GF
# 4. MODMYD11-AIRS GF

# SET B. Monthly & resampled to Annual (Ground): \validation_monthly
# 1. MODMYD11-AIRS GST
# 2. MODMYD11-AIRS TTOP
# 3. MYD11-AIRS GST 
# 4. MYD11-AIRS TTOP

# SET B_LIT. Annual (Ground): E:\processed-data\HMA_Permafrost\MAGT_litrev
# 1. MODMYD11-AIRS GST
# 2. MODMYD11-AIRS TTOP


'''
# SET C. Daily & resampled to Monthly (MODIS LST for Surface and Air): \MODIS_LST\validation
# 1. MOD11 (non-filtered)
# 2. MYD11 (non-filtered)
# 3. Zhang MYD11 GF
'''



#NOAA GHCN
#index 61 - Toktogul, Kirgiz KG - does not contain temperature data
csv_files = sorted(glob.glob(r'\NOAA_GHCN\*.csv'))[:-1]
ghcn_points = pd.read_csv(r'\\validation_points.csv')
#ICIMOD
path = r'\HMA\ICIMOD'
xcel_file = sorted(glob.glob(path+'\\*\\data\\*.xlsx'))[1]
xcel_sheets = pd.ExcelFile(xcel_file).sheet_names
xcel_data = [pd.read_excel(xcel_file,sheet_name=sheet) for sheet in xcel_sheets]
icimod_points =  xcel_data[0].iloc[:,1:6]
#ICIMOD Snow
path = r'\HMA\ICIMOD'
icimod_snow_points = pd.read_csv(sorted(glob.glob(path+'\\*.csv'))[0]).loc[1:]
#HiWAT
stations = pd.read_excel(r'\frozen_ground_obs\Station location.xlsx')
hiwat_points = gpd.GeoDataFrame(stations, columns=['station','Longtitude', 'Latitude'], geometry=gpd.points_from_xy(stations.Longtitude,stations.Latitude)).set_crs('epsg:4326')
#Zhao2021
stations = pd.read_csv(r'\Zhao_2021_ALL\stations.csv')
zhao2021_points = gpd.GeoDataFrame(stations, columns=['Site','Longitude', 'Latitude'], geometry=gpd.points_from_xy(stations.Longitude,stations.Latitude)).set_crs('epsg:4326')
#zhao2021_points=zhao2021_points.iloc[sorted([1,10,57,90,38,35,43,71,34,64,37,23])]
#Others: Wu & Qin & Wani 
others_points = pd.read_csv(r'\Permafrost\others.csv').iloc[:-12,:]
#ISMN (only 5cm or 80 cm (20 & 40 cm not included))
ismn_points = pd.read_csv(r'\ISMN\05_cm_ts.csv')
#Ma 2020
ma_points = pd.read_csv(r'\Ma_ESSD\stations.csv')



#SET A
raster_array =xr.open_dataset(r'\MODMYD11_monthly.nc').MLST
save_path = r'\validation_monthly'

airs = r'\AIRS1km_monthly.nc'
modmyd11 = r'\MODMYD11_monthly.nc'
myd11_airs = r'\MYD11_AIRS_monthly.nc'
modmyd11_airs = r'\MODMYD11_AIRS_monthly.nc'


val_points_lat = [ghcn_points.latitude,icimod_points.Lat,icimod_snow_points.Y,hiwat_points.Latitude,zhao2021_points.Latitude,others_points.LAT,ismn_points.lat,ma_points.Latitude]
val_points_lon = [ghcn_points.longitude,icimod_points.Lon,icimod_snow_points.X,hiwat_points.Longtitude,zhao2021_points.Longitude,others_points.LON,ismn_points.lon,ma_points.Longitude]
validation_ids = [ghcn_points.Name, icimod_points.NAME, icimod_snow_points.Station,hiwat_points.station, zhao2021_points.Site, 
                  pd.DataFrame(['{}_{}'.format(PI,code) for PI,code in zip(others_points.PI,others_points.Code)]).iloc[:,0], ismn_points.site, ma_points.Site]
paths = ['\\GHCN','\\ICIMOD','\\ICIMOD','\\HiWAT','\\Zhao2021','\\Others','\\ISMN','\\Ma2020']
queries = [[(find_nearest(raster_array.y,lat)[1],find_nearest(raster_array.x,lon)[1]) for lat,lon in zip(lats,lons)] for lats,lons in zip(val_points_lat,val_points_lon)]

for points, path, validation in zip(queries,paths,validation_ids):
    for point, i in zip(points,range(0,len(points))):

        airs_ds = xr.open_mfdataset(airs,parallel=True,chunks={"y":225,"x":225}).MST[:,point[0],point[1]].rename('AIRS')
        modmyd11_ds = xr.open_mfdataset(modmyd11,parallel=True,chunks={"y":225,"x":225}).MLST[:,point[0],point[1]].rename('MODMYD11')
        myd11_airs_ds = xr.open_mfdataset(myd11_airs,parallel=True,chunks={"y":225,"x":225}).MLST[:,point[0],point[1]].rename('MYD11AIRS')
        modmyd11_airs_ds = xr.open_mfdataset(modmyd11_airs,parallel=True,chunks={"y":225,"x":225}).MLST[:,point[0],point[1]].rename('MODMYD11AIRS')

        datetimes = pd.date_range(start='1/1/2003', end='1/1/2017', freq='M')
        surface_ds = pd.DataFrame([data.to_dataframe().iloc[:,-1] for data in [airs_ds,modmyd11_ds,myd11_airs_ds,modmyd11_airs_ds]]).T
        surface_ds.to_csv(save_path+path+'\setA_{}.csv'.format(validation.iloc[i]))



#SET B
raster_array =xr.open_dataset(r'\MODMYD11_monthly.nc').MLST
save_path = r'\validation_monthly'

myd11_airs_nival = r'\MYD11_AIRS_monthly.nc'
modmyd11_airs_nival = r'\MODMYD11_AIRS_monthly.nc'
myd11_airs_thermal = r'\MYD11_AIRS_monthly.nc'
modmyd11_airs_thermal = r'\MODMYD11_AIRS_monthly.nc'

val_points_lat = [icimod_points.Lat,icimod_snow_points.Y,hiwat_points.Latitude,zhao2021_points.Latitude,others_points.LAT,ismn_points.lat,ma_points.Latitude]
val_points_lon = [icimod_points.Lon,icimod_snow_points.X,hiwat_points.Longtitude,zhao2021_points.Longitude,others_points.LON,ismn_points.lon,ma_points.Longitude]
validation_ids = [icimod_points.NAME, icimod_snow_points.Station,hiwat_points.station, zhao2021_points.Site, 
                  pd.DataFrame(['{}_{}'.format(PI,code) for PI,code in zip(others_points.PI,others_points.Code)]).iloc[:,0], ismn_points.site, ma_points.Site]
paths = ['\\ICIMOD','\\ICIMOD','\\HiWAT','\\Zhao2021','\\Others','\\ISMN','\\Ma2020']
queries = [[(find_nearest(raster_array.y,lat)[1],find_nearest(raster_array.x,lon)[1]) for lat,lon in zip(lats,lons)] for lats,lons in zip(val_points_lat,val_points_lon)]

for points, path, validation in zip(queries,paths,validation_ids):
    for point, i in zip(points,range(0,len(points))):

        nival_A = xr.open_mfdataset(myd11_airs_nival,parallel=True,chunks={"y":225,"x":225}).MGT[:,point[0],point[1]].rename('MYDAIRS Nival')
        nival_B = xr.open_mfdataset(modmyd11_airs_nival,parallel=True,chunks={"y":225,"x":225}).MGT[:,point[0],point[1]].rename('MODMYDAIRS Nival')
        thermal_A = xr.open_mfdataset(myd11_airs_thermal,parallel=True,chunks={"y":225,"x":225}).MGT[:,point[0],point[1]].rename('MYDAIRS Thermal')
        thermal_B = xr.open_mfdataset(modmyd11_airs_thermal,parallel=True,chunks={"y":225,"x":225}).MGT[:,point[0],point[1]].rename('MODMYDAIRS Thermal')

        datetimes = pd.date_range(start='1/1/2003', end='1/1/2017', freq='M')
        gt_ds = pd.DataFrame([data.to_dataframe().iloc[:,-1] for data in [nival_A,nival_B,thermal_A,thermal_B]]).T
        gt_ds.to_csv(save_path+path+'\setB_{}.csv'.format(validation.iloc[i]))



#SET B_LIT
magt_lit = sorted(glob.glob(r'\HMA_Permafrost\MAGT_litrev\*.tif'))
save_path = r'\HMA_Permafrost\MAGT_litrev'

#In order of set names alphabetized
val_points_lat = [hiwat_points.Latitude,icimod_points.Lat,icimod_snow_points.Y,ismn_points.lat, ma_points.Latitude,others_points.LAT, zhao2021_points.Latitude]
val_points_lon = [hiwat_points.Longtitude,icimod_points.Lon,icimod_snow_points.X,ismn_points.lon, ma_points.Longitude,others_points.LON, zhao2021_points.Longitude]
validation_ids = [hiwat_points.station, icimod_points.NAME, icimod_snow_points.Station, ismn_points.site, ma_points.Site, 
                pd.DataFrame(['{}_{}'.format(PI,code) for PI,code in zip(others_points.PI,others_points.Code)]).iloc[:,0] , zhao2021_points.Site ]


obu_array =xr.open_dataset(magt_lit[0]).band_data[0]
ran_array =xr.open_dataset(magt_lit[1]).band_data[0]

obu_queries = [[(find_nearest(obu_array.y,lat)[1],find_nearest(obu_array.x,lon)[1]) for lat,lon in zip(lats,lons)] for lats,lons in zip(val_points_lat,val_points_lon)]
ran_queries = [[(find_nearest(ran_array.y,lat)[1],find_nearest(ran_array.x,lon)[1]) for lat,lon in zip(lats,lons)] for lats,lons in zip(val_points_lat,val_points_lon)]
paths = sorted(['ICIMOD','ICIMOD','HiWAT','Zhao2021','Others','ISMN','Ma2020'])

all_data_per_source = []
for queries,raster in zip([obu_queries,ran_queries],[obu_array,ran_array]):

    site_data = []
    for points, validation,path in zip(queries,validation_ids,paths):

        model_data_per_set = [float(raster[point[0],point[1]]) for point in points]
        sites_per_set = [validation.iloc[i] for i in range(0,len(validation))]
        set_origin = [[path]*len(validation)][0]
        magt_ds = pd.DataFrame((sites_per_set,model_data_per_set,set_origin)).T.rename(columns={0:'site',1:'magt',2:'origin'}).sort_values(by='site')
        site_data.append(magt_ds)

    all_data_per_source.append(site_data)

all_data_obu= pd.concat(all_data_per_source[0],axis=0).reset_index()
all_data_ran = pd.concat(all_data_per_source[1],axis=0).reset_index()