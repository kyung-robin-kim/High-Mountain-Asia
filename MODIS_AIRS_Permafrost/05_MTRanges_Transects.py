import xarray as xr
import geopandas as gpd
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import glob
import rasterstats as rs

def read_file(file):
    with rio.open(file) as src:
        return(src.read(1))

degree_sign = u"\N{DEGREE SIGN}"

#########################################################
#CALL ALL DATA
#########################################################

################################
#PZIs
path=r'\pzis'
pzi_files = sorted(list(glob.glob(path+'\*.tif')))
pzi_arrays = [xr.open_dataarray(file)[0] for file in pzi_files]
pzi_arrays[-1] = pzi_arrays[-1]/100

################################
#Mean Annual Ground Temperatures
path=r'\transects'
mat_files = sorted(list(glob.glob(path+'\*.tif')))
path=r'\stdev'
mat_stdev_files = sorted(list(glob.glob(path+'\*.tif')))

mat_arrays = [xr.open_dataarray(file)[0] for file in mat_files]
mat_names = ['AIRS GF','MAST','MAGST','TTOP ERA5-L','TTOP GLDAS','TTOP (Obu)','DZAAGT (Ran)']


################################
#Elevation
path = r'\01_elevation_1km.tif'
elev_1km = xr.open_mfdataset(path,parallel=True,chunks={'x':100,'y':100})

path = r'\03_elevation_stdev_1km.tif'
elev_std_1km = xr.open_mfdataset(path,parallel=True,chunks={'x':100,'y':100})

################################
#Mean Annual Snow Cover & Snow Depth
#masd_daily = r'E:\processed-data\SnowDepth\MASD_daily_2003_2016.tif'
masd = r'\annual_mean_snow_depths.nc'
masd_mean = xr.open_mfdataset(masd,parallel=True,chunks={'x':100,'y':100}).mean(dim='year')
masd_stdev = xr.open_mfdataset(masd,parallel=True,chunks={'x':100,'y':100}).std(dim='year')

#masc_daily = r'E:\processed-data\SnowCover\MODIS\ANNUAL\BOTH_CLIPPED_AVG_MSC_03_16.tif'
masc_mean = xr.open_mfdataset(r'\MASC_monthly_03_16.tif')
masc_stdev  = xr.open_mfdataset(r'\mean_snowcover_stdev_03_16.tif')


################################
#Transects
shpnames = sorted(glob.glob(r'\transects\*.shp'))
transect_points = [gpd.read_file(shpname).to_crs({'init': 'epsg:4326'}) for shpname in shpnames]
transect_points_xy = [gpd.GeoDataFrame([gdf.geometry.x,gdf.geometry.y,gdf.geometry]).T.rename(columns={'Unnamed 0':'X','Unnamed 1':'Y',2:'geometry'}) for gdf in transect_points]

#POINT ANALYSIS
#https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

pzi_queries = [[[(find_nearest(raster.y,transect_points.Y.iloc[i])[1],find_nearest(raster.x,transect_points.X.iloc[i])[1]) 
             for i in range(0,len(transect_points))] for transect_points in transect_points_xy] for raster in pzi_arrays]

mat_queries = [[[(find_nearest(raster.y,transect_points.Y.iloc[i])[1],find_nearest(raster.x,transect_points.X.iloc[i])[1]) 
             for i in range(0,len(transect_points))] for transect_points in transect_points_xy] for raster in mat_arrays]

elev_data = [elev_1km.band_data[0],elev_std_1km.band_data[0]]
elev_queries = [[[(find_nearest(raster.y,transect_points.Y.iloc[i])[1],find_nearest(raster.x,transect_points.X.iloc[i])[1]) 
             for i in range(0,len(transect_points))] for transect_points in transect_points_xy] for raster in elev_data]

snow_data = [masd_mean.__xarray_dataarray_variable__[0],masd_stdev.__xarray_dataarray_variable__[0],masc_mean.band_data[0],masc_stdev.band_data[0]]
snow_queries = [[[(find_nearest(raster.y,transect_points.Y.iloc[i])[1],find_nearest(raster.x,transect_points.X.iloc[i])[1]) 
             for i in range(0,len(transect_points))] for transect_points in transect_points_xy] for raster in snow_data]

pzi_transects = []
for raster,i in zip(pzi_arrays,range(0,len(pzi_arrays))):
    pzi_transects.append([[[float(raster[point[0],point[1]])] for point in transect] for transect in pzi_queries[i]])
magt_transects = []
for raster,i in zip(mat_arrays,range(0,len(mat_arrays))):
    magt_transects.append([[[float(raster[point[0],point[1]])] for point in transect] for transect in mat_queries[i]])
elev_transects = []
for raster,i in zip(elev_data,range(0,len(elev_data))):
    elev_transects.append([[[float(raster[point[0],point[1]])] for point in transect] for transect in elev_queries[i]])
snow_transects = []
for raster,i in zip(snow_data,range(0,len(snow_data))):
    snow_transects.append([[[float(raster[point[0],point[1]])] for point in transect] for transect in snow_queries[i]])

#Glaciers & Lakes
glaciers = xr.open_dataarray(r'\glacier_mask_1km_nearest.tif')[0]
lakes = xr.open_dataarray(r'\lake_mask_1km_nearest.tif')[0]
masks = [glaciers,lakes]

mask_queries = [[[(find_nearest(raster.y,transect_points.Y.iloc[i])[1],find_nearest(raster.x,transect_points.X.iloc[i])[1]) 
             for i in range(0,len(transect_points))] for transect_points in transect_points_xy] for raster in masks]
mask_transects = []
for raster,i in zip(masks,range(0,len(masks))):
    mask_transects.append([[[float(raster[point[0],point[1]])] for point in transect] for transect in mask_queries[i]])


mat_names = ['AIRS GF','MAST','MAGST','TTOP ERA5-L','TTOP GLDAS','TTOP (Obu)','DZAAGT (Ran)']
transect_names = ['Nyainqentanglha','Qilian','Southern Himal','Tien Shan', 'Western Himal']

pzis = [[np.array(transect).flatten() for transect in transects] for transects in pzi_transects]
magts = [[np.array(transect).flatten() for transect in transects] for transects in magt_transects]
elevs = [[np.array(transect).flatten() for transect in transects] for transects in elev_transects]
snows = [[np.array(transect).flatten() for transect in transects] for transects in snow_transects]
masks = [[np.array(transect).flatten() for transect in transects] for transects in mask_transects]

latitudes = [transect.Y for transect in transect_points_xy]
longitudes = [np.array(transect.X) for transect in transect_points_xy]

for i in range(0,5):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 20})
    #fig, ((ax1, ax2, ax3)) = plt.subplots(3,1,figsize=(15, 12))
    fig, ((ax1, ax3)) = plt.subplots(2,1,figsize=(15, 8))
    ax1.grid(color='black', linestyle='-', linewidth=0.05)
    y_line = 0
    ax1.axhline(y_line,color='black')
    ax1_5 = ax1.twinx()
    #ax2_5 = ax2.twinx()
    ax3_5 = ax3.twinx()

    transect_magt = [magts[ii][i] for ii in range(0,7)]
    magt_colors = ['darkgray','#56B4E9','#009E73','#D55E00','#CC79A7','#E69F00','#0072B2'] 

    transect_pzi = [pzis[ii][i] for ii in range(0,7)]
    transect_snow = [snows[ii][i] for ii in range(0,4)]
    transect_elev = [elevs[ii][i] for ii in range(0,2)]
    transect_mask = [masks[ii][i] for ii in range(0,2)]

    [ax1.plot(longitudes[i],temp,color='{}'.format(code),label='{}'.format(source)) for temp,code,source in zip(transect_magt,magt_colors,mat_names)]
    ax1_5.plot(longitudes[i],transect_elev[0]/1000,'--',linewidth=1,color='black',label='MERIT DEM')
    ax1_5.scatter(longitudes[i][transect_mask[0]==1],(transect_elev[0]/1000)[transect_mask[0]==1],color='blue',label='Glaciers')
    ax3.plot(longitudes[i],transect_snow[0],color='#51009C',label='HMA Snow Reanalysis')
    ax3_5.plot(longitudes[i],transect_snow[2],'--',color='#8E79C3',label='MODIS fSCA')
    ax1.set(xticklabels=[])


    ax1.set_ylabel('Temperature ({}C)'.format(degree_sign), fontsize = 17)
    ax1.set_ylim(-10,12)
    ax1_5.set_ylabel('Elevation (km)', fontsize = 17)
    ax1_5.set_ylim(2,6.5)
    ax3.set_ylim(0,2.5)
    ax3.set_ylabel('Snow Depth (m)', fontsize = 17)
    ax3_5.set_ylim(0,100)
    ax3_5.set_ylabel('Snow Cover Fraction (%)', fontsize = 17)
    ax1.set_xlim(min(longitudes[i]),max(longitudes[i]))
    ax3.set_xlim(min(longitudes[i]),max(longitudes[i]))
    ax3.set_xlabel('Longitude ({})'.format(degree_sign))

    ax1.set_title(transect_names[i])
    plt.savefig(r'\line_plots\{}.png'.format(transect_names[i]))
