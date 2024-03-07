import rasterio as rio
import glob
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, mapping
from rasterio.enums import Resampling
import pandas as pd
from scipy.stats import mode

def read_file(file):
    with rio.open(file) as src:
        return(src.read(1))

degree_sign = u"\N{DEGREE SIGN}"

#https://www.geeksforgeeks.org/how-to-print-superscript-and-subscript-in-python/
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(20,16))
    image = ax1.imshow(array,cmap = 'RdBu_r',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))

def unique_vals(array,roundoff):
    unique_values = np.unique(np.round(array,roundoff))

    return unique_values

shpname = r'/MTRanges.shp'
shapefile = gpd.read_file(shpname)

dem_files = sorted(list(glob.glob(r'\MERIT_DEM\*.tif')))
dem_1km = xr.open_rasterio(dem_files[0])


#ELEVATION vs. ZONATION
###########################
files = r'\HMA_Permafrost\PZI\*.tif'
pzi_files = sorted(list(glob.glob(files)))
shapefile = gpd.read_file(shpname).to_crs('EPSG:6933')
pzis = [xr.open_rasterio(pzi)[0].rio.reproject('EPSG:6933') for pzi in pzi_files]
pzis = [pzi.rio.reproject_match(pzis[0], resampling = Resampling.nearest).rio.reproject('EPSG:6933') for pzi in pzis]
pzis[5] = pzis[5]/100
lakes = xr.open_mfdataset(r'\lake_mask_1km_nearest.tif').band_data[0].rio.reproject_match(pzis[0], resampling = Resampling.nearest).rio.reproject('EPSG:6933')
glaciers = xr.open_mfdataset(r'\glacier_mask_1km_nearest.tif').band_data[0].rio.reproject_match(pzis[0], resampling = Resampling.nearest).rio.reproject('EPSG:6933')
pzis_masked = [pzi.where((lakes!=1) & (glaciers!=1),np.nan) for pzi in pzis]
#titles = ['MAST', 'MAGST','TTOP (GLDAS)', 'TTOP (ERA5-L)', 'TTOP (Obu)', 'DZAAGT (Ran)','MAATGT (Gruber)']
titles = ['MAGT-Ia', 'MAGT-II','MAGT-IIIa', 'MAGT-IIIb', 'MAGT-IIIc (Obu)', 'MAGT-IV (Ran)','MAGT-Ib (Gruber)']

shapefile = gpd.read_file(shpname)
mtn_regions = [shapefile.Region[i] for i in np.array([0,14,6,5,3,9,8,2,10,4,13,11,1,12,7])]
regional_shapefiles = [gpd.GeoDataFrame(index=[i],crs='epsg:4326',geometry=[shapefile.loc[ii].geometry]) 
                       for i,ii in zip(range(0,15),np.array([0,14,6,5,3,9,8,2,10,4,13,11,1,12,7]))]

#All histogram
plt.rcParams["font.family"] = "Times New Roman"
for pzi_raster, title in zip(pzis_masked,titles):
    dem_file_interp = np.array(dem_1km.rio.reproject_match(pzi_raster, resampling = Resampling.nearest))
    dem_masked_pzi = np.where(pzi_raster >= 0.01, dem_file_interp, np.nan)
    non_nan_mask = ~np.isnan(dem_masked_pzi)
    dem_masked_pzi = dem_masked_pzi[non_nan_mask]
    x = sorted(dem_masked_pzi.ravel())
    y = np.arange(np.size(x))/float(np.size(x))

    continuous = np.where(pzi_raster >= 0.9, dem_file_interp, np.nan)
    discontinuous = np.where( (pzi_raster >= 0.5) & (pzi_raster < 0.9), dem_file_interp, np.nan )
    sporadic = np.where((pzi_raster >= 0.1) & (pzi_raster < 0.5), dem_file_interp, np.nan)
    isolated = np.where((pzi_raster >= 0.001) & (pzi_raster < 0.1), dem_file_interp, np.nan)

    import seaborn as sns
    palette = 'BuPu'
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(16,11))
    sns.distplot(continuous, bins=range(int(np.floor(np.nanmin(continuous))),int(np.ceil(np.nanmax(continuous))), 1), ax=ax, kde=False,color=sns.color_palette('{}'.format(palette))[5])
    sns.distplot(discontinuous, bins=range(int(np.floor(np.nanmin(discontinuous))),int(np.ceil(np.nanmax(discontinuous))), 1), ax=ax, kde=False,color=sns.color_palette('{}'.format(palette))[4])
    sns.distplot(sporadic, bins=range(int(np.floor(np.nanmin(sporadic))),int(np.ceil(np.nanmax(sporadic))), 1), ax=ax, kde=False,color=sns.color_palette('{}'.format(palette))[3])
    sns.distplot(isolated, bins=range(int(np.floor(np.nanmin(isolated))),int(np.ceil(np.nanmax(isolated))), 1), ax=ax, kde=False,color=sns.color_palette('{}'.format(palette))[2])
    ax.set_title('{} PZI Histogram'.format(title),weight='bold')
    ax2 = ax.twinx()
    ax2.plot(x[0::10],y[0::10],color='black')
    ax2.set_ylabel('Cumulative Frequency', weight='bold')
    ax.set_xlim([0,8250])
    ax.set_ylim(0,1900)
    ax2.set_ylim([0,1])
    ax.set_xlabel('Elevation (m)', weight='bold',fontsize=22)
    ax.set_ylabel('Region (km{})'.format(get_super('2')),weight='bold',fontsize=22)
    fig.legend(labels=['Continuous','Discontinuous','Sporadic','Isolated'], loc='upper left',borderaxespad=5)
    plt.savefig(r'\CDFs\elev_hist_{}.png'.format(title),bbox_inches='tight',dpi = 400)
    

#Mtn Region Histogram
for pzi_raster, title in zip(pzis_masked[1:],titles[1:]):

    dem_1km_clipped = [dem_1km.rio.clip(region.geometry.apply(mapping), region.crs, drop=True,all_touched=True)[0] for region in regional_shapefiles]
    dem_1km_clipped = [dem.where(dem<8000,np.nan) for dem in dem_1km_clipped]
    pzi_clipped = [pzi_raster.rio.clip(region.geometry.apply(mapping), region.crs, drop=True,all_touched=True) for region in regional_shapefiles]

    for i in range(0,len(mtn_regions)):
        dem_file_interp = np.array(dem_1km_clipped[i].rio.reproject_match(pzi_clipped[i], resampling = Resampling.nearest))

        dem_masked_pzi = np.where(pzi_clipped[i] >= 0.01, dem_file_interp, np.nan)
        non_nan_mask = ~np.isnan(dem_masked_pzi)
        dem_masked_pzi = dem_masked_pzi[non_nan_mask]
        x = sorted(dem_masked_pzi.ravel())
        y = np.arange(np.size(x))/float(np.size(x))

        continuous = np.where(pzi_clipped[i] >= 0.9, dem_file_interp, np.nan)
        discontinuous = np.where( (pzi_clipped[i] >= 0.5) & (pzi_clipped[i] < 0.9), dem_file_interp, np.nan )
        sporadic = np.where((pzi_clipped[i] >= 0.1) & (pzi_clipped[i] < 0.5), dem_file_interp, np.nan)
        isolated = np.where((pzi_clipped[i] >= 0.05) & (pzi_clipped[i] < 0.1), dem_file_interp, np.nan)


        import seaborn as sns
        plt.rcParams["font.family"] = "Times New Roman"
        palette = 'BuPu'
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(figsize=(16,11))
        try:
            sns.distplot(continuous, bins=range(int(np.floor(np.nanmin(continuous))),int(np.ceil(np.nanmax(continuous))), 1), ax=ax, kde=False,color=sns.color_palette('{}'.format(palette))[5])
            sns.distplot(discontinuous, bins=range(int(np.floor(np.nanmin(discontinuous))),int(np.ceil(np.nanmax(discontinuous))), 1), ax=ax, kde=False,color=sns.color_palette('{}'.format(palette))[4])
            sns.distplot(sporadic, bins=range(int(np.floor(np.nanmin(sporadic))),int(np.ceil(np.nanmax(sporadic))), 1), ax=ax, kde=False,color=sns.color_palette('{}'.format(palette))[3])
            sns.distplot(isolated, bins=range(int(np.floor(np.nanmin(isolated))),int(np.ceil(np.nanmax(isolated))), 1), ax=ax, kde=False,color=sns.color_palette('{}'.format(palette))[2])
            ax.set_title('{} PZI Histogram'.format(title),weight='bold')
            ax2 = ax.twinx()
            ax2.plot(x[0::10],y[0::10],color='black')
            ax2.set_ylabel('Cumulative Frequency', weight='bold')
            ax.set_xlim([0,8250])
            ax.set_ylim(0,1900)
            ax2.set_ylim([0,1])
            ax.set_xlabel('Elevation (m)', weight='bold',fontsize=22)
            ax.set_ylabel('Region (km{})'.format(get_super('2')),weight='bold',fontsize=22)
            fig.legend(labels=['Continuous','Discontinuous','Sporadic','Isolated'], loc='upper left',borderaxespad=5)
            plt.savefig(r'\elev_hist_{}_{}.png'.format(title,mtn_regions[i]),bbox_inches='tight',dpi = 400)

        except ValueError:
            print('error on {}'.format(mtn_regions[i]))



#Table of elevation stats per region
def stats(array,mtn_df):
    mean = np.nanmean(array)
    stdev = np.nanstd(array, ddof=1)
    median = np.nanmedian(array)

    stats = pd.DataFrame([mean,stdev,median,mtn_df]).T.rename(columns={0:'mean',1:'stdev',2:'median',3:'region'})
    return stats

pzi_rasters = [xr.open_rasterio(pzi_file)[0] for pzi_file in pzi_files[0:7]]
shapefile = gpd.read_file(shpname)
mtn_regions = [shapefile.Region[i] for i in np.array([0,14,6,5,3,9,8,2,10,4,13,11,1,12,7])]
regional_shapefiles = [gpd.GeoDataFrame(index=[i],crs='epsg:4326',geometry=[shapefile.loc[ii].geometry]) 
                       for i,ii in zip(range(0,15),np.array([0,14,6,5,3,9,8,2,10,4,13,11,1,12,7]))]
pzi_elev_stats = []
for pzi_raster, title in zip(pzis_masked,titles):

    dem_1km_clipped = [dem_1km.rio.clip(region.geometry.apply(mapping), region.crs, drop=True,all_touched=True)[0] for region in regional_shapefiles]
    dem_1km_clipped = [dem.where(dem<8000,np.nan) for dem in dem_1km_clipped]
    pzi_clipped = [pzi_raster.rio.clip(region.geometry.apply(mapping), region.crs, drop=True,all_touched=True) for region in regional_shapefiles]

    elev_stats_all=[]
    for i in range(0,len(mtn_regions)):
        dem_file_interp = np.array(dem_1km_clipped[i].rio.reproject_match(pzi_clipped[i], resampling = Resampling.nearest))
        dem_masked_pzi = np.where(pzi_clipped[i] >= 0.01, dem_file_interp, np.nan)
        non_nan_mask = ~np.isnan(dem_masked_pzi)
        dem_masked_pzi = dem_masked_pzi[non_nan_mask]
        x = sorted(dem_masked_pzi.ravel())
        y = np.arange(np.size(x))/float(np.size(x))

        continuous = stats(np.where(pzi_clipped[i] >= 0.9, dem_file_interp, np.nan),mtn_regions[i])
        discontinuous = stats(np.where( (pzi_clipped[i] >= 0.5) & (pzi_clipped[i] < 0.9), dem_file_interp, np.nan ),mtn_regions[i])
        sporadic = stats(np.where((pzi_clipped[i] >= 0.1) & (pzi_clipped[i] < 0.5), dem_file_interp, np.nan),mtn_regions[i])
        isolated = stats(np.where((pzi_clipped[i] >= 0.05) & (pzi_clipped[i] < 0.1), dem_file_interp, np.nan),mtn_regions[i])
        all_pzi = stats(np.where((pzi_clipped[i] >= 0.05) & (pzi_clipped[i] <= 1), dem_file_interp, np.nan),mtn_regions[i])

        elev_stats = pd.concat([continuous,discontinuous,sporadic,isolated,all_pzi],axis=0).reset_index().rename({0:'continuous',1:'disc',2:'sporadic',3:'isolated',4:'all'}).iloc[:,1:]
        elev_stats_all.append(elev_stats)

    elev_stats_all = pd.concat(elev_stats_all,axis=0)
    pzi_elev_stats.append(elev_stats_all)

#mean for all
types_elevs_all = []
for pzi in ['continuous','disc','sporadic','isolated','all']:
    type_means = pd.concat([pzi_elev_stats[i].loc[pzi_elev_stats[i].index=='{}'.format(pzi)]['mean'] for i in range(0,len(pzi_elev_stats))],axis=1).reset_index().iloc[:,1:]
    type_means.columns = titles
    types_elevs_all.append(type_means)


dem_1km_clipped = [dem_1km.rio.clip(region.geometry.apply(mapping), region.crs, drop=True,all_touched=True)[0] for region in regional_shapefiles]
dem_1km_clipped = [dem.where((dem<9000) & (dem>0),np.nan) for dem in dem_1km_clipped]


mean_elev = np.array([elev.mean(dim={'x','y'})/1000 for elev in dem_1km_clipped])
mean_med = np.array([elev.median(dim={'x','y'})/1000 for elev in dem_1km_clipped])
mean_max = np.array([elev.max(dim={'x','y'})/1000 for elev in dem_1km_clipped])
mean_min = np.array([elev.min(dim={'x','y'})/1000 for elev in dem_1km_clipped])
mean_10 = np.array([elev.quantile(0.01)/1000 for elev in dem_1km_clipped])
mean_90 = np.array([elev.quantile(0.99)/1000 for elev in dem_1km_clipped])

snow_depth_mean = [xr.open_rasterio(snow)[0].rio.reproject('EPSG:4326') for snow in sorted(glob.glob(r'\*mean.tif'))]
snow_depth_stdev = [xr.open_rasterio(snow)[0].rio.reproject('EPSG:4326') for snow in sorted(glob.glob(r'\*stdev.tif'))]
snow_mean_depth = np.array([snow.mean(dim={'x','y'}) for snow in  snow_depth_mean])
snow_stdev_depth = np.array([snow.quantile(0.50) for snow in snow_depth_stdev])


snow_depth_monthly = [xr.open_rasterio(snow)[0].rio.reproject('EPSG:4326') for snow in sorted(glob.glob(r'\*.tif'))]
snow_depth_monthly_clipped = [[month.rio.clip(region.geometry.apply(mapping), region.crs, drop=True,all_touched=True) for region in regional_shapefiles] for month in snow_depth_monthly]

snow_depth_monthlies = [[float(region.mean(dim={'y','x'})) for region in month] for month in snow_depth_monthly_clipped]
snow_df = pd.DataFrame(snow_depth_monthlies)

snow_df_means = snow_df.mean(axis=0)
snow_df_max = snow_df.max(axis=0)
snow_df_min = snow_df.min(axis=0)

PE_labels = ['Continuous','Discontinuous','Sporadic','Isolated']
PE_colors = ['purple','violet','blue','skyblue']





for zone in range(0,4):
    fig,ax = plt.subplots(figsize=(20,12))
    ax.set_ylim(0,8.5)
    ax.set_xlim(0,14)
    ax.set_ylabel('Elevation (km)',weight='bold')
    ax.fill_between(range(0,15),mean_min,mean_max,color='gray',alpha=0.2)
    for mtn in range(0,15):
        ax.plot([mtn,mtn],[types_elevs_all[zone].iloc[mtn,:].max()/1000,types_elevs_all[zone].iloc[mtn,:].min()/1000],color=PE_colors[zone],linewidth=10)
    ax2 = ax.twinx() 
    ax.set_xticks(range(0,15))
    ax.set_xticklabels(mtn_regions,fontsize=20,rotation=34)
    ax.set_xlim(0,14)

    ax2.set_ylabel('Monthly Snow Depth (m)',weight='bold')
    ax2.set_ylim(0,2)
    
    ax.set_xlabel('Mountain Range',weight='bold')
    ax.set_title('Mean Zonation Height',weight='bold')
    plt.savefig(r'\line_plots\{}_elev.pdf'.format(PE_labels[zone]),bbox_inches='tight')


for zone in range(0,4):
    fig,ax = plt.subplots(figsize=(20,12))
    ax.set_ylim(0,8.7)
    ax.set_ylabel('Elevation (km)',weight='bold')

    for mtn in range(0,15):
        ax.scatter(mtn,types_elevs_all[zone].iloc[mtn,:].max()/1000,color=PE_colors[zone])
        ax.scatter(mtn,types_elevs_all[zone].iloc[mtn,:].min()/1000,color=PE_colors[zone])
        ax.plot([mtn,mtn],[types_elevs_all[zone].iloc[mtn,:].max()/1000,types_elevs_all[zone].iloc[mtn,:].min()/1000],color=PE_colors[zone],linewidth=10)

    ax2 = ax.twinx() 
    ax.set_xticks(range(0,15))
    ax.set_xticklabels(mtn_regions,fontsize=20,rotation=34)

    ax2.set_ylabel('Monthly Snow Depth (m)',weight='bold')
    ax2.set_ylim(0,2)

    
    ax.set_xlabel('Mountain Range',weight='bold')
    ax.set_title('Mean Zonation Height',weight='bold')
    plt.savefig(r'\line_plots\{}.pdf'.format(PE_labels[zone]),bbox_inches='tight')




for type_means,shade,type_pzi in zip(types_elevs_all,['purple','violet','blue','skyblue'],['Continuous','Discontinuous','Sporadic','Isolated']):
    plt.rcParams.update({'font.size': 27})
    ax2 = ax.twinx()
    ax.plot(type_means/1000,color=shade,label=type_pzi)
    ax.set_ylim(0,8.5)
    ax.set_xlim(0,14)
    ax.set_ylabel('Elevation (km)',weight='bold')

    ax2.set_ylabel('Monthly Snow Depth (m)',weight='bold')
    ax2.set_ylim(0,2)
    ax2.plot(snow_df_means,'o', markersize = 8,color='black',label='MSD Mean')
    ax2.scatter(range(0,15),snow_df_max,marker='v',s = 60, color='blue',label='MSD Max')
    ax2.scatter(range(0,15),snow_df_min,marker='^',s= 60,color='red',label='MSD Min')

    ax.set_xlabel('Mountain Range',weight='bold')
    ax.set_title('Mean Zonation Height',weight='bold')
    ax.set_xticks(range(0,15))
    ax.set_xticklabels(mtn_regions,fontsize=20,rotation=34)

#ax.legend()
ax2.legend()
plt.savefig(r'\line_plots\PZI-elev.png',bbox_inches='tight',dpi=400)
