import rasterio as rio
import glob
from scipy import special
import matplotlib.pyplot as plt 
import numpy as np
import xarray as xr
import geopandas as gpd
from scipy import special
from shapely.geometry import box, mapping
from rasterio.enums import Resampling
import pandas as pd
from rasterstats import zonal_stats


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
    image = ax1.imshow(array,cmap = 'Blues',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))

def unique_vals(array,roundoff):
    unique_values = np.unique(np.round(array,roundoff))

    return unique_values

#########################################################
#CALL ALL DATA
#########################################################
################################
#Mean Annual Ground Temperatures
mean_paths = [r'\2003_2016_MODMYD11AIRS.tif',
                r'\2003_2016_MAGT_nival.tif',
                r'\ERA5_L\MAGT\mean\2003_2016_MAGT_thermaloffset.tif',
                r'\GLDAS\MAGT\mean\2003_2016_MAGT_thermaloffset.tif']

stdev_paths = [r'\MALST\mean\clim_stdev.tif',
                r'\SmRise_NivalOnly\MAGT\mean\clim_stdev.tif',
                r'\ERA5_L\MAGT\mean\clim_stdev.tif',
                r'\GLDAS\MAGT\mean\clim_stdev.tif']

MATs = [xr.open_dataarray(file)[0] for file in mean_paths]
MAT_CLIM_STDEVs = [xr.open_dataarray(file)[0] for file in stdev_paths]
MAT_NAMES = ['MALST','MAGST','TTOP_ERA5-L','TTOP-GLDAS']

#Topographic Variables
dem_files = sorted(list(glob.glob(r'\MERIT_DEM\*.tif')))
dem = xr.open_rasterio(dem_files[0])[0]
dem_std_1km = xr.open_rasterio(dem_files[2])[0]


##################################################################################################################
#PRE-PROCESSING (MASK)
##################################################################################################################
#Mask out glaciers and lakes from MAGTs
lakes = xr.open_mfdataset(r'\lake_mask_1km_nearest.tif').band_data[0]
glaciers = xr.open_mfdataset(r'\glacier_mask_1km_nearest.tif').band_data[0]

#########################################################################################################################
#Permafrost Probabilities
#########################################################################################################################

regression_metrics = [pd.read_csv(file) for file in sorted(glob.glob(r'\MULTI_ELEV_LAT\*.csv'))]
am_lapserates = pd.DataFrame(regression_metrics[0].B1_lapserate).set_index(regression_metrics[0].date).dropna()
pm_lapserates = pd.DataFrame(regression_metrics[1].B1_lapserate).set_index(regression_metrics[1].date).dropna()
avg_lapserate = pd.concat([am_lapserates,pm_lapserates],axis=1).mean(axis=1).mean()
stdev_lapserate = pd.concat([am_lapserates,pm_lapserates],axis=1).mean(axis=1).std(ddof=1)


###############
#AIRS LAPSE RATE DOCUMENTATION

am_R2 = pd.DataFrame(regression_metrics[0].R2	).set_index(regression_metrics[0].date).dropna()
pm_R2 = pd.DataFrame(regression_metrics[1].R2	).set_index(regression_metrics[1].date).dropna()
am_B2 = pd.DataFrame(regression_metrics[0].B2	).set_index(regression_metrics[0].date).dropna()
pm_B2 = pd.DataFrame(regression_metrics[1].B2	).set_index(regression_metrics[1].date).dropna()
am_int = pd.DataFrame(regression_metrics[0].intercept	).set_index(regression_metrics[0].date).dropna()
pm_int = pd.DataFrame(regression_metrics[1].intercept	).set_index(regression_metrics[1].date).dropna()

ts_lapse = pd.concat([am_lapserates,pm_lapserates],axis=1).mean(axis=1)


night_metrics = [pm_R2,pm_int,pm_lapserates,pm_B2]
day_metrics = [am_R2,am_int,am_lapserates,am_B2]
labels = ['R2','Base Temperature ({}C)'.format(degree_sign),'Lapse Rate, B1 ({}C/m)'.format(degree_sign),'Latitudinal Coefficient, ({}C/{} Lat)'.format(degree_sign,degree_sign)]
plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "Times New Roman"
for i in range(0,4):
    plt.figure(figsize=(10,5))
    plt.boxplot(pd.concat([night_metrics[i],day_metrics[i]],axis=1).dropna(),showmeans=True)
    plt.xticks([1,2],['AM (Night)','PM (Day)'])
    plt.ylabel(labels[i],fontsize=18)
    plt.tight_layout()


#Total Error
MALST_error = 0.603729
MAGST_error = 1.920996
TTOP_ERA_error = 3.17552
TTOP_GLDAS_error = 2.446096

errors = [MALST_error,MAGST_error,TTOP_ERA_error,TTOP_GLDAS_error]

avg_lapserate
dem_std_1km

############

def calculate_pzi(average,clim_var,error_dist):
    variance = np.array(clim_var)**2 + (avg_lapserate*dem_std_1km)**2 + error_dist
    pzi = (1/2)*special.erfc(np.array(average)/(np.sqrt(2*variance)))
    pzi = np.where(pzi==0.5, np.nan, pzi)
    return pzi

pzis = [calculate_pzi(MAT,STDEV,error) for MAT,STDEV,error in zip(MATs,MAT_CLIM_STDEVs,errors)]

lat = np.array(MATs[0].y)
lon = np.array(MATs[0].x)
for pzi,name in zip(pzis,MAT_NAMES):
    new_array = xr.DataArray(pzi, dims=("y", "x"), coords={"y": lat, "x": lon}, name='PZI')
    new_array.rio.set_crs("epsg:4326")
    new_array.rio.set_spatial_dims('x','y',inplace=True)

    masked_array = new_array.where((lakes!=1) & (glaciers!=1),np.nan)
    masked_array.rio.to_raster(r'\PZI\{}_PE.tif'.format(name))


MAT_NAMES
#########################################################
#CDF of PZI against MALST
files = r'C:\Users\robin\Desktop\HMA_Permafrost\Data\Raster\PZI\*.tif'
pzi_lit_path = sorted(list(glob.glob(files)))

shpname = r'C:\Users\robin\Desktop\HMA_Permafrost\Data\Shapefiles/MTRanges.shp'
shapefile = gpd.read_file(shpname)

files = r'E:\processed-data\HMA_Permafrost\PZI\*.tif'
pzis_path = sorted(list(glob.glob(files)))


pzis_lit = [xr.open_rasterio(i).rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)[0] for i in pzi_lit_path]
pzis_lit[3] = pzis_lit[3].where(pzis_lit[3]>0, np.nan) #Obu
pzis_lit[4]= pzis_lit[4].where(pzis_lit[4]>0, np.nan)/100 #Ran
pzis_lit[5] = pzis_lit[5].where(pzis_lit[5]>0, np.nan) #Gruber
pzis_lit = [pzis_lit[3],pzis_lit[4],pzis_lit[5]]


mean_paths = [r'\2003_2016_MODMYD11AIRS.tif',
                r'\2003_2016_MAGT_nival.tif',
                r'\ERA5_L\MAGT\mean\2003_2016_MAGT_thermaloffset.tif',
                r'\GLDAS\MAGT\mean\2003_2016_MAGT_thermaloffset.tif',
                r'\05_MAGT_TTOP_OBU.tif',
                r'\06_MAGT_DZAA_RAN.tif']
MAGTs = [xr.open_dataarray(file)[0] for file in mean_paths]

pzi_clipped = [xr.open_rasterio(i).rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True)[0] for i in pzis_path]
mat_clipped = [i.rio.write_crs('epsg:4326').rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True).rio.reproject_match(pzi_clipped[0],resampling=Resampling.nearest) for i in MATs]
pzi_clipped[-3] = pzi_clipped[-3].where(pzi_clipped[-3]>0, np.nan) #Obu
pzi_clipped[-2] = pzi_clipped[-2].where(pzi_clipped[-2]>0, np.nan)/100 #Ran
pzi_clipped[-1] = pzi_clipped[-1].where(pzi_clipped[-1]>0, np.nan) #Gruber

#labels = ['MAST', 'MAGST','TTOP (GLDAS)', 'TTOP (ERA5-L)', 'TTOP (Obu)', 'DZAAGT (Ran)','MAATGT (Gruber)']
labels =  ['MAGT-Ia', 'MAGT-II','MAGT-IIIa', 'MAGT-IIIb', 'MAGT-IIIc (Obu)', 'MAGT-IV (Ran)','MAGT-Ib (Gruber)']

MAT_NAMES
from scipy.stats import gaussian_kde
for name,pzi,mat in zip(labels,pzi_clipped,mat_clipped):
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.family"] = "Times New Roman"
    fig,ax = plt.subplots(figsize=(12, 10))
    plt.grid(color='black', linestyle='-', linewidth=0.05)
    independent = np.ravel(mat)
    dependent = np.ravel(pzi)
    nanmask = ~np.isnan(dependent) & ~np.isnan(independent)
    dependent = dependent[nanmask]
    independent = independent[nanmask]

     #Calculate the point density
    xy = np.vstack([independent,dependent])
    z = gaussian_kde(xy)(xy)
    
    ax.scatter(independent,dependent,c=z,s=10,alpha=1)
    ax.set_xlim(np.nanmin(-30),np.nanmax(30))
    ax.set_ylim(0,1)
    ax.set_title("{} Permafrost CDF".format(name),weight='bold')
    ax.set_xlabel('Mean Annual Ground Temperature ({}C)'.format(degree_sign), weight='bold',fontsize=22)
    ax.set_ylabel('Permafrost Extent', weight='bold',fontsize=22)
    fig.tight_layout()
    plt.savefig(r'\{}_MALST_density.png'.format(name))


################################################################
#Zonation Analyses
################################################################
####
#NOTE on permafrost area calculations:
#Total pixel area sums up the total permafrost regions
#Total permafrost area sums up the fractional permafrost region ~ truer to total permafrost area, if accurate 

###########################
#ZONATION CALCULATION
def calculate_zones(pzi):
    all = pzi.where(pzi>1,1)#For all pixels NOT>1, change to 1.
    continuous = pzi.where( (pzi >= 0.9), np.nan) #For all pixels NOT >= 0.9, change to NaN
    discontinuous = pzi.where( (pzi >= 0.5) & (pzi < 0.9), np.nan ) #For all pixels NOT >= 0.5 and <0.9, change to NaN
    sporadic = pzi.where((pzi >= 0.1) & (pzi < 0.5), np.nan)
    isolated = pzi.where((pzi >= 0.05) & (pzi < 0.1), np.nan)
    return all, continuous, discontinuous, sporadic, isolated


files = r'\HMA_Permafrost\PZI\*.tif'
pzi_files = sorted(list(glob.glob(files)))
shpname = r'/MTRanges.shp'
shapefile = gpd.read_file(shpname).to_crs('EPSG:6933')
pzis = [xr.open_rasterio(pzi)[0].rio.reproject('EPSG:6933') for pzi in pzi_files]
pzis = [pzi.rio.reproject_match(pzis[0], resampling = Resampling.nearest).rio.reproject('EPSG:6933') for pzi in pzis]
pzis[5] = pzis[5]/100
lakes = xr.open_mfdataset(r'\lake_mask_1km_nearest.tif').band_data[0].rio.reproject_match(pzis[0], resampling = Resampling.nearest).rio.reproject('EPSG:6933')
glaciers = xr.open_mfdataset(r'\glacier_mask_1km_nearest.tif').band_data[0].rio.reproject_match(pzis[0], resampling = Resampling.nearest).rio.reproject('EPSG:6933')
pzis_masked = [pzi.where((lakes!=1) & (glaciers!=1),np.nan).rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True) for pzi in pzis]

#labels = ['MAST', 'MAGST','TTOP (GLDAS)', 'TTOP (ERA5-L)', 'TTOP (Obu)', 'DZAAGTsd (Ran)','MAATGT (Gruber)']
labels =  ['MAGT-Ia', 'MAGT-II','MAGT-IIIa', 'MAGT-IIIb', 'MAGT-IIIc', 'MAGT-IV','MAGT-Ib']
all_totals, cont_totals, disc_totals, spor_totals, isol_totals = map(list,zip(*[calculate_zones(pzi) for pzi in pzis_masked]))

#[pzi.rio.to_raster(r'E:\processed-data\HMA_Permafrost\PZI\affine_epsg6933\{}.tif'.format(name)) for pzi,name in zip(pzis,labels)]
pzis_affine = [rio.open(pzi).transform for pzi in sorted(glob.glob(r'\affine_epsg6933\*.tif'))]

total_area = (shapefile.area)/10**6 #in km^2
all_pixel_sum = [zonal_stats(shapefile,np.array(all),affine = pzi_a,stats='count sum',all_touched=True) for all,pzi_a in zip(all_totals,pzis_affine)]
cont_pixel_sum = [zonal_stats(shapefile,np.array(cont),affine = pzi_a,stats='count sum',all_touched=True) for cont,pzi_a in zip(cont_totals,pzis_affine)]
disc_pixel_sum = [zonal_stats(shapefile,np.array(disc),affine = pzi_a,stats='count sum',all_touched=True) for disc,pzi_a in zip(disc_totals,pzis_affine)]
spor_pixel_sum = [zonal_stats(shapefile,np.array(spor),affine = pzi_a,stats='count sum',all_touched=True) for spor,pzi_a in zip(spor_totals,pzis_affine)]
isol_pixel_sum = [zonal_stats(shapefile,np.array(isol),affine = pzi_a,stats='count sum',all_touched=True) for isol,pzi_a in zip(isol_totals,pzis_affine)]

#All PZIs should equal same number of pixels
all_pixel_cumsum = [all_pixel_sum[0][mtnrange]['sum'] for mtnrange in range(0,15)]
all_pixel_count = [all_pixel_sum[0][mtnrange]['count'] for mtnrange in range(0,15)]

#Calculate ratio of area (km^2) per pixel - to account for projection effects
area2pixel_ratio = total_area/all_pixel_cumsum

#Add total pixel count of all permafrost zonations
total_pzi_pixels = []
for mtnrange in range(0,15):
    pzi_pixel_sum = [np.float(c[mtnrange]['count'] or 0) + np.float(d[mtnrange]['count'] or 0) + np.float(s[mtnrange]['count'] or 0) + np.float(i[mtnrange]['count'] or 0) for c,d,s,i in
                        zip(cont_pixel_sum,disc_pixel_sum,spor_pixel_sum,isol_pixel_sum) ]
    total_pzi_pixels.append((pzi_pixel_sum))

total_pzi_pixels_df = pd.DataFrame(total_pzi_pixels)

#Multiply total permafrost pixel by ratio (area/pixel) to get total area
total_pzi_pixelarea_df = total_pzi_pixels_df.multiply(area2pixel_ratio,axis=0).set_index(shapefile.Region)

#Calculate fraction of total permafrost per mountain region by PZI
permafrost_fraction_pixelarea_df = total_pzi_pixelarea_df.divide(total_area,axis=0)

df = total_pzi_pixelarea_df
W_to_E = pd.DataFrame([df.iloc[i] for i in np.array([0,14,6,5,3,9,8,2,10,4,13,11,1,12,7])])
W_E_mtn_areas = [shapefile.Area[i] for i in np.array([0,14,6,5,3,9,8,2,10,4,13,11,1,12,7])]
pzi_percents = pd.DataFrame([W_to_E.iloc[:,i]/np.array(W_E_mtn_areas) for i in range(0,len(W_to_E.keys()))])

plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots()
pzi_percents.T.plot.bar(rot=25,figsize=(18.5, 10),ax=ax,fontsize=20,color=['#56B4E9','#E69F00','#D55E00','#CC79A7','#009E73','#755B90','#0072B2'])
plt.ylim(0,1)
plt.title("Permafrost Regions",fontsize=40, weight='bold')
plt.ylabel('Fractional Area',fontsize=30, weight='bold')
plt.xlabel('Mountain Range',fontsize=35, weight='bold')
ax.legend(labels,fontsize=18)
plt.tight_layout()
plt.savefig(r'\Total_Permafrost_Regional_Area.png',dpi=500)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 25})
plt.figure(figsize=(18.5, 10))
plt.boxplot(W_to_E.T,showmeans=True,medianprops={'color': 'green'})
plt.xticks(range(1,16),[str(range) for range in W_to_E.index],rotation=25)
plt.ylabel('Permafrost Region (km{})'.format(get_super('2')),weight='bold')
plt.tight_layout()
plt.savefig(r'\Total_Permafrost_Regional_Area_histplot.png',dpi=500)

labels
W_to_E.T.mean()
W_to_E.T.std(ddof=1)

sorted(W_to_E.T.std(ddof=1)/W_to_E.T.mean())
#Add pixel count for each zonation
#0:Surface MAGT
#1,2: TTOP MAGT A & B
#3: TTOP MAGT - Obu
#4: DZAA MAGT - Ran
#5: MAAT MAGT - Gruber

for i,name in zip(range(0,7),labels):

    cont_pixel_cumcount = [cont_pixel_sum[i][mtnrange]['count'] for mtnrange in range(0,15)]
    disc_pixel_cumcount = [disc_pixel_sum[i][mtnrange]['count'] for mtnrange in range(0,15)]
    spor_pixel_cumcount = [spor_pixel_sum[i][mtnrange]['count'] for mtnrange in range(0,15)]
    isol_pixel_cumcount = [isol_pixel_sum[i][mtnrange]['count'] for mtnrange in range(0,15)]

    total_cont_pixelarea_df = (pd.DataFrame(cont_pixel_cumcount).multiply(area2pixel_ratio,axis=0).rename(columns={0:'Cont'}))
    total_disc_pixelarea_df = (pd.DataFrame(disc_pixel_cumcount).multiply(area2pixel_ratio,axis=0).rename(columns={0:'Disc'}))
    total_spor_pixelarea_df = (pd.DataFrame(spor_pixel_cumcount).multiply(area2pixel_ratio,axis=0).rename(columns={0:'Spor'}))
    total_isol_pixelarea_df = (pd.DataFrame(isol_pixel_cumcount).multiply(area2pixel_ratio,axis=0).rename(columns={0:'Isol'}))

    total_pixelarea_zonation_df = pd.concat([total_cont_pixelarea_df,total_disc_pixelarea_df,total_spor_pixelarea_df,total_isol_pixelarea_df],axis=1).set_index(shapefile.Region)

    cont_pixel_cumsum = [cont_pixel_sum[i][mtnrange]['count'] for mtnrange in range(0,15)]
    disc_pixel_cumsum = [disc_pixel_sum[i][mtnrange]['count'] for mtnrange in range(0,15)]
    spor_pixel_cumsum = [spor_pixel_sum[i][mtnrange]['count'] for mtnrange in range(0,15)]
    isol_pixel_cumsum = [isol_pixel_sum[i][mtnrange]['count'] for mtnrange in range(0,15)]

    total_cont_permafarea_df = (pd.DataFrame(cont_pixel_cumsum).multiply(area2pixel_ratio,axis=0).rename(columns={0:'Cont'}))
    total_disc_permafarea_df = (pd.DataFrame(disc_pixel_cumsum).multiply(area2pixel_ratio,axis=0).rename(columns={0:'Disc'}))
    total_spor_permafarea_df = (pd.DataFrame(spor_pixel_cumsum).multiply(area2pixel_ratio,axis=0).rename(columns={0:'Spor'}))
    total_isol_permafarea_df = (pd.DataFrame(isol_pixel_cumsum).multiply(area2pixel_ratio,axis=0).rename(columns={0:'Isol'}))

    total_permafarea_zonation_df = pd.concat([total_cont_permafarea_df,total_disc_permafarea_df,total_spor_permafarea_df,total_isol_permafarea_df],axis=1).set_index(shapefile.Region)/1000

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    plt.grid()
    total_permafarea_zonation_df.plot.bar(rot=45,figsize=(10, 6),ax=ax,colormap='PuBu_r',edgecolor='black')
    plt.title("{}".format(name),fontsize=22, weight='bold')
    plt.ylabel('Area (1000 km{})'.format(get_super('2')),fontsize=18, weight='bold')
    plt.xlabel('Mountain Range',fontsize=18, weight='bold')
    ax.legend(['Continuous','Discontinuous','Sporadic','Isolated'],fontsize=15)
    ax.set_ylim(0,350)
    plt.tight_layout()
    plt.savefig(r'\zonations\{}_{}_PixelArea.png'.format(i,name))

    #total_permafarea_zonation_df.sum(axis=0)
    #total_permafarea_zonation_df/total_pixelarea_zonation_df
    #For Ran, total_permafarea_zonation_df/100 to get 0~1




###########################################
#Pixel-by-Pixel Analysis (Confusion Matrix)
def permafrost_reclassify(pzi):
    all = pzi.where(pzi>1,50)
    yes = (((pzi>=0.05) & (pzi<=1))*100)
    no = (~((pzi>=0.05) & (pzi<=1))*50)
    if np.nanmax(pzi)>1:
        all = pzi.where(pzi>1,50)
        yes = (((pzi>=5) & (pzi<=100))*100)
        no = (~((pzi>=5) & (pzi<=100))*50)
    return all, yes, no

files = r'\HMA_Permafrost\PZI\*.tif'
pzi_files = sorted(list(glob.glob(files)))
shpname = r'/MTRanges.shp'
shapefile = gpd.read_file(shpname).to_crs('EPSG:6933')
pzis = [xr.open_rasterio(pzi)[0].rio.reproject('EPSG:6933') for pzi in pzi_files]
pzis = [pzi.rio.reproject_match(pzis[0], resampling = Resampling.nearest).rio.reproject('EPSG:6933') for pzi in pzis]
pzis_affine = [rio.open(pzi).transform for pzi in sorted(glob.glob(r'\affine_epsg6933\*.tif'))]

#Mask out glaciers and lakes from MAGTs
lakes = xr.open_mfdataset(r'\lake_mask_1km_nearest.tif').band_data[0].rio.reproject_match(pzis[0], resampling = Resampling.nearest).rio.reproject('EPSG:6933')
glaciers = xr.open_mfdataset(r'\glacier_mask_1km_nearest.tif').band_data[0].rio.reproject_match(pzis[0], resampling = Resampling.nearest).rio.reproject('EPSG:6933')
pzis_masked = [pzi.where((lakes!=1) & (glaciers!=1),np.nan).rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=True,all_touched=True) for pzi in pzis]
all_totals, yes_totals, no_totals = map(list,zip(*[permafrost_reclassify(pzi) for pzi in pzis_masked]))

#MAST Comparison
surface = [(yes_totals[0])+(no_totals[i]) for i in range(0,7)]
#MAGST
nival = [(yes_totals[1])+(no_totals[i]) for i in range(0,7)]
#TTOP GLDAS
gldas = [(yes_totals[2])+(no_totals[i]) for i in range(0,7)]
#TTOP ERA5
era5 = [(yes_totals[3])+(no_totals[i]) for i in range(0,7)]
#Obu 
obu = [(yes_totals[4])+(no_totals[i]) for i in range(0,7)]
#Ran
ran = [(yes_totals[5])+(no_totals[i]) for i in range(0,7)]
#Gruber
gruber = [(yes_totals[6])+(no_totals[i]) for i in range(0,7)]


#Comparison TIFS for Mapping
comparisons = [surface,nival,gldas,era5,obu,ran,gruber]
labels = ['1MAST', '2MAGST','3GLDAS', '4ERA5L', '5Obu', '6Ran','7Gruber']
change_maps = [[comparisons[i][ii] for i in range(0,7)] for ii in range(0,7)]

lat = pzis_masked[0].y
lon = pzis_masked[0].x
changemaps = [[xr.DataArray(raster, dims=("y", "x"), coords={"y": lat, "x": lon}).rio.set_spatial_dims('x','y',inplace=True).rio.set_crs("epsg:6933") for raster in change_map] for change_map in change_maps]
changemaps = [[raster.rio.set_spatial_dims('x','y',inplace=True).rio.set_crs("epsg:6933").rio.reproject('EPSG:4326') for raster in change_map] for change_map in change_maps]
[[raster.rio.to_raster(r'\HMA_Permafrost\PZI\change_maps\{}_{}.tif'.format(labels[i],labels[ii])) for raster,i in zip(changemap,range(0,7))] for changemap,ii in zip(changemaps,range(0,7))]


def permafrost_binary(pzi):
    all = pzi.where(pzi<=0,1)
    PP = pzi.where( (pzi==100), np.nan)/100
    NP = pzi.where( pzi==0, np.nan) + 1
    PN = pzi.where(pzi==150,np.nan)/150
    NN = pzi.where(pzi==50,np.nan)/50
    return PP,NP,PN,NN,all

PP_surface, NP_surface, PN_surface, NN_surface,all  = map(list,zip(*[permafrost_binary(pzi) for pzi in surface]))
PP_nival, NP_nival, PN_nival, NN_nival,all  = map(list,zip(*[permafrost_binary(pzi) for pzi in nival]))
PP_ttopA, NP_ttopA, PN_ttopA, NN_ttopA,all  = map(list,zip(*[permafrost_binary(pzi) for pzi in gldas]))
PP_ttopB, NP_ttopB, PN_ttopB, NN_ttopB,all  = map(list,zip(*[permafrost_binary(pzi) for pzi in era5]))
PP_obu, NP_obu, PN_obu, NN_obu,all  = map(list,zip(*[permafrost_binary(pzi) for pzi in obu]))
PP_ran, NP_ran, PN_ran, NN_ran,all  = map(list,zip(*[permafrost_binary(pzi) for pzi in ran]))
PP_gruber, NP_gruber, PN_gruber, NN_gruber,all  = map(list,zip(*[permafrost_binary(pzi) for pzi in gruber]))



all_sum = [zonal_stats(shapefile,np.array(raster),affine = pzi_affine,stats='count sum',all_touched=True) for raster,pzi_affine in zip(all,pzis_affine)]

surface_PP_sum = [zonal_stats(shapefile,np.array(PP),affine = pzi_affine,stats='count sum',all_touched=True) for PP,pzi_affine in  zip(PP_surface,pzis_affine)]
surface_NP_sum = [zonal_stats(shapefile,np.array(NP),affine = pzi_affine,stats='count sum',all_touched=True) for NP,pzi_affine in  zip(NP_surface,pzis_affine)]
surface_PN_sum = [zonal_stats(shapefile,np.array(PN),affine = pzi_affine,stats='count sum',all_touched=True) for PN,pzi_affine in  zip(PN_surface,pzis_affine)]
surface_NN_sum = [zonal_stats(shapefile,np.array(NN),affine = pzi_affine,stats='count sum',all_touched=True) for NN,pzi_affine in  zip(NN_surface,pzis_affine)]

nival_PP_sum = [zonal_stats(shapefile,np.array(PP),affine = pzi_affine,stats='count sum',all_touched=True) for PP,pzi_affine in  zip(PP_nival,pzis_affine)]
nival_NP_sum = [zonal_stats(shapefile,np.array(NP),affine = pzi_affine,stats='count sum',all_touched=True) for NP,pzi_affine in  zip(NP_nival,pzis_affine)]
nival_PN_sum = [zonal_stats(shapefile,np.array(PN),affine = pzi_affine,stats='count sum',all_touched=True) for PN,pzi_affine in  zip(PN_nival,pzis_affine)]
nival_NN_sum = [zonal_stats(shapefile,np.array(NN),affine = pzi_affine,stats='count sum',all_touched=True) for NN,pzi_affine in  zip(NN_nival,pzis_affine)]

ttopA_PP_sum = [zonal_stats(shapefile,np.array(PP),affine = pzi_affine,stats='count sum',all_touched=True) for PP,pzi_affine in  zip(PP_ttopA,pzis_affine)]
ttopA_NP_sum = [zonal_stats(shapefile,np.array(NP),affine = pzi_affine,stats='count sum',all_touched=True) for NP,pzi_affine in  zip(NP_ttopA,pzis_affine)]
ttopA_PN_sum = [zonal_stats(shapefile,np.array(PN),affine = pzi_affine,stats='count sum',all_touched=True) for PN,pzi_affine in  zip(PN_ttopA,pzis_affine)]
ttopA_NN_sum = [zonal_stats(shapefile,np.array(NN),affine = pzi_affine,stats='count sum',all_touched=True) for NN,pzi_affine in  zip(NN_ttopA,pzis_affine)]

ttopB_PP_sum = [zonal_stats(shapefile,np.array(PP),affine = pzi_affine,stats='count sum',all_touched=True) for PP,pzi_affine in  zip(PP_ttopB,pzis_affine)]
ttopB_NP_sum = [zonal_stats(shapefile,np.array(NP),affine = pzi_affine,stats='count sum',all_touched=True) for NP,pzi_affine in  zip(NP_ttopB,pzis_affine)]
ttopB_PN_sum = [zonal_stats(shapefile,np.array(PN),affine = pzi_affine,stats='count sum',all_touched=True) for PN,pzi_affine in  zip(PN_ttopB,pzis_affine)]
ttopB_NN_sum = [zonal_stats(shapefile,np.array(NN),affine = pzi_affine,stats='count sum',all_touched=True) for NN,pzi_affine in  zip(NN_ttopB,pzis_affine)]

obu_PP_sum = [zonal_stats(shapefile,np.array(PP),affine = pzi_affine,stats='count sum',all_touched=True) for PP,pzi_affine in  zip(PP_obu,pzis_affine)]
obu_NP_sum = [zonal_stats(shapefile,np.array(NP),affine = pzi_affine,stats='count sum',all_touched=True) for NP,pzi_affine in  zip(NP_obu,pzis_affine)]
obu_PN_sum = [zonal_stats(shapefile,np.array(PN),affine = pzi_affine,stats='count sum',all_touched=True) for PN,pzi_affine in  zip(PN_obu,pzis_affine)]
obu_NN_sum = [zonal_stats(shapefile,np.array(NN),affine = pzi_affine,stats='count sum',all_touched=True) for NN,pzi_affine in  zip(NN_obu,pzis_affine)]

ran_PP_sum = [zonal_stats(shapefile,np.array(PP),affine = pzi_affine,stats='count sum',all_touched=True) for PP,pzi_affine in  zip(PP_ran,pzis_affine)]
ran_NP_sum = [zonal_stats(shapefile,np.array(NP),affine = pzi_affine,stats='count sum',all_touched=True) for NP,pzi_affine in  zip(NP_ran,pzis_affine)]
ran_PN_sum = [zonal_stats(shapefile,np.array(PN),affine = pzi_affine,stats='count sum',all_touched=True) for PN,pzi_affine in  zip(PN_ran,pzis_affine)]
ran_NN_sum = [zonal_stats(shapefile,np.array(NN),affine = pzi_affine,stats='count sum',all_touched=True) for NN,pzi_affine in  zip(NN_ran,pzis_affine)]

gruber_PP_sum = [zonal_stats(shapefile,np.array(PP),affine = pzi_affine,stats='count sum',all_touched=True) for PP,pzi_affine in  zip(PP_gruber,pzis_affine)]
gruber_NP_sum = [zonal_stats(shapefile,np.array(NP),affine = pzi_affine,stats='count sum',all_touched=True) for NP,pzi_affine in  zip(NP_gruber,pzis_affine)]
gruber_PN_sum = [zonal_stats(shapefile,np.array(PN),affine = pzi_affine,stats='count sum',all_touched=True) for PN,pzi_affine in  zip(PN_gruber,pzis_affine)]
gruber_NN_sum = [zonal_stats(shapefile,np.array(NN),affine = pzi_affine,stats='count sum',all_touched=True) for NN,pzi_affine in  zip(NN_gruber,pzis_affine)]


#All PZIs should equal same number of pixels
all_pixel_cumsum = [all_sum[0][mtnrange]['sum'] for mtnrange in range(0,15)]

#Calculate ratio of area (km^2) per pixel - to account for projection effects
total_area = (shapefile.area)/10**6 #in km^2
area2pixel_ratio = total_area/all_pixel_cumsum

total_pzi_pixels=[]
#Add total pixel count of all
for mtnrange in range(0,15):
    pzi_pixel_sum = [(pp[mtnrange]['count'] or 0) + (np[mtnrange]['count'] or 0) + (pn[mtnrange]['count'] or 0) + (nn[mtnrange]['count'] or 0) for pp,np,pn,nn in zip(surface_PP_sum,surface_NP_sum,surface_PN_sum,surface_NN_sum) ]
    total_pzi_pixels.append((pzi_pixel_sum))
    print(mtnrange)

total_pzi_pixels_df = pd.DataFrame(total_pzi_pixels)


def calculate_area(list_pixelcounts):

    total_pixel_dfs = []
    for combo in list_pixelcounts:
        total_pixels = []
        for mtnrange in range(0,15):
            pixel_sum = [(pixels[mtnrange]['count']) for pixels in combo]
            total_pixels.append((pixel_sum))

        total_pixels_df = pd.DataFrame(total_pixels)
        total_pixel_dfs.append(total_pixels_df)

    total_area_df = [ds.multiply(area2pixel_ratio,axis=0).set_index(shapefile.Region) for ds in total_pixel_dfs] 

    return total_area_df


surface_pixel_sums = [surface_PP_sum,surface_NP_sum,surface_PN_sum,surface_NN_sum]
nival_pixel_sums = [nival_PP_sum,nival_NP_sum,nival_PN_sum,nival_NN_sum]
ttopA_pixel_sums = [ttopA_PP_sum,ttopA_NP_sum,ttopA_PN_sum,ttopA_NN_sum]
ttopB_pixel_sums = [ttopB_PP_sum,ttopB_NP_sum,ttopB_PN_sum,ttopB_NN_sum]
obu_pixel_sums = [obu_PP_sum,obu_NP_sum,obu_PN_sum,obu_NN_sum]
ran_pixel_sums = [ran_PP_sum,ran_NP_sum,ran_PN_sum,ran_NN_sum]
gruber_pixel_sums = [gruber_PP_sum,gruber_NP_sum,gruber_PN_sum,gruber_NN_sum]

surface_magt_pzi_areas = calculate_area(surface_pixel_sums)
nival_magt_pzi_areas = calculate_area(nival_pixel_sums)
ttopA_magt_pzi_areas = calculate_area(ttopA_pixel_sums)
ttopB_magt_pzi_areas = calculate_area(ttopB_pixel_sums)
obu_magt_pzi_areas = calculate_area(obu_pixel_sums)
ran_magt_pzi_areas = calculate_area(ran_pixel_sums)
gruber_magt_pzi_areas = calculate_area(gruber_pixel_sums)

savepath = r'\HMA_Permafrost\Data\Tables\change_maps'
for i in range(0,4):
    surface_magt_pzi_areas[i].to_csv(savepath+r'\{:02d}_surface.csv'.format(i))
    nival_magt_pzi_areas[i].to_csv(savepath+r'\{:02d}_nival.csv'.format(i))
    ttopA_magt_pzi_areas[i].to_csv(savepath+r'\{:02d}_gldas_ttop.csv'.format(i))
    ttopB_magt_pzi_areas[i].to_csv(savepath+r'\{:02d}_era5_ttop.csv'.format(i))
    obu_magt_pzi_areas[i].to_csv(savepath+r'\{:02d}_obu_ttop.csv'.format(i))
    ran_magt_pzi_areas[i].to_csv(savepath+r'\{:02d}_ran_dzaa.csv'.format(i))
    gruber_magt_pzi_areas[i].to_csv(savepath+r'\{:02d}_gruber_maat.csv'.format(i))


#Note:
#File Number
#0: PP 
#1: NP
#2: PN
#3: NN

#Columns
#0: between Surface (or self)
#1: between TTOP A PZI (or self)
#2: between TTOP B PZI (or self)
#3: between Obu (or self)
#4: between Ran (or self)
#5: between Gruber (or self)

labels = ['1MAST', '2MAGST','3GLDAS', '4ERA5L', '5Obu', '6Ran','7Gruber']

metrics = []
for change_pzi in ['surface','nival','gldas','era5','obu','ran','gruber']:
    confusion_matrix = [pd.read_csv(file).rename(columns={'0':'surface','1':'nival','2':'gldas','3':'era5','4':'obu','5':'ran','6':'gruber'})
                         for file in sorted(glob.glob(r'\HMA_Permafrost\Data\Tables\change_maps\*{}*.csv'.format(change_pzi)))]
    
    change_surface = [[confusion_matrix[i]['{}'.format(key)] for i in range(0,4)] for key in ['surface','nival','gldas','era5','obu','ran','gruber']]
    vs_pzi = [pd.concat(sets,axis=1).set_axis(range(4),axis=1).rename(columns={0:'PP',1:'NP',2:'PN',3:'NN'}) for sets in change_surface]
    total_area_by_mtn = [vs.sum(axis=1) for vs in vs_pzi]
    total_area_by_cond = [vs.sum(axis=0) for vs in vs_pzi]
    metrics.append([vs_pzi,total_area_by_mtn,total_area_by_cond])

#0,1,2
# 0 [7] -- call pzi change map source
# 1 [3] -- call which dataset to get (vs_pzi; total area by mtn; total area by condition)
# 2 [7] -- call pzi change map target

for source,name in zip(metrics,labels):
    print(name)
    total_permafrost_area = [source[2][i].loc[['PP','NP']].sum() for i in range(0,7)]
    #print(total_permafrost_area)

    total_area =  [source[2][i].sum() for i in range(0,7)]
    #print(total_area)

    percents = [source[2][i]/total_area[i]*100 for i in range(0,7)]
    common_percents = [percents[i].loc[['PP','NN']].sum() for i in range(0,7)]

    [print(percents[i]) for i in  range(0,7)]
    print(common_percents,'\n\n')