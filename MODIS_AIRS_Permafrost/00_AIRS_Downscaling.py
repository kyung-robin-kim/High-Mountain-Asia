import rasterio as rio
import os
import glob
from scipy import special
import math
import matplotlib.pyplot as plt 
import matplotlib
from pathlib import Path
import numpy as np
import seaborn as sns
import xarray as xr
import rioxarray
import geopandas as gpd
import pandas as pd
from scipy import special
from shapely.geometry import box, mapping
from rasterio.enums import Resampling
import gc
from scipy import stats
import itertools
import geopandas as gpd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def linear_plot(temp,lat,elev,title):
    try:

        nanmask = ~np.isnan(temp) & ~np.isnan(elev) & ~np.isnan(lat)
        temp = temp[nanmask]
        lat = lat[nanmask]
        elev = elev[nanmask]

        independent = [[elev[i],lat[i]] for i in range(0,len(temp))]
        M = sm.add_constant(independent)
        multiregress = sm.OLS(temp, M)
        fitted_multiregress = multiregress.fit()
        modeled_values = fitted_multiregress.predict(M)

        for var,unit in zip([elev,lat],['Elevation (m)','Latitude']):
            slope, intercept, r_value, p_value, std_err = stats.linregress(var,temp)
            fig,ax = plt.subplots()
            ax.scatter(var, temp)
            ax.plot(var, intercept + slope*var, label='r: {}'.format(round(r_value,3)),color='r')
            ax.set_ylabel('Surface Temp (C)')
            ax.set_xlabel(unit)
            ax.legend()
            ax.set_ylim(-30,30)
            ax.text(40,-5,s="p-val: {}".format(round(p_value,4)), fontsize=10, ha="left", va="top")
            ax.text(40,-10,s="slope: {}".format(round(slope,4)), fontsize=10, ha="left", va="top")
            #plt.savefig(r'\{}\{}.png'.format(path,title))

    except ValueError:
        print('error {}'.format(title))

def read_file(file):
    with rio.open(file) as src:
        return(src.read(1))

degree_sign = u"\N{DEGREE SIGN}"
plt.rcParams["font.family"] = "Times New Roman"

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


dem_paths = sorted(glob.glob(r'\MERIT_DEM\*.tif'))
dem_1km = xr.open_mfdataset(dem_paths[0],parallel=True,chunks={'y':1000,'x':1000}).band_data[0]

airs_path = sorted(glob.glob(r'\AIRS\SurfaceSkinTemp\HMA\MONTHLY\**\*.tif'))
airs_res = xr.open_mfdataset(airs_path[0],parallel=True,chunks={'y':5,'x':5}).band_data[0].rio.write_crs('epsg:4326')
dem_111km = dem_1km.rio.reproject_match(airs_res, resampling=5)


def lapse_rate_R2(temp,lat,elev):
    try:
        nanmask = ~np.isnan(temp) & ~np.isnan(elev) & ~np.isnan(lat)
        temp = temp[nanmask]
        lat = lat[nanmask]
        elev = elev[nanmask]

        independent = [[elev[i],lat[i]] for i in range(0,len(temp))]
        M = sm.add_constant(independent)
        multiregress = sm.OLS(temp, M)
        fitted_multiregress = multiregress.fit()

        return fitted_multiregress.rsquared
        
    except ValueError:
        print('missing')
        return np.nan

def lapse_rate_pval(temp,lat,elev):
    try:
        nanmask = ~np.isnan(temp) & ~np.isnan(elev) & ~np.isnan(lat)
        temp = temp[nanmask]
        lat = lat[nanmask]
        elev = elev[nanmask]

        independent = [[elev[i],lat[i]] for i in range(0,len(temp))]
        M = sm.add_constant(independent)
        multiregress = sm.OLS(temp, M)
        fitted_multiregress = multiregress.fit()

        return fitted_multiregress.pvalues
        
    except ValueError:
        print('missing')
        return [np.nan,np.nan,np.nan]

def lapse_rate_predict(temp,lat,elev,data):
    try:
        nanmask = ~np.isnan(temp) & ~np.isnan(elev) & ~np.isnan(lat)
        temp = temp[nanmask]
        lat = lat[nanmask]
        elev = elev[nanmask]

        independent = [[elev[i],lat[i]] for i in range(0,len(temp))]
        M = sm.add_constant(independent)
        multiregress = sm.OLS(temp, M)
        fitted_multiregress = multiregress.fit()

        return fitted_multiregress.predict(data)
        
    except ValueError:
        print('missing')


def lapse_rate_b0(temp,lat,elev):
    try:
        nanmask = ~np.isnan(temp) & ~np.isnan(elev) & ~np.isnan(lat)
        temp = temp[nanmask]
        lat = lat[nanmask]
        elev = elev[nanmask]

        independent = [[elev[i],lat[i]] for i in range(0,len(temp))]
        M = sm.add_constant(independent)
        multiregress = sm.OLS(temp, M)
        fitted_multiregress = multiregress.fit()

        return fitted_multiregress.params[0]
    
    except ValueError:
        print('missing')
        return np.nan

def lapse_rate_b1(temp,lat,elev):
    try:
        nanmask = ~np.isnan(temp) & ~np.isnan(elev) & ~np.isnan(lat)
        temp = temp[nanmask]
        lat = lat[nanmask]
        elev = elev[nanmask]

        independent = [[elev[i],lat[i]] for i in range(0,len(temp))]
        M = sm.add_constant(independent)
        multiregress = sm.OLS(temp, M)
        fitted_multiregress = multiregress.fit()

        return fitted_multiregress.params[1]
        
    except ValueError:
        print('missing')
        return np.nan

def lapse_rate_b2(temp,lat,elev):
    try:
        nanmask = ~np.isnan(temp) & ~np.isnan(elev) & ~np.isnan(lat)
        temp = temp[nanmask]
        lat = lat[nanmask]
        elev = elev[nanmask]

        independent = [[elev[i],lat[i]] for i in range(0,len(temp))]
        M = sm.add_constant(independent)
        multiregress = sm.OLS(temp, M)
        fitted_multiregress = multiregress.fit()

        return fitted_multiregress.params[2]
        
    except ValueError:
        print('missing')
        return np.nan

#Daily Downscaled Skin Temp (DAY/NIGHT)

D_dates=[]
D_rs = []
D_pvals = []
D_b0s = []
D_b1s = []
D_b2s = []

N_dates=[]
N_rs = []
N_pvals = []
N_b0s = []
N_b1s = []
N_b2s = []

D_arrays = [D_dates,D_rs,D_pvals,D_b0s,D_b1s,D_b2s]
N_arrays = [N_dates,N_rs,N_pvals,N_b0s,N_b1s,N_b2s]
for year in range(2003,2017):
    for month in range(1,13):
        D_skin = sorted(glob.glob(r'\AIRS\SurfaceSkinTemp\HMA\DAILY\DAY\{}-{:02d}*.tif'.format(year,month)))
        N_skin = sorted(glob.glob(r'\AIRS\SurfaceSkinTemp\HMA\DAILY\NIGHT\{}-{:02d}*.tif'.format(year,month)))

        for skin_temp,savepath,array in zip([D_skin,N_skin],
                                            [r'\AIRS\SurfaceSkinTemp\HMA\DAILY\DAY\downscaled_MERIT_1km\multi_elev_lat',
                                             r'\AIRS\SurfaceSkinTemp\HMA\DAILY\NIGHT\downscaled_MERIT_1km\multi_elev_lat'],
                                            [D_arrays,N_arrays]):

            all_dates =  np.array([pd.to_datetime(file[-14:-4]) for file in skin_temp])
            daily_skin = [xr.open_dataarray(file)[0].where(xr.open_dataarray(file)[0]>0,np.nan)/100 - 273.15 for file in skin_temp]
            X_1deg,Y_1deg = np.meshgrid(daily_skin[0].x,daily_skin[0].y)

            #[linear_plot(temps.values.reshape(-1),Y.reshape(-1),Y.reshape(-1),dem_111km.values.reshape(-1),date.strftime('%Y%m%d')) for temps,date in zip(daily_skin,dates)]
            
            X_1km,Y_1km = np.meshgrid(dem_1km.x,dem_1km.y)
            constant_column = np.ones(dem_1km.shape)
            M = np.stack((constant_column,dem_1km,Y_1km),axis=2)

            r_s = [lapse_rate_R2(temps.values.reshape(-1),Y_1deg.reshape(-1),dem_111km.values.reshape(-1)) for temps in daily_skin]
            p_vals = [lapse_rate_pval(temps.values.reshape(-1),Y_1deg.reshape(-1),dem_111km.values.reshape(-1)) for temps in daily_skin]
            b0s = [lapse_rate_b0(temps.values.reshape(-1),Y_1deg.reshape(-1),dem_111km.values.reshape(-1)) for temps in daily_skin]
            b1s = [lapse_rate_b1(temps.values.reshape(-1),Y_1deg.reshape(-1),dem_111km.values.reshape(-1)) for temps in daily_skin]
            b2s = [lapse_rate_b2(temps.values.reshape(-1),Y_1deg.reshape(-1),dem_111km.values.reshape(-1)) for temps in daily_skin]
            #surface_temps_1km_models = [lapse_rate_predict(temps.values.reshape(-1),Y_1deg.reshape(-1),dem_111km.values.reshape(-1),M) for temps in daily_skin]

            dates = all_dates[~np.isnan(r_s)]
            p_vals = np.array(p_vals)[~np.isnan(r_s)]
            b0s = np.array(b0s)[~np.isnan(r_s)]
            b1s = np.array(b1s)[~np.isnan(r_s)]
            b2s = np.array(b2s)[~np.isnan(r_s)]
            r_s = np.array(r_s)[~np.isnan(r_s)]

            #st_1km_models = [st for st in surface_temps_1km_models if st is not None]

            #ds_st_1km= [xr.DataArray(da).rename({'dim_0':'y','dim_1':'x'}).assign_coords({'y':dem_1km.y,'x':dem_1km.x}).rename('ST_1km') for da in st_1km_models]
            #[(ds_st_1km[i]).rio.to_raster(savepath+r'\{}.tif'.format(date.strftime('%Y-%m-%d'))) for i,date in zip(range(0,len(dates)),dates)]

            array[0].append(dates)
            array[1].append(r_s)
            array[2].append(p_vals)
            array[3].append(b0s)
            array[4].append(b1s)
            array[5].append(b2s)

            del M,r_s,p_vals,b0s,b1s,b2s
            gc.collect()


D_dates = list(itertools.chain.from_iterable(D_arrays[0]))
D_rs = list(itertools.chain.from_iterable(D_arrays[1]))
D_pvals = list(itertools.chain.from_iterable(D_arrays[2]))
D_b0s = list(itertools.chain.from_iterable(D_arrays[3]))
D_b1s = list(itertools.chain.from_iterable(D_arrays[4]))
D_b2s = list(itertools.chain.from_iterable(D_arrays[5]))

N_dates = list(itertools.chain.from_iterable(N_arrays[0]))
N_rs = list(itertools.chain.from_iterable(N_arrays[1]))
N_pvals = list(itertools.chain.from_iterable(N_arrays[2]))
N_b0s = list(itertools.chain.from_iterable(N_arrays[3]))
N_b1s = list(itertools.chain.from_iterable(N_arrays[4]))
N_b2s = list(itertools.chain.from_iterable(N_arrays[5]))

D_df = pd.DataFrDe([D_dates,D_rs,D_pvals,D_b0s,D_b1s,D_b2s]).T.rename(columns={0:'date',1:'R2',2:'pval',3:'intercept',4:'B1_lapserate',5:'B2'})
N_df = pd.DataFrame([N_dates,N_rs,N_pvals,N_b0s,N_b1s,N_b2s]).T.rename(columns={0:'date',1:'R2',2:'pval',3:'intercept',4:'B1_lapserate',5:'B2'})

D_df.set_index('date',inplace=True)
N_df.set_index('date',inplace=True)

D_df.to_csv(r'\regression_metrics_D.csv')
N_df.to_csv(r'\regression_metrics_N.csv')
