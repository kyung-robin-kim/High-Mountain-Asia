import xarray as xr
import rioxarray
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import gc
import glob

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

def unique_vals(array,roundoff):
    unique_values = np.unique(np.round(array,roundoff))
    return unique_values

def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(100,80))
    image = ax1.imshow(array,cmap = 'RdBu_r',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))

def thawing_index(lst):
    TDD = lst*(lst>0)
    count = lst>0
    return TDD,count

def freezing_index(lst):
    FDD = lst*(lst<=0)
    count = lst<=0
    return FDD,count

def surface_temp(thaw,frozen,nf):
    gst = (thaw + nf*frozen)
    lst = (thaw + frozen)
    return gst,lst

def sum_nfs(nf_A,nf_B):
    return np.nansum([nf_A,nf_B],axis=0)

def smrise_snow_only(temperatures, mat, snow):
    exp_params = pd.read_csv(r'\nf_SmithRiseborough\parameters_doubleexponential.csv',delimiter=',')
    nf_0 = np.exp(exp_params['a'][0]*np.exp(exp_params['b'][0]*snow.where(mat >= 5))) + np.exp(exp_params['c'][0]*np.exp(exp_params['d'][0]*snow.where(mat >= 5)))
    nf_1 = np.exp(exp_params['a'][1]*np.exp(exp_params['b'][1]*snow.where((mat >=2) & (mat <5)))) + np.exp(exp_params['c'][1]*np.exp(exp_params['d'][1]*snow.where((mat >=2) & (mat <5))))
    nf_2 = np.exp(exp_params['a'][2]*np.exp(exp_params['b'][2]*snow.where((mat>=0) & (mat<2)))) + np.exp(exp_params['c'][2]*np.exp(exp_params['d'][2]*snow.where((mat>=0) & (mat<2))))
    nf_3 = np.exp(exp_params['a'][3]*np.exp(exp_params['b'][3]*snow.where((mat >= -2) & (mat<0)))) + np.exp(exp_params['c'][3]*np.exp(exp_params['d'][3]*snow.where((mat >= -2) & (mat<0))))
    nf_4 = np.exp(exp_params['a'][4]*np.exp(exp_params['b'][4]*snow.where((mat >= -4) & (mat<-2)))) + np.exp(exp_params['c'][4]*np.exp(exp_params['d'][4]*snow.where((mat >= -4) & (mat<-2))))
    nf_5 = np.exp(exp_params['a'][5]*np.exp(exp_params['b'][5]*snow.where((mat >= -6) & (mat<-4)))) + np.exp(exp_params['c'][5]*np.exp(exp_params['d'][5]*snow.where((mat >= -6) & (mat<-4))))
    nf_6 = np.exp(exp_params['a'][6]*np.exp(exp_params['b'][6]*snow.where((mat >= -8) & (mat<-6)))) + np.exp(exp_params['c'][6]*np.exp(exp_params['d'][6]*snow.where((mat >= -8) & (mat<-6))))
    nf_7 = np.exp(exp_params['a'][7]*np.exp(exp_params['b'][7]*snow.where((mat >= -10) & (mat<-8)))) + np.exp(exp_params['c'][7]*np.exp(exp_params['d'][7]*snow.where((mat >= -10) & (mat<-8))))
    nf_8 = np.exp(exp_params['a'][8]*np.exp(exp_params['b'][8]*snow.where(mat<-10))) + np.exp(exp_params['c'][8]*np.exp(exp_params['d'][8]*snow.where(mat<-10)))

    nf_summation = [nf_0,nf_1,nf_2,nf_3,nf_4,nf_5,nf_6,nf_7,nf_8]
    nfs = xr.DataArray(np.empty(shape=(np.shape(snow)))).rename({'dim_0':'time', 'dim_1':'lat','dim_2':'lon'})
    for i in nf_summation:
        nfs = xr.apply_ufunc(sum_nfs,nfs,i,
                            input_core_dims=[['time','lat','lon'],['time','lat','lon']], output_core_dims=[['time','lat', 'lon']],output_dtypes=[np.float32],
                            dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)
        del i
        gc.collect()

    del nf_summation
    gc.collect()

    TDD, T_count= xr.apply_ufunc(thawing_index,temperatures,input_core_dims=[['time','lat','lon']],output_core_dims=[['time','lat','lon'],['time','lat','lon']],
                                 dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)
    FDD, F_count= xr.apply_ufunc(freezing_index,temperatures,input_core_dims=[['time','lat','lon']],output_core_dims=[['time','lat','lon'],['time','lat','lon']],
                                 dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)
    gst, lst = xr.apply_ufunc(surface_temp,TDD,FDD,nfs,input_core_dims=[['time','lat','lon'],['time','lat','lon'],['time','lat','lon']],output_core_dims=[['time','lat','lon'],['time','lat','lon']],
                              dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)

    return(gst,lst,nfs,TDD,FDD,T_count,F_count)


################################
###Smith & Riseborough,  2002

#Only snow factor correction (NIVAL only)
#MONTHLY - MODMYD11_AIRS
sd_path = r'\SnowDepth\MONTHLY_MEAN'
lst_path = sorted(glob.glob(r'\multi_elev_lat\MALST\*.nc'))[0]
MLST = xr.open_mfdataset(lst_path,parallel=True,chunks={"time": 1,"y":225,"x":225}).rio.write_crs("epsg:4326").MLST.rename({'time':'month'}).set_index(month='month')
years = range(2008,2017)

for year in years:
    sd_nc = glob.glob(sd_path+'\\{}\\*.nc'.format(year))
    snow = xr.open_mfdataset(sd_nc,parallel=True,chunks={"month": 1,"y":225,"x":225}).rio.write_crs("epsg:4326").rename({'__xarray_dataarray_variable__':'SD_m'}).SD_m[:,0,:,:].interp(y=MLST["y"], x=MLST["x"], method='nearest').fillna(0)

    mat_path = r'\multi_elev_lat\MALST\{}.tif'.format(year)
    mat = xr.open_mfdataset(mat_path,parallel=True,chunks={"month": 1,"y":225,"x":225}).rio.write_crs("epsg:4326").band_data[0]

    exp_params = pd.read_csv(r'C:\Users\robin\Box\HMA_robin\02-data\csv\nf_SmithRiseborough\parameters_doubleexponential.csv',delimiter=',')
    nf_0 = np.exp(exp_params['a'][0]*np.exp(exp_params['b'][0]*np.where((mat >= 5),snow,np.nan))) + np.exp(exp_params['c'][0]*np.exp(exp_params['d'][0]*np.where((mat >= 5),snow,np.nan)))
    nf_1 = np.exp(exp_params['a'][1]*np.exp(exp_params['b'][1]*np.where((mat >=2) & (mat <5),snow,np.nan))) + np.exp(exp_params['c'][1]*np.exp(exp_params['d'][1]*np.where((mat >=2) & (mat <5),snow,np.nan)))
    nf_2 = np.exp(exp_params['a'][2]*np.exp(exp_params['b'][2]*np.where((mat>=0) & (mat<2),snow,np.nan))) + np.exp(exp_params['c'][2]*np.exp(exp_params['d'][2]*np.where((mat>=0) & (mat<2),snow,np.nan)))
    nf_3 = np.exp(exp_params['a'][3]*np.exp(exp_params['b'][3]*np.where((mat >= -2) & (mat<0),snow,np.nan))) + np.exp(exp_params['c'][3]*np.exp(exp_params['d'][3]*np.where((mat >= -2) & (mat<0),snow,np.nan)))
    nf_4 = np.exp(exp_params['a'][4]*np.exp(exp_params['b'][4]*np.where((mat >= -4) & (mat<-2),snow,np.nan))) + np.exp(exp_params['c'][4]*np.exp(exp_params['d'][4]*np.where((mat >= -4) & (mat<-2),snow,np.nan)))
    nf_5 = np.exp(exp_params['a'][5]*np.exp(exp_params['b'][5]*np.where((mat >= -6) & (mat<-4),snow,np.nan))) + np.exp(exp_params['c'][5]*np.exp(exp_params['d'][5]*np.where((mat >= -6) & (mat<-4),snow,np.nan)))
    nf_6 = np.exp(exp_params['a'][6]*np.exp(exp_params['b'][6]*np.where((mat >= -8) & (mat<-6),snow,np.nan))) + np.exp(exp_params['c'][6]*np.exp(exp_params['d'][6]*np.where((mat >= -8) & (mat<-6),snow,np.nan)))
    nf_7 = np.exp(exp_params['a'][7]*np.exp(exp_params['b'][7]*np.where((mat >= -10) & (mat<-8),snow,np.nan))) + np.exp(exp_params['c'][7]*np.exp(exp_params['d'][7]*np.where((mat >= -10) & (mat<-8),snow,np.nan)))
    nf_8 = np.exp(exp_params['a'][8]*np.exp(exp_params['b'][8]*np.where((mat<-10),snow,np.nan))) + np.exp(exp_params['c'][8]*np.exp(exp_params['d'][8]*np.where((mat<-10),snow,np.nan)))

    nf_summation = [nf_0,nf_1,nf_2,nf_3,nf_4,nf_5,nf_6,nf_7,nf_8]
    nfs = xr.DataArray(np.empty(shape=(np.shape(snow)))).rename({'dim_0':'month', 'dim_1':'y','dim_2':'x'})
    for i in nf_summation:
        nfs = xr.apply_ufunc(sum_nfs,nfs,i,
                            input_core_dims=[['month','y','x'],['month','y','x']], output_core_dims=[['month','y', 'x']],output_dtypes=[np.float32],
                            dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)
        del i
        gc.collect()

    del nf_summation
    gc.collect()

    MMLST = MLST.where(MLST['month'].dt.year.isin(year), drop=True)

    TDD, T_count= xr.apply_ufunc(thawing_index,MMLST,input_core_dims=[['month','y','x']],output_core_dims=[['month','y','x'],['month','y','x']],
                                 dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)
    FDD, F_count= xr.apply_ufunc(freezing_index,MMLST,input_core_dims=[['month','y','x']],output_core_dims=[['month','y','x'],['month','y','x']],
                                 dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)
    gst = (TDD + nfs*FDD)
    
    gst = gst.rio.set_spatial_dims('x','y',inplace=True).rio.set_crs("epsg:4326")
    [gst[i-1].rio.to_raster(r'\SmRise_NivalOnly\{}_{:02d}.tif'.format(year,i)) for i in range(1,13)]

    del gst,snow,TDD,FDD,MMLST
    gc.collect()


#MODMYD11_AIRS
all_arrays = sorted(glob.glob(r'\MAGT\MODMYD11_AIRS\GF_multi_elev_lat\SmRise_NivalOnly\*.tif'))
datetimes = pd.date_range('2003-01','2017-01',freq='M')
dataset = xr.concat([xr.open_rasterio(array)[0] for array in all_arrays], dim='time')
dataset['time']=datetimes
dataset = dataset.rename('MGT')
dataset.to_netcdf(r'\MAGT\MODMYD11_AIRS\GF_multi_elev_lat\SmRise_NivalOnly\MAGT\MODMYD11_AIRS_monthly.nc')
gc.collect()
dataset.mean(dim='time').rio.to_raster(r'\MAGT\MODMYD11_AIRS\GF_multi_elev_lat\SmRise_NivalOnly\MAGT\2003_2016_MAGT_nival.tif')
del dataset
gc.collect()


############################
#TTOP - snow and thermal offset

#MONTHLY MODMYD11_AIRS 
sd_path = r'\SnowDepth\MONTHLY_MEAN'
lst_path = sorted(glob.glob(r'\multi_elev_lat\MALST\*.nc'))[0]
MLST = xr.open_mfdataset(lst_path,parallel=True,chunks={"time": 1,"y":225,"x":225}).rio.write_crs("epsg:4326").MLST.rename({'time':'month'}).set_index(month='month')
years = range(2003,2017)

for sm_source in ['ERA5_L','GLDAS']:
    for year in years:
        sd_nc = glob.glob(sd_path+'\\{}\\*.nc'.format(year))
        snow = xr.open_mfdataset(sd_nc,parallel=True,chunks={"month": 1,"y":225,"x":225}).rio.write_crs("epsg:4326").rename({'__xarray_dataarray_variable__':'SD_m'}).SD_m[:,0,:,:].interp(y=MLST["y"], x=MLST["x"], method='nearest').fillna(0)

        mat_path = r'\multi_elev_lat\MALST\{}.tif'.format(year)
        mat = xr.open_mfdataset(mat_path,parallel=True,chunks={"month": 1,"y":225,"x":225}).rio.write_crs("epsg:4326").band_data[0]

        rk_path = sorted(glob.glob(r'\HMA_Permafrost\rk\{}\{}*.tif'.format(sm_source,year)))
        rks = xr.concat([xr.open_dataarray(file)[0] for file in rk_path],dim='month')


        exp_params = pd.read_csv(r'\nf_SmithRiseborough\parameters_doubleexponential.csv',delimiter=',')
        nf_0 = np.exp(exp_params['a'][0]*np.exp(exp_params['b'][0]*np.where((mat >= 5),snow,np.nan))) + np.exp(exp_params['c'][0]*np.exp(exp_params['d'][0]*np.where((mat >= 5),snow,np.nan)))
        nf_1 = np.exp(exp_params['a'][1]*np.exp(exp_params['b'][1]*np.where((mat >=2) & (mat <5),snow,np.nan))) + np.exp(exp_params['c'][1]*np.exp(exp_params['d'][1]*np.where((mat >=2) & (mat <5),snow,np.nan)))
        nf_2 = np.exp(exp_params['a'][2]*np.exp(exp_params['b'][2]*np.where((mat>=0) & (mat<2),snow,np.nan))) + np.exp(exp_params['c'][2]*np.exp(exp_params['d'][2]*np.where((mat>=0) & (mat<2),snow,np.nan)))
        nf_3 = np.exp(exp_params['a'][3]*np.exp(exp_params['b'][3]*np.where((mat >= -2) & (mat<0),snow,np.nan))) + np.exp(exp_params['c'][3]*np.exp(exp_params['d'][3]*np.where((mat >= -2) & (mat<0),snow,np.nan)))
        nf_4 = np.exp(exp_params['a'][4]*np.exp(exp_params['b'][4]*np.where((mat >= -4) & (mat<-2),snow,np.nan))) + np.exp(exp_params['c'][4]*np.exp(exp_params['d'][4]*np.where((mat >= -4) & (mat<-2),snow,np.nan)))
        nf_5 = np.exp(exp_params['a'][5]*np.exp(exp_params['b'][5]*np.where((mat >= -6) & (mat<-4),snow,np.nan))) + np.exp(exp_params['c'][5]*np.exp(exp_params['d'][5]*np.where((mat >= -6) & (mat<-4),snow,np.nan)))
        nf_6 = np.exp(exp_params['a'][6]*np.exp(exp_params['b'][6]*np.where((mat >= -8) & (mat<-6),snow,np.nan))) + np.exp(exp_params['c'][6]*np.exp(exp_params['d'][6]*np.where((mat >= -8) & (mat<-6),snow,np.nan)))
        nf_7 = np.exp(exp_params['a'][7]*np.exp(exp_params['b'][7]*np.where((mat >= -10) & (mat<-8),snow,np.nan))) + np.exp(exp_params['c'][7]*np.exp(exp_params['d'][7]*np.where((mat >= -10) & (mat<-8),snow,np.nan)))
        nf_8 = np.exp(exp_params['a'][8]*np.exp(exp_params['b'][8]*np.where((mat<-10),snow,np.nan))) + np.exp(exp_params['c'][8]*np.exp(exp_params['d'][8]*np.where((mat<-10),snow,np.nan)))

        nf_summation = [nf_0,nf_1,nf_2,nf_3,nf_4,nf_5,nf_6,nf_7,nf_8]
        nfs = xr.DataArray(np.empty(shape=(np.shape(snow)))).rename({'dim_0':'month', 'dim_1':'y','dim_2':'x'})
        for i in nf_summation:
            nfs = xr.apply_ufunc(sum_nfs,nfs,i,
                                input_core_dims=[['month','y','x'],['month','y','x']], output_core_dims=[['month','y', 'x']],output_dtypes=[np.float32],
                                dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)
            del i
            gc.collect()

        del nf_summation
        gc.collect()

        MMLST = MLST.where(MLST['month'].dt.year.isin(year), drop=True)

        TDD, T_count= xr.apply_ufunc(thawing_index,MMLST,input_core_dims=[['month','y','x']],output_core_dims=[['month','y','x'],['month','y','x']],
                                    dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)
        FDD, F_count= xr.apply_ufunc(freezing_index,MMLST,input_core_dims=[['month','y','x']],output_core_dims=[['month','y','x'],['month','y','x']],
                                    dask='parallelized',dask_gufunc_kwargs={'allow_rechunk':True},vectorize=True)
        
        TDD_total = TDD.sum(dim='month')
        FDD_total = FDD.sum(dim='month')

        Tcount_total = T_count.sum(dim='month')
        Fcount_total = F_count.sum(dim='month')

        mask_rk = (FDD_total*-1)>(TDD_total)
        test=xr.DataArray(np.array(rks[0]*mask_rk))
                 
        thermal_offset = xr.concat([xr.DataArray(np.array(rks[i])*np.array(TDD[i])) for i in range(0,12)],dim='month').rename({'dim_0':'y', 'dim_1':'x'})
        gst = (thermal_offset + nfs*FDD)
        
        gst = gst.rio.set_spatial_dims('x','y',inplace=True).rio.set_crs("epsg:4326")
        [gst[i-1].rio.to_raster(r'\SmRise_ThermalOffset\{}\{}_{:02d}.tif'.format(sm_source,year,i)) for i in range(1,13)]

        del gst,MMLST,snow,TDD,FDD
        gc.collect()


#Call monthly SM
ERA5_path=r'\ERA5\MONTHLY_SM\sm_avg.nc'
GLDAS_path=r'\GLDAS\NOAH\HMA\soil_moisture\monthly_sm_weight_avg.nc'
sm_monthly = (xr.open_mfdataset(GLDAS_path,parallel=True,chunks={"latitude":30,"longitude":60,'time': 1})).rename({'__xarray_dataarray_variable__':'VWC'})

#Williams and Smith (2008) // Monteith (1973)
k_water = 0.56
k_ice = 2.24

years = range(2003,2017)
for year in years:
    mat_path = r'\multi_elev_lat\MALST\{}.tif'.format(year)
    mat = xr.open_mfdataset(mat_path,parallel=True,chunks={"month": 1,"y":225,"x":225}).rio.write_crs("epsg:4326").band_data[0]

    MS = sm_monthly.VWC.sel(time='{}'.format(year)).rio.write_crs("epsg:4326")
    MS_1km = MS.rio.reproject_match(mat, resampling=rio.enums.Resampling.bilinear)
    r_k = 1/((k_ice/k_water)**MS_1km).rename({'time':'month'})
    r_k = r_k.rio.set_spatial_dims('x','y',inplace=True).rio.set_crs("epsg:4326")
    [r_k[i].rio.to_raster(r'\rk\{}\{}_{:02d}_k.tif'.format(year,i+1)) for i in range(0,12)]


#MODMYD11_AIRS
for sm_source in ['ERA5_L','GLDAS']:
    all_arrays = sorted(glob.glob(r'\SmRise_ThermalOffset\{}\*.tif'.format(sm_source)))
    datetimes = pd.date_range('2003-01','2017-01',freq='M')
    dataset = xr.concat([xr.open_rasterio(array)[0] for array in all_arrays], dim='time')
    dataset['time']=datetimes
    dataset = dataset.rename('MGT')
    dataset.to_netcdf(r'\SmRise_ThermalOffset\{}\MAGT\MODMYD11_AIRS_monthly.nc'.format(sm_source))
    gc.collect()
    dataset.mean(dim='time').rio.to_raster(r'\SmRise_ThermalOffset\{}\MAGT\2003_2016_MAGT_thermaloffset.tif'.format(sm_source))
    del dataset
    gc.collect()


#rk netcdf
for sm_source in ['ERA5_L','GLDAS']:
    all_arrays = sorted(glob.glob(r'\HMA_Permafrost\rk\{}\*.tif'.format(sm_source)))
    datetimes = pd.date_range('2003-01','2017-01',freq='M')
    dataset = xr.concat([xr.open_rasterio(array)[0] for array in all_arrays], dim='time')
    dataset['time']=datetimes
    dataset = dataset.rename('rk')
    dataset.to_netcdf(r'\HMA_Permafrost\rk\{}\rks_monthly.nc'.format(sm_source))
    del dataset
    gc.collect()