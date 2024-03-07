#Import functions

import pandas as pd
import xarray as xr
import rioxarray
import rasterio as rio
import numpy as np
import gc
import glob

###########################################################################
#MOD11 + MYD11 + AIRS Skin Temperature Daily to Monthly

#FOR MYD11
#Define path
path=r'\MOD_MYD_11'
#Define savepath
savepath=r'\MYD11\Monthly'
qc_day = pd.read_csv(r'\MYD11A1-006-QC-Day-lookup.csv')
qc_night = pd.read_csv(r'\MYD11A1-006-QC-Night-lookup.csv')

qcd_1 = qc_day[(qc_day['LST Error Flag'] == 'Average LST error <= 1K')]['Value']
qcn_1 = qc_night[(qc_night['LST Error Flag'] == 'Average LST error <= 1K')]['Value']


#FOR MOD11
#Define path
modpath=r'\MOD_MYD_11'
#Define savepath
modsavepath=r'\MOD11\Monthly'
modqc_day = pd.read_csv(r'\MOD11A1-006-QC-Day-lookup.csv')
modqc_night = pd.read_csv(r'\MOD11A1-006-QC-Night-lookup.csv')

modqcd_1 = modqc_day[(modqc_day['LST Error Flag'] == 'Average LST error <= 1K')]['Value']
modqcn_1 = modqc_night[(modqc_night['LST Error Flag'] == 'Average LST error <= 1K')]['Value']


months = np.arange(1,13)
years = np.arange(2003,2017) 

for year in years:

    for month in months:

        
        mod11 = sorted(glob.glob(r'\MOD_MYD_11\MOD_11\{}\*.nc'.format(year)))[0]
        myd11 = sorted(glob.glob(r'\MOD_MYD_11\MYD_11\{}\*.nc'.format(year)))[0]

        lst_day = xr.open_mfdataset(myd11,parallel=True,chunks={"y":100,"x":100,'time': 10}).LST_Day_1km.sel(time='{}-'f"{month:02d}".format(year))-273.15
        qc_day =  xr.open_mfdataset(myd11,parallel=True,chunks={"y":100,"x":100,'time': 10}).QC_Day.sel(time='{}-'f"{month:02d}".format(year))
        mask = qc_day.isin(qcd_1)
        lst_masked_day = lst_day.where(mask).rio.write_crs('epsg:4326').rename({'lat':'y','lon':'x'}).astype('float32')
        lst_masked_day['time'] = pd.to_datetime(lst_masked_day.time.dt.strftime('%Y-%m-%d'))        

        lst_day_mod = xr.open_mfdataset(mod11,parallel=True,chunks={"y":100,"x":100,'time': 10}).LST_Day_1km.sel(time='{}-'f"{month:02d}".format(year))-273.15
        qc_day_mod = xr.open_mfdataset(mod11,parallel=True,chunks={"y":100,"x":100,'time': 10}).QC_Day.sel(time='{}-'f"{month:02d}".format(year))
        mask_mod = qc_day_mod.isin(modqcd_1)
        lst_masked_day_mod = lst_day_mod.where(mask_mod).rio.write_crs('epsg:4326').rename({'lat':'y','lon':'x'}).astype('float32')
        lst_masked_day_mod['time'] = pd.to_datetime(lst_masked_day_mod.time.dt.strftime('%Y-%m-%d'))

        skin_D_files = sorted(glob.glob(r'\AIRS\SurfaceSkinTemp\HMA\DAILY\DAY\downscaled_MERIT_1km\multi_elev_lat\{}-{:02d}*.tif'.format(year,month)))
        skin_D_files_dates = [pd.to_datetime(file[-14:-4]) for file in skin_D_files]
        skin_d = xr.concat([(xr.open_dataarray(file)[0]) for file in skin_D_files],dim='time').rio.reproject_match(lst_masked_day, resampling=0)
        skin_d['time']=skin_D_files_dates

        if len(lst_masked_day.time) == len(skin_d.time):
            lst_gap_filled_day = xr.concat([xr.where(lst.isnull(),skin,lst) for lst,skin in zip(lst_masked_day,skin_d)],dim='time')
        else:
            common_datetimes = xr.DataArray(list(set(pd.to_datetime(lst_masked_day.time.values)) & set(pd.to_datetime(skin_d.time.values))), dims='time')
            common_datetimes = common_datetimes.sortby(common_datetimes)
            lst_gap_filled_day = xr.concat([xr.where(lst.isnull(),skin,lst) for lst,skin in zip(lst_masked_day.loc[common_datetimes],skin_d.loc[common_datetimes])],dim='time')
            uncommon_datetimes = xr.DataArray(list(set(pd.to_datetime(lst_masked_day.time.values)) ^ set(pd.to_datetime(skin_d.time.values))), dims='time')
            uncommon_datetimes = uncommon_datetimes.sortby(uncommon_datetimes)
            if len(lst_masked_day.time) > len(skin_d.time):
                lst_gap_filled_day = xr.concat([lst_gap_filled_day,lst_masked_day.loc[uncommon_datetimes]],dim='time')
            else:
                lst_gap_filled_day = xr.concat([lst_gap_filled_day,skin_d.loc[uncommon_datetimes]],dim='time')
                print('Check missing day data for {}-{:02d}'.format(year,month))
        
        lst_night = xr.open_mfdataset(myd11,parallel=True,chunks={"y":100,"x":100,'time': 10}).LST_Night_1km.sel(time='{}-'f"{month:02d}".format(year))-273.15
        qc_night =  xr.open_mfdataset(myd11,parallel=True,chunks={"y":100,"x":100,'time': 10}).QC_Night.sel(time='{}-'f"{month:02d}".format(year))
        mask = qc_night.isin(qcn_1)
        lst_masked_night = lst_night.where(mask).rio.write_crs('epsg:4326').rename({'lat':'y','lon':'x'}).astype('float32')
        lst_masked_night['time'] = pd.to_datetime(lst_masked_night.time.dt.strftime('%Y-%m-%d'))

        lst_night_mod = xr.open_mfdataset(mod11,parallel=True,chunks={"y":100,"x":100,'time': 10}).LST_Night_1km.sel(time='{}-'f"{month:02d}".format(year))-273.15
        qc_night_mod = xr.open_mfdataset(mod11,parallel=True,chunks={"y":100,"x":100,'time': 10}).QC_Night.sel(time='{}-'f"{month:02d}".format(year))
        mask_mod = qc_night_mod.isin(modqcn_1)
        lst_masked_night_mod = lst_night_mod.where(mask_mod).rio.write_crs('epsg:4326').rename({'lat':'y','lon':'x'}).astype('float32')
        lst_masked_night_mod['time'] = pd.to_datetime(lst_masked_night_mod.time.dt.strftime('%Y-%m-%d'))

        skin_N_files = sorted(glob.glob(r'\AIRS\SurfaceSkinTemp\HMA\DAILY\NIGHT\downscaled_MERIT_1km\multi_elev_lat\{}-{:02d}*.tif'.format(year,month)))
        skin_N_files_dates = [pd.to_datetime(file[-14:-4]) for file in skin_N_files]
        skin_n = xr.concat([(xr.open_dataarray(file)[0]) for file in skin_N_files],dim='time').rio.reproject_match(lst_masked_night, resampling=0)
        skin_n['time']=skin_N_files_dates

        if len(lst_masked_night.time) == len(skin_n.time):
            lst_gap_filled_night = xr.concat([xr.where(lst.isnull(),skin,lst) for lst,skin in zip(lst_masked_night,skin_n)],dim='time')
        else:
            common_datetimes = xr.DataArray(list(set(pd.to_datetime(lst_masked_night.time.values)) & set(pd.to_datetime(skin_n.time.values))), dims='time')
            common_datetimes = common_datetimes.sortby(common_datetimes)
            lst_gap_filled_night = xr.concat([xr.where(lst.isnull(),skin,lst) for lst,skin in zip(lst_masked_night.loc[common_datetimes],skin_n.loc[common_datetimes])],dim='time')
            uncommon_datetimes = xr.DataArray(list(set(pd.to_datetime(lst_masked_night.time.values)) ^ set(pd.to_datetime(skin_n.time.values))), dims='time')
            uncommon_datetimes = uncommon_datetimes.sortby(uncommon_datetimes)
            if len(lst_masked_night.time) > len(skin_n.time):
                lst_gap_filled_night = xr.concat([lst_gap_filled_night,lst_masked_day.loc[uncommon_datetimes]],dim='time')
            else:
                lst_gap_filled_night = xr.concat([lst_gap_filled_night,skin_n.loc[uncommon_datetimes]],dim='time')
                print('Check missing day data for {}-{:02d}'.format(year,month))


        lst_GF_avg = xr.concat([lst_gap_filled_day.drop('band'),lst_gap_filled_night.drop('band'),lst_masked_day_mod,lst_masked_night_mod],dim='time').mean(dim='time')

        lst_GF_avg.rio.to_raster(r'\multi_elev_lat\{}_{:02d}.tif'.format(year,month))

        del lst_gap_filled_day,lst_gap_filled_night,lst_day,lst_night,qc_day,qc_night
        gc.collect()




for year in range(2003,2017):
    month_files = sorted(glob.glob(r'\\multi_elev_lat\{}*.tif'.format(year)))
    da = xr.concat([xr.open_dataarray(file) for file in month_files],dim='time').mean(dim='time')
    da.rio.to_raster(r'\multi_elev_lat\MAST\{}.tif'.format(year))

all_arrays = sorted(glob.glob(r'\multi_elev_lat\*.tif'))
datetimes = pd.date_range('2003-01','2017-01',freq='M')
dataset = xr.concat([xr.open_rasterio(array)[0] for array in all_arrays], dim='time')
dataset['time']=datetimes
dataset = dataset.rename('MLST')
dataset.to_netcdf(r'\MODMYD11_AIRS_monthly.nc')
dataset.mean(dim='time').rio.to_raster(r'\2003_2016_MODMYD11AIRS.tif')
