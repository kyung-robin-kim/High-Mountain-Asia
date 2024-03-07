
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import glob,os
import pandas as pd
import numpy as np 
from scipy import stats
from scipy.stats import gaussian_kde



def read_file(file):
    with rio.open(file) as src:
        return(src.read())
    
def plot_np(array,vmin,vmax,title):
    array = np.where(array==0,np.nan,array)
    fig1, ax1 = plt.subplots(figsize=(20,16))
    image = ax1.imshow(array,cmap = 'Blues',vmin=vmin,vmax=vmax)
    cbar = fig1.colorbar(image,ax=ax1)
    ax1.set_title('{}'.format(title))


def linear_stats(model,insitu,title):
    try:
        nanmask = ~np.isnan(model) & ~np.isnan(insitu)
        model = model[nanmask]
        insitu = insitu[nanmask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(insitu,model)
        modeled_y = intercept + slope*insitu

        rmse = np.sqrt(np.mean((modeled_y - insitu)**2))
        bias = np.mean(modeled_y - insitu)
        ub_rmse = np.sqrt(np.sum((modeled_y - insitu)**2)/(len(insitu)-1))
        std_err

        metrics = pd.DataFrame([round(r_value,3),round(rmse,4),round(bias,4),round(p_value,4),title]).rename({0:'R',1:'rmse',2:'bias',3:'pval',4:'label'}).T

        return metrics
    
    except ValueError:
        print('error {}'.format(title))

plt.rcParams["font.family"] = "Times New Roman"
def linear_plot(model,insitu,title):
    try:
        nanmask = ~np.isnan(model) & ~np.isnan(insitu)
        model = model[nanmask]
        insitu = insitu[nanmask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(insitu,model)
        modeled_y = intercept + slope*insitu

        rmse = np.sqrt(np.mean((modeled_y - insitu)**2))
        bias = np.mean(modeled_y - insitu)
        ub_rmse = np.sqrt(np.sum((modeled_y - insitu)**2)/(len(insitu)-1))
        std_err

        # Calculate the point density
        xy = np.vstack([insitu,model])
        z = gaussian_kde(xy)(xy)

        fig,ax = plt.subplots()
        ax.scatter(insitu, model, c=z, s=1)
        ax.plot(insitu, intercept + slope*insitu, label='r: {}'.format(round(r_value,3)),color='r')
        ax.set_ylabel('Modeled Snow Depth (m)')
        ax.set_xlabel('In situ Snow Depth (m)')
        ax.legend()
        ax.set_ylim(0,3)
        ax.set_xlim(0,3)
        ax.plot([-1,3],[-1,3],'--',color='black')
        ax.text(2.1,0.80,s="count: {}".format(len(insitu)), fontsize=12, ha="left", va="top")
        ax.text(2.1,0.65,s="p-val < 0.001".format(round(p_value,4)), fontsize=12, ha="left", va="top")
        #ax.text(2.1,0.65,s="p-val: {:.3f}".format(round(p_value,4)), fontsize=12, ha="left", va="top")
        ax.text(2.1,.50,s="RMSE: {:.3f}".format(round(rmse,4)), fontsize=12, ha="left", va="top")
        ax.text(2.1,0.35,s="bias: {:.3f}".format(round(bias,4)), fontsize=12, ha="left", va="top")
        ax.set_title(title,weight='bold')

        print(title,round(r_value,3))

    except ValueError:
        print('error {}'.format(title))







######################################
#CALL INSITU DATASETS (SD) ~12 minutes
######################################

import geopandas as gpd
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np


# (Yakou only) for snow
path= r'\Frozen_Ground\frozen_ground_obs\Yakou_superstation'
xls_files_SD = sorted(glob.glob(path+'\*SnowDepth*.xlsx'))[0]
yakou_SD_data = pd.read_excel(xls_files_SD)
yakou_sd_daily = yakou_SD_data.set_index(pd.to_datetime(yakou_SD_data.DATE,'%Y-%m-d')).resample('1D').mean()



#ICIMOD - MAGST, SD (12 seconds)
path = r'\HMA\ICIMOD'
csv_files = sorted(glob.glob(path+'\\*\\data\\*.csv'))[:-2]
csv_data = [pd.read_csv(csv) for csv in csv_files]
icimod_sd_points = pd.read_csv(r'\HMA\ICIMOD\stations.csv')
icimod_sd_points = gpd.GeoDataFrame(icimod_sd_points, geometry=gpd.points_from_xy(icimod_sd_points.X, icimod_sd_points.Y))[1:]

icimod_sd_data = [csv.set_index(pd.to_datetime(csv.Date, format='%Y-%m-%d')) for csv in csv_data]
icimod_sd_data = [data.resample('1D').mean() for data in icimod_sd_data]

#precip =  [data.filter(like='Precipitation TB(mm)').iloc[:,0] for data in icimod_sd_data[0:2]]
icimod_sd_depth_meters = [data.filter(like='Snow Depth').iloc[:,0] for data in icimod_sd_data] #meters
icimod_sd_quality = [data.filter(like='Quality').iloc[:,0] for data in icimod_sd_data]
icimod_sd_at = [data.filter(like='Air Temperature').iloc[:,0] for data in icimod_sd_data]
icimod_sd_gt = [data.filter(like='Ground').iloc[:,0] for data in icimod_sd_data]



# NOAA GHCN - MAAT (5 seconds)
csv_files = sorted(glob.glob( r'\NOAA_GHCN\*.csv'))
ghcn_points = pd.read_csv(r'\NOAA_GHCN\validation_points.csv').sort_values(by='Name')
ghcn_data = [pd.read_csv(file) for id in ghcn_points['Station_ID'] for file in csv_files if file.find(id) > 0]

snow_true_index = [(site=='SNWD').any() for site in [data.keys() for data in ghcn_data]]
ghcn_snow = [ghcn_data[i].set_index(pd.to_datetime(ghcn_data[i].DATE)).SNWD/1000 for i in range(0,len(ghcn_data)) if snow_true_index[i]==True]
ghcn_snow_stations = [ghcn_data[i].STATION[0] for i in range(0,len(ghcn_data)) if snow_true_index[i]==True]

uncommon_stations = set(ghcn_snow_stations) ^ set(ghcn_points.Station_ID)
uncommon_stations = [[name,id] for name,id in zip(ghcn_points.Station_ID,ghcn_points.Name) if name in list(uncommon_stations)] 
uncommon_names =  [station[1] for station in uncommon_stations]

common_stations = [[name,id] for name,id in zip(ghcn_points.Station_ID,ghcn_points.Name) if name in ghcn_snow_stations] 
common_names = [station[1] for station in common_stations]


#Wani - SD at site NIH001/2
sd_wani = pd.read_csv(r'\Wani_2020\SnowDepth_Wani.csv',header=None)
julian_days = round(sd_wani.iloc[:,0])
depth_cm = round(sd_wani.iloc[:,1]*100)

import datetime #ChatGPT
def datetime_array_from_julian_day(start_date, julian_days):
    start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    datetime_array = []
    
    for julian_day in julian_days:
        delta = datetime.timedelta(days=julian_day - 1)
        date = start_datetime + delta
        datetime_array.append(date)
    
    return datetime_array

start_date = "2015-09-15"
dates = datetime_array_from_julian_day(start_date, julian_days)
wani_sd = pd.DataFrame([dates,depth_cm]).T.rename(columns={0:'dates',1:'sd_cm'}).set_index('dates')

##################################################################################################
#Monthly resample insitu datasets
hiwat_monthly_meter = yakou_sd_daily.resample('1M').mean().dropna().iloc[:,1]
icimod_monthly_meter = [(ds.resample('1M').mean()).dropna() for ds in icimod_sd_depth_meters]
ghcn_monthly_meter = [(ds.resample('1M').mean()).dropna() for ds in ghcn_snow]
wani_monthly_meter = (wani_sd.resample('1M').mean()/100).dropna().iloc[:,0]
insitu_sd = [[hiwat_monthly_meter],icimod_monthly_meter,ghcn_monthly_meter,[wani_monthly_meter]]
set_names = ['HiWAT (Yakou)','ICIMOD','NOAA GHCN','Wani 2020']
source_names = ['HiWAT (Yakou)','ICIMOD','NOAA GHCN','Wani 2020']


#Mask out negative values
icimod_monthly_meter = [icimod_monthly_meter[i][icimod_monthly_meter[i]>0] for i in range(0,len(icimod_monthly_meter))]


main_500m = r'\UCLA_SnowDepth\validation'
paths_500m = sorted(os.listdir(main_500m))
set_paths_500m = [sorted(glob.glob(main_500m+'\\'+path+'\*.csv')) for path in paths_500m]
set_snow = [[pd.read_csv(file) for file in set_path] for set_path in set_paths_500m]
[[set_snow[c][i].set_index(pd.to_datetime(set_snow[c][i].iloc[:,0]),inplace=True) for i in range(0,len(set_snow[c]))] for c in range(0,len(set_snow))]

#For validations, extracted daily model SD is in centimeters (to conserve memory when saving (int16))
#When processing, monthly model SD is in meters (before I knew better in June, 2021)
set_snow_mean = [[set.iloc[:,1].resample('1M').mean()/100 for set in sets] for sets in set_snow]
set_snow_stdev = [[set.iloc[:,2].resample('1M').mean()/100 for set in sets] for sets in set_snow]
###########################################################################################



###########################################################################################
#Annual resample insitu datasets
hiwat_monthly_meter = yakou_sd_daily.resample('1Y').mean().dropna().iloc[:,0]
icimod_monthly_meter = [(ds.resample('1Y').mean()).dropna() for ds in icimod_sd_depth_meters]
ghcn_monthly_meter = [(ds.resample('1Y').mean()/100).dropna() for ds in ghcn_snow]
wani_monthly_meter = (wani_sd.resample('1Y').mean()/100).dropna().iloc[:,0]
insitu_sd = [[hiwat_monthly_meter],icimod_monthly_meter,ghcn_monthly_meter,[wani_monthly_meter]]
set_names = ['HiWAT (Yakou)','ICIMOD','NOAA GHCN','Wani 2020']
source_names = ['HiWAT (Yakou)','ICIMOD','NOAA GHCN','Wani 2020']


main_500m = r'\UCLA_SnowDepth\validation'
paths_500m = sorted(os.listdir(main_500m))
set_paths_500m = [sorted(glob.glob(main_500m+'\\'+path+'\*.csv')) for path in paths_500m]
set_snow = [[pd.read_csv(file) for file in set_path] for set_path in set_paths_500m]
[[set_snow[c][i].set_index(pd.to_datetime(set_snow[c][i].iloc[:,0]),inplace=True) for i in range(0,len(set_snow[c]))] for c in range(0,len(set_snow))]

#For validations, extracted daily model SD is in centimeters (to conserve memory when saving (int16))
#When processing, monthly model SD is in meters (before I knew better in June, 2021)
set_snow_mean = [[set.iloc[:,1].resample('1Y').mean()/100 for set in sets] for sets in set_snow]
set_snow_stdev = [[set.iloc[:,2].resample('1Y').mean()/100 for set in sets] for sets in set_snow]
###########################################################################################



merged_ids = [[idx1.index.intersection(idx2.index) for idx1,idx2 in zip(insitu_set,model_set)] for insitu_set,model_set in zip(insitu_sd,set_snow_mean)]
all_metrics = []
all_data = []

for insitu,snow_set,merged_id_set,set_name in zip(insitu_sd,set_snow_mean,merged_ids,set_names):

    set_all_metrics = []
    set_all_data = []

    for i in range(0,len(insitu)):

        if len(merged_id_set[i])!=0:
            try:
                model_set = (snow_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([i for i in range(1,13)])]]) #all
                insitu_set = (insitu[i].loc[merged_id_set[i][merged_id_set[i].month.isin([i for i in range(1,13)])]]) #all
                set_all_metrics.append(linear_stats(model_set,insitu_set,set_name+'monthly snow all'))
                set_all_data.append(pd.DataFrame([model_set,insitu_set]).T.rename(columns={0:'model',1:'insitu'}))

            except IndexError or ValueError:
                continue
    
    #Metrics
    set_all_metrics = pd.concat(set_all_metrics,axis=0).reset_index().iloc[:,1:]
    all_metrics.append(set_all_metrics)

    #Data
    set_all_data = pd.concat(set_all_data,axis=0)
    all_data.append(set_all_data)



#ALL DATA
import itertools
plt.rcParams["font.family"] = "Times New Roman"
degree_sign = u"\N{DEGREE SIGN}"
plt.rcParams.update({'font.size': 15})

linear_plot(np.array(list(itertools.chain.from_iterable([(all_data[i].iloc[:,0]) for i in range(0,len(all_data))]))),
            np.array(list(itertools.chain.from_iterable([all_data[i].iloc[:,1] for i in range(0,len(all_data))]))), 'Liu et al. (2021)')


titles = ['HiWAT','ICIMOD','GHCN','Wani']
i=2
linear_plot(all_data[i].iloc[:,0],all_data[i].iloc[:,1],titles[i])

for i in range(0,len(source_names)):
    for ii in range(0,len(set_names)):
        linear_plot(all_data[i][ii].modis,all_data[i][ii].insitu,source_names[i] + ' ' + set_names[ii])

for i in range(0,len(source_names)):
    linear_plot(all_data[i].iloc[:,0],all_data[i].iloc[:,1], source_names[i] + ' All Year')
