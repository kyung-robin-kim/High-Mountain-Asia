#########################################
#For Validation of LST products for SET C
#########################################
#Daily MOD11, MYD11, Zhang GF MYD11

#There are multiple variables to assess: MAAT (air), MAGST (ground surface & active layer), MAGT (DZAA), SM (soil moisture), SD (snow depth)
#There are multiple validation datasets:
#1. NOAA GHCN - MAAT [snow]
#2. ISMN - SM, MAGST
#3. HiWAT - SM, MAGST, MAAT [snow]
#4. GNT-P (others) & Wani - MAGT
#5. Zhao 2021 - SM, MAGT (x6 for MAGST and MAAT too)
#6. Ma 2020 - SM, MAGST
#7. ICIMOD - MAGST [snow]

import xarray as xr
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import itertools
import geopandas as gpd
from scipy.stats import gaussian_kde
import os
import itertools
import os

degree_sign = u"\N{DEGREE SIGN}"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})

def linear_plot(model,insitu,title,savepath):
    try:
        nanmask = ~np.isnan(model) & ~np.isnan(insitu)
        model = model[nanmask]
        insitu = insitu[nanmask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(insitu,model)
        modeled_y = intercept + slope*insitu

        rmse = np.sqrt(np.mean((modeled_y - insitu)**2))
        bias = np.mean(modeled_y - insitu)
        std_err

        # Calculate the point density
        xy = np.vstack([insitu,model])
        z = gaussian_kde(xy)(xy)

        fig,ax = plt.subplots()
        ax.scatter(insitu, model, c=z, s=1)
        ax.plot(insitu, intercept + slope*insitu, label='r: {:.3f}'.format(round(r_value,3)),color='r')
        ax.set_ylabel('Remotely Sensed ST {}C'.format(degree_sign))
        ax.set_xlabel('In situ {} ({}C)'.format(savepath,degree_sign))
        ax.legend()
        ax.set_ylim(-60,60)
        ax.set_xlim(-60,60)
        ax.plot([-60,60],[-60,60],'--',color='black')
        ax.text(25,-15,s="count: {}".format(len(insitu)), fontsize=12, ha="left", va="top")
        ax.text(25,-20,s="p-val < 0.001", fontsize=12, ha="left", va="top")
        ax.text(25,-25,s="RMSE: {:.3f}".format(round(rmse,3)), fontsize=12, ha="left", va="top")
        ax.text(25,-30,s="bias: {:.3f}".format(round(bias,3)), fontsize=12, ha="left", va="top")
        ax.set_title(title,weight='bold')
        
        ax.set_title(title)
        plt.savefig(r'\{}\{}.png'.format(savepath,title))
        plt.close()

        print(title,round(r_value,3))

    except ValueError:
        print('error {}'.format(title))


def linear_stats(model,insitu,title):
    try:
        nanmask = ~np.isnan(model) & ~np.isnan(insitu)
        model = model[nanmask]
        insitu = insitu[nanmask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(insitu,model)
        modeled_y = intercept + slope*insitu

        rmse = np.sqrt(np.mean((modeled_y - insitu)**2))
        bias = np.mean(modeled_y - insitu)
        std_err

        metrics = pd.DataFrame([round(r_value,3),round(rmse,4),round(bias,4),round(p_value,4),title]).rename({0:'R',1:'rmse',2:'bias',3:'pval',4:'label'}).T

        return metrics
    
    except ValueError:
        print('error {}'.format(title))



######################################
#CALL INSITU DATASETS (LST & AT)
######################################

# HiWAT - SM, MAGST, MAAT (4 minutes)
path= r'\frozen_ground_obs'
xls_files_AWS = sorted(glob.glob(path+'\*\*AWS.xlsx'))

hiwat_stations = pd.read_excel(r'\Station location.xlsx')
xls_files_AWS_station = [[file for file in xls_files_AWS if file.find(id) > 0] for id in sorted(hiwat_stations.station)]

AWS_data = [[pd.read_excel(file).replace(-6999,np.nan) for file in xls_files] for xls_files in xls_files_AWS_station] #6 minutes to run
AWS_data_daily = [[data.set_index(pd.to_datetime(data.TIMESTAMP,'%Y-%m-d')).resample('1D').mean() for data in dataset] for dataset in AWS_data]
AWS_data_stdev = [[data.set_index(pd.to_datetime(data.TIMESTAMP,'%Y-%m-d')).resample('1D').std(ddof=1) for data in dataset] for dataset in AWS_data]

hiwat_air_temp_annual = [pd.concat([data.filter(like='Ta_') for data in dataset],axis=0) for dataset in AWS_data_daily] #2014,2017: 2m; 2015,2016: 5m ---- ##5m based on meta
hiwat_soil_temp_annual = [pd.concat([data.filter(like='Ts_') for data in dataset],axis=0) for dataset in AWS_data_daily] #2014-2017: 0,4,10,20,40,80,120cm



#ICIMOD - MAGST, SD
path = r'\ICIMOD'
xcel_file = sorted(glob.glob(path+'\\*\\data\\*.xlsx'))[1]
xcel_sheets = pd.ExcelFile(xcel_file).sheet_names
xcel_data = [pd.read_excel(xcel_file,sheet_name=sheet) for sheet in xcel_sheets]
icimod_gst_points =  xcel_data[0].iloc[:,1:6]
icimod_gst_points = gpd.GeoDataFrame(icimod_gst_points, geometry=gpd.points_from_xy(icimod_gst_points.Lon, icimod_gst_points.Lat)).sort_values(by='NAME')

daily_gst = [data.set_index('TIME').resample('1D').mean() for data in xcel_data[1:]]
icimod_daily_gst = pd.concat([daily_gst[0],daily_gst[1],daily_gst[2],daily_gst[3]])
icimod_daily_gst = icimod_daily_gst.rename(columns={'SN2081':'PF14','SN2543':'PF13','SN2142':'PF12','SN2111':'PF11'})
icimod_daily_gst = icimod_daily_gst[sorted(list(icimod_daily_gst.columns.values))]

csv_files = sorted(glob.glob(path+'\\*\\data\\*.csv'))[:-2]
csv_data = [pd.read_csv(csv) for csv in csv_files]
icimod_sd_points = pd.read_csv(r'\stations.csv')
icimod_sd_points = gpd.GeoDataFrame(icimod_sd_points, geometry=gpd.points_from_xy(icimod_sd_points.X, icimod_sd_points.Y))[1:]

icimod_sd_data = [csv.set_index(pd.to_datetime(csv.Date, format='%Y-%m-%d')) for csv in csv_data]
icimod_sd_data = [data.iloc[:,2:].resample('1D').mean() for data in icimod_sd_data]

icimod_sd_depth = [data.filter(like='Snow Depth').iloc[:,0] for data in icimod_sd_data]
icimod_sd_quality = [data.filter(like='Quality').iloc[:,0] for data in icimod_sd_data]
icimod_sd_at = [data.filter(like='Air Temperature').iloc[:,0] for data in icimod_sd_data]
icimod_sd_gt = [data.filter(like='Ground').iloc[:,0] for data in icimod_sd_data]


# ISMN - SM, MAGST 
ismn_dir = r'\ISMN'
ismn_points = [pd.read_csv(file) for file in sorted(glob.glob(ismn_dir+'\points\*.csv'))]
ts_file_names = [sorted(glob.glob(ismn_dir+r'\TEMP\*_{}cm*.csv'.format(cm))) for cm in ['05','20','40','80']] 
sm_file_names = [sorted(glob.glob(ismn_dir+r'\MOIST\*_{}cm*.csv'.format(cm))) for cm in ['05','20','40','80']]

ts_cm_data = [[pd.read_csv(file).set_index(pd.to_datetime(pd.read_csv(file).iloc[:,0])).iloc[:,6] for file in cm] for cm in ts_file_names] 

magsts = [[csv.resample('1Y').mean().loc[pd.to_datetime('2003-12-31'):pd.to_datetime('2016-12-31')] for csv in cm] for cm in ts_cm_data]
stdevs = [[csv.resample('1Y').std(ddof=1).loc[pd.to_datetime('2003-12-31'):pd.to_datetime('2016-12-31')] for csv in cm] for cm in ts_cm_data]
counts = [[csv.resample('1Y').count().loc[pd.to_datetime('2003-12-31'):pd.to_datetime('2016-12-31')] for  csv in cm] for cm in ts_cm_data]
ismn_dicts_ts = [pd.DataFrame([{'in_situ_MAGST':magst,'in_situ_MAAT_std':stdev,'in_situ_count':count} for magst,stdev,count in zip(magsts_cm,stdevs_cm,counts_cm)])
        for magsts_cm,stdevs_cm,counts_cm in zip(magsts,stdevs,counts)]



#Ma 2020 - GST,AT, SM 
mainpath = r'\Ma_ESSD'
ma_2020_stations = sorted(os.listdir(r'\Ma_ESSD')[1:-2])
soil_files = [sorted(glob.glob(mainpath+'\{}\SOIL*\*20*.csv'.format(station))) for station in ma_2020_stations]
soil_data = [[pd.read_csv(i) for i in files] for files in soil_files]
soil_data = [[i.set_index(pd.to_datetime(i.Timestamp)) for i in data] for data in soil_data]
ma_soil_temps = [pd.concat([i.filter(like='Ts').astype(float) for i in data],axis=0) for data in soil_data] 

air_files = [sorted(glob.glob(mainpath+'\{}\GRAD*\*20*.csv'.format(station))) for station in ma_2020_stations]
air_data = [[pd.read_csv(i) for i in files] for files in air_files]
air_data = [[i.set_index(pd.to_datetime(i.Timestamp)) for i in data] for data in air_data]
ma_air_temps = [pd.concat([i.filter(like='Ta').astype(float) for i in data],axis=0) for data in air_data] 



# NOAA GHCN - MAAT 
csv_files = sorted(glob.glob( r'\NOAA_GHCN\*.csv'))
ghcn_points = pd.read_csv(r'\NOAA_GHCN\validation_points.csv').sort_values(by='Name')
ghcn_data = [pd.read_csv(file) for id in ghcn_points['Station_ID'] for file in csv_files if file.find(id) > 0]
ghcn_data_indexed = [(csv.set_index(pd.to_datetime(csv.DATE))).TAVG/10 for csv in ghcn_data]

maats = [csv.resample('1Y').mean().loc[pd.to_datetime('2003-12-31'):pd.to_datetime('2016-12-31')].mean() for csv in ghcn_data_indexed]
stdevs = [csv.resample('1Y').mean().loc[pd.to_datetime('2003-12-31'):pd.to_datetime('2016-12-31')].std(ddof=1) for csv in ghcn_data_indexed]
counts = [csv.resample('1Y').mean().loc[pd.to_datetime('2003-12-31'):pd.to_datetime('2016-12-31')].count() for  csv in ghcn_data_indexed]
dicts = pd.DataFrame([{'in_situ_MAAT':maat,'in_situ_MAAT_std':stdev,'in_situ_count':count} for maat,stdev,count in zip(maats,stdevs,counts)])

ghcn_maat = pd.concat([pd.DataFrame(ghcn_points.iloc[:,0:5]),dicts],axis=1)
ghcn_maat = ghcn_maat[ghcn_maat.in_situ_count > 2]



#Zhao_2021 -- MAGT, SM, MAGST
path= r'\Zhao_2021_ALL'
xls_files = sorted(glob.glob(path+'\*.xlsx'))[:-1]
xls_sheets = [pd.ExcelFile(file).sheet_names for file in xls_files]
xls_data = [pd.read_excel(xls_file,sheet_name=sheet) for sheet,xls_file in zip(xls_sheets,xls_files)]
zhao2021_stations = pd.read_csv(r'\stations.csv')
val_points = gpd.GeoDataFrame(zhao2021_stations, columns=['Site','Longitude', 'Latitude'], geometry=gpd.points_from_xy(zhao2021_stations.Longitude,zhao2021_stations.Latitude)).set_crs('epsg:4326').drop([32,63])

AL_temp, AL_mois, GTemp, AWS = xls_data[0], xls_data[1], xls_data[2], xls_data[3]
in_situ = [AL_temp, AL_mois, GTemp, AWS]
in_situ_df = [[df['{}'.format(site)] for site in sorted(df.keys())] for df in in_situ]

#Reindex AL Temp & Moist to datetime:
for ii in range(0,2):
    for i in range(0,len(in_situ_df[ii])):
        in_situ_df[ii][i]['DateIndex'] = in_situ_df[ii][i]['Year'].astype(str) + '-' + in_situ_df[ii][i]['Month'].astype(str) + '-' + in_situ_df[ii][i]['Day'].astype(str)
        in_situ_df[ii][i]['DateIndex'] = pd.to_datetime(in_situ_df[ii][i]['DateIndex'], format='%Y-%m-%d')
        in_situ_df[ii][i] = in_situ_df[ii][i].set_index('DateIndex').replace(-6999,np.nan)
        in_situ_df[ii][i] = in_situ_df[ii][i].drop(columns=['Year','Month','Day'])
#Reindex GT to datetime:
for i in range(0,len(in_situ_df[2])):
    in_situ_df[2][i] = in_situ_df[2][i].set_index(pd.to_datetime(in_situ_df[2][i]['TS'], format='%Y'))
    in_situ_df[2][i] = in_situ_df[2][i].drop(columns=['TS'])
#Re-list per site for GT observations:
in_situ_df[2][0] = [[in_situ_df[2][0][in_situ_df[2][0].Id_site == i]] for i in GTemp['Automatic_observation'].Id_site.unique()]
in_situ_df[2][0] = [i[0] for i in in_situ_df[2][0]]

in_situ_df[2][1] =  [[in_situ_df[2][1][in_situ_df[2][1].Id_site == i]]  for i in GTemp['Manual_observation'].Id_site.unique()]
in_situ_df[2][1] = [i[0] for i in in_situ_df[2][1]]
#Reindex AWS to datetime:
for i in range(0,len(in_situ_df[3])):
    in_situ_df[3][i] = in_situ_df[3][i].drop(index=[0,1])
    in_situ_df[3][i] = in_situ_df[3][i].set_index(pd.to_datetime(in_situ_df[3][i]['DateTime'], format='%Y-%m-%d'))

aws_stations = sorted(xls_sheets[3])
aws_data = [i.iloc[2:,:].set_index(pd.to_datetime(i.iloc[2:,0])) for i in in_situ_df[3]]
zhao_Ta = [aws.filter(like='Ta') for aws in aws_data]

AL_datetimes = [pd.to_datetime(AL_temp['{}'.format(key)].iloc[:,0:3]) for key in sorted(AL_temp.keys())]
AL_temps_surface = [AL_temp['{}'.format(key)].set_index(datetimes).resample('1D').mean().iloc[:,3:] for key,datetimes in zip(sorted(AL_temp.keys()),AL_datetimes)]
sorted(AL_temp.keys())
       
AYK_GT = GTemp['Automatic_observation'][GTemp['Automatic_observation'].Id_site.str.contains("AYK")]
Ch04_GT = GTemp['Automatic_observation'][GTemp['Automatic_observation'].Id_site.str.contains("QTB18")] #LDH
QT04_GT = GTemp['Manual_observation'][GTemp['Manual_observation'].Id_site.str.contains("TGL")]
TSH_GT = GTemp['Automatic_observation'][GTemp['Automatic_observation'].Id_site.str.contains("TSH")]
QT09_GT = GTemp['Automatic_observation'][GTemp['Automatic_observation'].Id_site.str.contains("XDT")]
ZNH_GT = GTemp['Automatic_observation'][GTemp['Automatic_observation'].Id_site.str.contains("ZNH")]

['AYK', 'LDH', 'TGL', 'TSH', 'XDT', 'ZNH']
zhao_GT_borehole = [i.iloc[:,2:].set_index(pd.to_datetime(i.iloc[:,1],format='%Y')).resample('1Y').mean() for i in [AYK_GT,Ch04_GT,QT04_GT,TSH_GT,QT09_GT,ZNH_GT]]
zhao_GT_AL = [AL_temps_surface[0],AL_temps_surface[2],AL_temps_surface[6],AL_temps_surface[10],AL_temps_surface[9],AL_temps_surface[-1]]
zhao_insitu = in_situ_df




######################################
#Remotely Sensed LST
######################################

#Check different depths
hiwat_soil_temp_annual[0].keys()
ma_soil_temps[0].keys()
len(ts_cm_data) #5, 20, 40, 80
zhao_insitu[0][0].keys()

hiwat_air_temp_annual[0].keys()
icimod_sd_data[3].keys()
ma_air_temps[0].keys()
zhao_Ta[0].keys()

#HiWAT
stations = pd.read_excel(r'\Station location.xlsx')
hiwat_points = gpd.GeoDataFrame(stations, columns=['station','Longtitude', 'Latitude'], geometry=gpd.points_from_xy(stations.Longtitude,stations.Latitude)).set_crs('epsg:4326')
#ICIMOD
path = r'\ICIMOD'
xcel_file = sorted(glob.glob(path+'\\*\\data\\*.xlsx'))[1]
xcel_sheets = pd.ExcelFile(xcel_file).sheet_names
xcel_data = [pd.read_excel(xcel_file,sheet_name=sheet) for sheet in xcel_sheets]
icimod_points =  xcel_data[0].iloc[:,1:6]
#ICIMOD Snow
path = r'\ICIMOD'
icimod_snow_points = pd.read_csv(sorted(glob.glob(path+'\\*.csv'))[0]).loc[1:]
#ISMN (only 5cm or 80 cm (20 & 40 cm not included))
ismn_points = pd.read_csv(r'\ISMN\05_cm_ts.csv')
#Ma 2020
ma_points = pd.read_csv(r'\Ma_ESSD\stations.csv')
#NOAA GHCN
csv_files = sorted(glob.glob(r'\NOAA_GHCN\*.csv'))[:-1]
ghcn_points = pd.read_csv(r'\\NOAA_GHCN\\validation_points.csv')
#Zhao2021
stations = pd.read_csv(r'\Zhao_2021_ALL\stations.csv')
zhao2021_points = gpd.GeoDataFrame(stations, columns=['Site','Longitude', 'Latitude'], geometry=gpd.points_from_xy(stations.Longitude,stations.Latitude)).set_crs('epsg:4326').drop([32,63])
zhao2021_points=zhao2021_points.iloc[sorted([1,10,57,90,38,35,43,71,34,64,37,23])]


hiwat_gst = [data.dropna().resample('1D').mean().iloc[:,0] for data in hiwat_soil_temp_annual]
icimod_gst_I = [icimod_daily_gst.iloc[:,i].dropna().resample('1D').mean() for i in range(0,len(icimod_daily_gst.keys()))]
icimod_gst_II = [icimod_sd_data[i].filter(like='Ground').dropna().resample('1D').mean() for i in range(0,len(icimod_sd_data))]
[icimod_gst_I.append(icimod_gst_II[i].iloc[:,0]) for i in range(0,len(icimod_gst_II))]
ismn_gst = [data.dropna().resample('1D').mean() for data in ts_cm_data[0]]
ma_gst = [df.iloc[:,0].dropna().resample('1D').mean() for df in ma_soil_temps] 
zhao_gst = [df.iloc[:,0].dropna().resample('1D').mean() for df in zhao_insitu[0]]


hiwat_at = [data.dropna().iloc[:,0].resample('1D').mean() for data in hiwat_air_temp_annual]
icimod_at = [data.iloc[:,0].dropna().resample('1D').mean() for data in icimod_sd_data]
ghcn_at = [data.dropna().resample('1D').mean() for data in ghcn_data_indexed]
ma_at = [data.iloc[:,0].dropna().resample('1D').mean() for data in ma_air_temps]
zhao_at = [df.iloc[:,0].dropna().resample('1D').mean().astype(float) for df in zhao_Ta]

insitu_gst = [hiwat_gst,icimod_gst_I,ismn_gst,ma_gst,zhao_gst]
insitu_at = [hiwat_at,icimod_at,ma_at,ghcn_at,zhao_at]

gst_names = sorted(['HiWAT','ICIMOD','ISMN','Ma et al. (2020)','Zhao et al. (2021)'])
at_names = sorted(['HiWAT','ICIMOD','NOAA GHCN','Ma et al. (2020)','Zhao et al. (2021)'])

GST = r'\GST'
AT = r'\AT'


#################################################################################
#Set C - AIRS Gap Filling

################################
#Choose GST
insitu_temps = insitu_gst
set_names = gst_names
temp = GST
savepath = 'GST'
keys = [sorted(hiwat_points.station),sorted(pd.concat([icimod_points.NAME,icimod_snow_points.Station])),sorted(ismn_points.site),sorted(ma_points.Site),sorted(zhao2021_points.Site)[-1]]

main_1km = r'\DAILY\validation\{}'.format(temp)
paths_1km = sorted(os.listdir(main_1km))
set_paths_1km = [sorted(glob.glob(main_1km+'\\'+path+'\*.csv')) for path in paths_1km]
set_paths_1km_grouped =  [[pd.concat([pd.read_csv(file) for file in set_paths if file.find(id) > 0]) for id in key] for key,set_paths in zip(keys,set_paths_1km)] 
setC_AIRS = [[data.iloc[:,1:].set_index(pd.to_datetime(data.iloc[:,0])).resample('1D').mean() for data in set_paths] for set_paths in set_paths_1km_grouped]
setC_AIRS_daily = [[data.mean(axis=1) for data in sets] for sets in setC_AIRS]



#Choose AT
insitu_temps = insitu_at
set_names = at_names
temp = AT
savepath = 'AT'
keys = [hiwat_points.station,icimod_snow_points.Station,ma_points.Site,ghcn_points.Name,['AYK','LDH','TGL','TSH','XDT']]
keys = [sorted(key) for key in keys]


airs_files = [[sorted(glob.glob(r'\DAILY\validation\AT\{}\*{}*.csv'.format(source,station))) 
              for station in key] for source,key in zip(['HiWAT','ICIMOD','Ma2020','NOAA_GHCN','Zhao2021'],keys)]
airs_st = [[pd.concat([pd.read_csv(year).set_index('Unnamed: 0').rename(columns={'am':'AIRS DS DAY','pm':'AIRS DS NIGHT'}) 
                      for year in station],axis=0) for station in source] for source in airs_files]
[[setattr(station, 'index', pd.to_datetime(station.index)) for station in source] for source in airs_st]
setC_AIRS_daily = [[data.mean(axis=1) for data in sets] for sets in airs_st]
################################


CCs_all = []
rmses_all = []
bias_all = []
pvals_all = []

CCs_monsoon = []
rmses_monsoon = []
bias_monsoon = []
pvals_monsoon = []

CCs_non = []
rmses_non = []
bias_non = []
pvals_non = []

#Call new lists for datasets (not necessary for AIRS downscaled set b/c only one)
all_data_concat = []
monsoon_data_concat = []
dry_data_concat = []


# Loop through metrics and datasets for plotting later
all_metrics = []
monsoon_metrics = []
dry_metrics = []

all_data = []
monsoon_data = []
dry_data = []

for insitus,sets,set_name in zip(insitu_temps,setC_AIRS_daily,set_names):

    merged_ids = [idx1.index.intersection(idx2.index) for idx1,idx2 in zip(insitus,sets)]
    set_all_metrics = []
    set_monsoon_metrics = []
    set_dry_metrics = []
    
    set_all_data = []
    set_monsoon_data = []
    set_dry_data = []

    for i in range(0,len(insitus)):

        try:
            model_set = (sets[i].loc[merged_ids[i]].ravel()) #all
            model_monsoon_set = (sets[i].loc[merged_ids[i][merged_ids[i].month.isin([6,7,8])]].ravel()) #monsoon
            model_dry_set = (sets[i].loc[merged_ids[i][merged_ids[i].month.isin([11,12,1,2])]].ravel()) #dry

            insitu_set = (insitus[i].loc[merged_ids[i]].ravel()) #all
            insitu_monsoon_set = (insitus[i].loc[merged_ids[i][merged_ids[i].month.isin([6,7,8])]].ravel()) #monsoon
            insitu_dry_set = (insitus[i].loc[merged_ids[i][merged_ids[i].month.isin([11,12,1,2])]].ravel()) #dry

            set_all_metrics.append(linear_stats(model_set,insitu_set,set_name+' AIRS all'))
            set_monsoon_metrics.append(linear_stats(model_monsoon_set,insitu_monsoon_set,set_name+' AIRS monsoon'))
            set_dry_metrics.append(linear_stats(model_dry_set,insitu_dry_set,set_name+' AIRS dry'))

            set_all_data.append(pd.DataFrame([model_set.ravel(),insitu_set.ravel()]).T.rename(columns={0:'model',1:'insitu'}))
            set_monsoon_data.append(pd.DataFrame([model_monsoon_set.ravel(),insitu_monsoon_set.ravel()]).T.rename(columns={0:'model',1:'insitu'}))
            set_dry_data.append(pd.DataFrame([model_dry_set.ravel(),insitu_dry_set.ravel()]).T.rename(columns={0:'model',1:'insitu'}))

        except IndexError or ValueError:
            continue

    #Metrics
    set_all_metrics = pd.concat(set_all_metrics,axis=0).reset_index().iloc[:,1:]
    set_monsoon_metrics = pd.concat(set_monsoon_metrics,axis=0).reset_index().iloc[:,1:]
    set_dry_metrics = pd.concat(set_dry_metrics,axis=0).reset_index().iloc[:,1:]

    all_metrics.append(set_all_metrics)
    monsoon_metrics.append(set_monsoon_metrics)
    dry_metrics.append(set_dry_metrics)

    #Data
    set_all_data = pd.concat(set_all_data,axis=0)
    set_monsoon_data = pd.concat(set_monsoon_data,axis=0)
    set_dry_data = pd.concat(set_dry_data,axis=0)

    all_data.append(set_all_data)
    monsoon_data.append(set_monsoon_data)
    dry_data.append(set_dry_data)

#Metrics (CC, rmse, bias, pval)
all_metrics = pd.concat(all_metrics,axis=0)
monsoon_metrics = pd.concat(monsoon_metrics,axis=0)
dry_metrics = pd.concat(dry_metrics,axis=0)

CCs_all.append(all_metrics.R)
rmses_all.append(all_metrics.iloc[:,1])
bias_all.append(all_metrics.iloc[:,2])
pvals_all.append(all_metrics.iloc[:,3])

CCs_monsoon.append(monsoon_metrics.R)
rmses_monsoon.append(monsoon_metrics.iloc[:,1])
bias_monsoon.append(monsoon_metrics.iloc[:,2])
pvals_monsoon.append(monsoon_metrics.iloc[:,3])

CCs_non.append(dry_metrics.R)
rmses_non.append(dry_metrics.iloc[:,1])
bias_non.append(dry_metrics.iloc[:,2])
pvals_non.append(dry_metrics.iloc[:,3])


#All datasets (only ONE source of data: AIRS downscaled)
all_data_concat.append(all_data)
monsoon_data_concat.append(monsoon_data)
dry_data_concat.append(dry_data)


CC_plots = [CCs_all,CCs_monsoon,CCs_non]
ub_rmse_plots = [rmses_all,rmses_monsoon,rmses_non]
bias_plots = [bias_all,bias_monsoon,bias_non]
pvals_plots = [pvals_all,pvals_monsoon,pvals_non]



titles = ['All Year (Daily)','Monsoon Months (Daily)','Non-Monsoon Months (Daily)']
source_names = ['AIRS DS']

for i in range(0,len(source_names)):
    for ii in range(0,len(set_names)):
        linear_plot(all_data_concat[i][ii].model,all_data_concat[i][ii].insitu,source_names[i] + ' ' + set_names[ii],savepath)

for i in range(0,len(source_names)):
    
    linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[i][ii].model for ii in range(0,len(all_data_concat[i]))]))),
              np.array(list(itertools.chain.from_iterable([all_data_concat[i][ii].insitu for ii in range(0,len(all_data_concat[i]))]))), source_names[i] + ' All Year',savepath)

for i in range(0,len(source_names)):

    linear_plot(np.array(list(itertools.chain.from_iterable([monsoon_data_concat[i][ii].model for ii in range(0,len(all_data_concat[i]))]))),
                np.array(list(itertools.chain.from_iterable([monsoon_data_concat[i][ii].insitu for ii in range(0,len(all_data_concat[i]))]))), source_names[i] + ' Monsoon',savepath)




################################################################################################
#MODIS DATASETS

#Choose GST
insitu_temps = insitu_gst
set_names = gst_names
temp = GST
savepath = 'GST'
keys = [sorted(hiwat_points.station),sorted(pd.concat([icimod_points.NAME,icimod_snow_points.Station])),sorted(ismn_points.site),sorted(ma_points.Site),sorted(zhao2021_points.Site)[-1]]


#Choose AT
insitu_temps = insitu_at
set_names = at_names
temp = AT
savepath = 'AT'
keys = [hiwat_points.station,icimod_snow_points.Station,ma_points.Site,ghcn_points.Name,['AYK','LDH','TGL','TSH','XDT']]
keys = [sorted(key) for key in keys]




#FOR Zhang's MYD11 GF

if temp == '\\GST':
    main_1km = r'\MODIS_LST\validation\{}\daily'.format(temp)
else:    
    main_1km = r'\MODIS_LST\validation\{}'.format(temp)
paths_1km = sorted(os.listdir(main_1km))
set_paths_1km = [sorted(glob.glob(main_1km+'\\'+path+'\*.csv')) for path in paths_1km]
setC = [[pd.read_csv(file) for file in set_path] for set_path in set_paths_1km]
[[setC[c][i].set_index(pd.to_datetime(setC[c][i].iloc[:,0]),inplace=True) for i in range(0,len(setC[c]))] for c in range(0,len(setC))]

setC_GF = [[set.iloc[:,-2:].mean(axis=1) for set in sets] for sets in setC]


################################################################################################
#Filtered for QC

main_1km = r'\{}'.format(temp)
paths_1km = sorted(os.listdir(main_1km))
modis_files = [sorted(glob.glob(main_1km+'\\'+path+'\*.csv')) for path in paths_1km]
setC = [[pd.read_csv(file) for file in set_path] for set_path in modis_files]
[[setC[c][i].set_index(pd.to_datetime(setC[c][i].iloc[:,0]),inplace=True) for i in range(0,len(setC[c]))] for c in range(0,len(setC))]


#FOR MYD11 QC
mydqc_day = pd.read_csv(r'\MYD11A1-006-QC-Day-lookup.csv')
mydqc_night = pd.read_csv(r'\MYD11A1-006-QC-Night-lookup.csv')
qcd_1 = mydqc_day[(mydqc_day['LST Error Flag'] == 'Average LST error <= 1K')]['Value']
qcd_2 = mydqc_day[(mydqc_day['LST Error Flag'] == 'Average LST error <= 1K') | (mydqc_day['LST Error Flag'] == 'Average LST error <= 2K')  ]['Value']
qcn_1 = mydqc_night[(mydqc_night['LST Error Flag'] == 'Average LST error <= 1K')]['Value']
qcn_2 = mydqc_night[(mydqc_night['LST Error Flag'] == 'Average LST error <= 1K') | (mydqc_night['LST Error Flag'] == 'Average LST error <= 2K')]['Value']

#FOR MOD11 QC
modqc_day = pd.read_csv(r'F:\raw-data\Samsung_T5_E\raw-data\MOD_MYD_11\MOD_11\2003\MOD11A1-006-QC-Day-lookup.csv')
modqc_night = pd.read_csv(r'F:\raw-data\Samsung_T5_E\raw-data\MOD_MYD_11\MOD_11\2003\MOD11A1-006-QC-Night-lookup.csv')
modqcd_1 = modqc_day[(modqc_day['LST Error Flag'] == 'Average LST error <= 1K')]['Value']
modqcd_2 = modqc_day[(modqc_day['LST Error Flag'] == 'Average LST error <= 1K') | (modqc_day['LST Error Flag'] == 'Average LST error <= 2K')]['Value']
modqcn_1 = modqc_night[(modqc_night['LST Error Flag'] == 'Average LST error <= 1K')]['Value']
modqcn_2 = modqc_night[(modqc_night['LST Error Flag'] == 'Average LST error <= 1K') | (modqc_day['LST Error Flag'] == 'Average LST error <= 2K')]['Value']

#Choose "LST error <= 2" or "LST error <= 1"
qcs = [modqcd_2,modqcn_2,qcd_2,qcn_2]
source_lsts = ['MOD11 Day','MOD11 Night','MYD11 Day','MYD11 Night']
source_qcs = ['QC MOD11 Day','QC MOD11 Night', 'QC MYD11 Day','QC MYD11 Night']

def mask_modis(dataset,qc,source_lst,source_qc):
    mask = dataset['{}'.format(source_qc)].isin(qc)
    masked_data = dataset['{}'.format(source_lst)].where(mask).astype('float32')
    return masked_data

#MASK MODIS by QC
masked_modis_lst = [[[mask_modis(station,qc,source_lst,source_qc) - 273.15 for qc,source_lst,source_qc in zip(qcs,source_lsts,source_qcs)] 
                    for station in set] for set in setC]


#MOD11 (AVG)
MOD11_daily = [[pd.concat(df[0:2],axis=1).mean(axis=1) for df in set] for set  in masked_modis_lst]
MOD11_monthly = [[pd.concat(df[0:2],axis=1).mean(axis=1).resample('1M').mean() for df in set] for set  in masked_modis_lst]
MOD11_annual = [[pd.concat(df[0:2],axis=1).mean(axis=1).resample('1Y').mean() for df in set] for set  in masked_modis_lst]
#MYD11 (AVG)
MYD11_daily = [[pd.concat(df[2:4],axis=1).mean(axis=1) for df in set] for set  in masked_modis_lst]
MYD11_monthly = [[pd.concat(df[2:4],axis=1).mean(axis=1).resample('1M').mean() for df in set] for set  in masked_modis_lst]
MYD11_annual = [[pd.concat(df[2:4],axis=1).mean(axis=1).resample('1Y').mean() for df in set] for set  in masked_modis_lst]



################################################################################################
CCs_all = []
rmses_all = []
bias_all = []
pvals_all = []

CCs_monsoon = []
rmses_monsoon = []
bias_monsoon = []
pvals_monsoon = []

CCs_non = []
rmses_non = []
bias_non = []
pvals_non = []

#Call new lists for datasets
all_data_concat = []
monsoon_data_concat = []
dry_data_concat = []

for modis,modis_name in zip([MOD11_daily,MYD11_daily,setC_GF],['MOD11','MYD11','Zhang et al. (2021) GF']):
    merged_ids = [[idx1.index.intersection(idx2.index) for idx1,idx2 in zip(temp_set,modis_set)] for temp_set,modis_set in zip(insitu_temps,modis)]
    
    all_metrics = []
    monsoon_metrics = []
    dry_metrics = []

    all_data = []
    monsoon_data = []
    dry_data = []

    for temp_set,modis_set,merged_id_set,set_name in zip(insitu_temps,modis,merged_ids,set_names):
            
        set_all_metrics = []
        set_monsoon_metrics = []
        set_dry_metrics = []
        
        set_all_data = []
        set_monsoon_data = []
        set_dry_data = []

        for i in range(0,len(temp_set)):

            try:
                model_set = (modis_set[i].loc[merged_id_set[i]].ravel()) #all
                model_monsoon_set = (modis_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([6,7,8])]].ravel()) #monsoon
                model_dry_set = (modis_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([11,12,1,2])]].ravel()) #dry

                insitu_set = (temp_set[i].loc[merged_id_set[i]].ravel()) #all
                insitu_monsoon_set = (temp_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([6,7,8])]].ravel()) #monsoon
                insitu_dry_set = (temp_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([11,12,1,2])]].ravel()) #dry

                set_all_metrics.append(linear_stats(model_set,insitu_set,set_name+' {} all'.format(modis_name)))
                set_monsoon_metrics.append(linear_stats(model_monsoon_set,insitu_monsoon_set,set_name+' {} monsoon'.format(modis_name)))
                set_dry_metrics.append(linear_stats(model_dry_set,insitu_dry_set,set_name+' {} dry '.format(modis_name)))

                set_all_data.append(pd.DataFrame([model_set.ravel(),insitu_set.ravel()]).T.rename(columns={0:'modis',1:'insitu'}))
                set_monsoon_data.append(pd.DataFrame([model_monsoon_set.ravel(),insitu_monsoon_set.ravel()]).T.rename(columns={0:'modis',1:'insitu'}))
                set_dry_data.append(pd.DataFrame([model_dry_set.ravel(),insitu_dry_set.ravel()]).T.rename(columns={0:'modis',1:'insitu'}))

            except IndexError or ValueError:
                continue

        #Metrics
        set_all_metrics = pd.concat(set_all_metrics,axis=0).reset_index().iloc[:,1:]
        set_monsoon_metrics = pd.concat(set_monsoon_metrics,axis=0).reset_index().iloc[:,1:]
        set_dry_metrics = pd.concat(set_dry_metrics,axis=0).reset_index().iloc[:,1:]

        all_metrics.append(set_all_metrics)
        monsoon_metrics.append(set_monsoon_metrics)
        dry_metrics.append(set_dry_metrics)

        #Data
        set_all_data = pd.concat(set_all_data,axis=0)
        set_monsoon_data = pd.concat(set_monsoon_data,axis=0)
        set_dry_data = pd.concat(set_dry_data,axis=0)

        all_data.append(set_all_data)
        monsoon_data.append(set_monsoon_data)
        dry_data.append(set_dry_data)

    #Metrics (CC, rmse, bias, pval)
    all_metrics = pd.concat(all_metrics,axis=0)
    monsoon_metrics = pd.concat(monsoon_metrics,axis=0)
    dry_metrics = pd.concat(dry_metrics,axis=0)

    CCs_all.append(all_metrics.R)
    rmses_all.append(all_metrics.iloc[:,1])
    bias_all.append(all_metrics.iloc[:,2])
    pvals_all.append(all_metrics.iloc[:,3])

    CCs_monsoon.append(monsoon_metrics.R)
    rmses_monsoon.append(monsoon_metrics.iloc[:,1])
    bias_monsoon.append(monsoon_metrics.iloc[:,2])
    pvals_monsoon.append(monsoon_metrics.iloc[:,3])

    CCs_non.append(dry_metrics.R)
    rmses_non.append(dry_metrics.iloc[:,1])
    bias_non.append(dry_metrics.iloc[:,2])
    pvals_non.append(dry_metrics.iloc[:,3])

    #Data ()
    all_data_concat.append(all_data)
    monsoon_data_concat.append(monsoon_data)
    dry_data_concat.append(dry_data)

CC_plots = [CCs_all,CCs_monsoon,CCs_non]
ub_rmse_plots = [rmses_all,rmses_monsoon,rmses_non]
bias_plots = [bias_all,bias_monsoon,bias_non]
pvals_plots = [pvals_all,pvals_monsoon,pvals_non]

titles = ['All Year (Daily)','Monsoon Months (Daily)','Non-Monsoon Months (Daily)']
source_names = ['MOD11', 'MYD11', 'Zhang et al. (2022) GF']

#ALL DATA
for i in range(0,len(source_names)):
    for ii in range(0,len(set_names)):
        linear_plot(all_data_concat[i][ii].modis,all_data_concat[i][ii].insitu,source_names[i] + ' ' + set_names[ii],savepath)


for i in range(0,len(source_names)):
    linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[i][ii].modis for ii in range(0,len(all_data_concat[i]))]))),
              np.array(list(itertools.chain.from_iterable([all_data_concat[i][ii].insitu for ii in range(0,len(all_data_concat[i]))]))), source_names[i] + ' All Year',savepath)
for i in range(0,len(source_names)):
    linear_plot(np.array(list(itertools.chain.from_iterable([monsoon_data_concat[i][ii].modis for ii in range(0,len(all_data_concat[i]))]))),
                np.array(list(itertools.chain.from_iterable([monsoon_data_concat[i][ii].insitu for ii in range(0,len(all_data_concat[i]))]))), source_names[i] + ' Monsoon',savepath)
