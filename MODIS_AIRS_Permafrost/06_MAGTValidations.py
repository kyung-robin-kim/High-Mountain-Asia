######################################
#For Validation of MGT/MAGT for SET B & Obu/Gruber/Ran
######################################

#There are multiple variables to assess: MAAT (air), MAGST (ground surface & active layer), MAGT (DZAA), SM (soil moisture), SD (snow depth)
#There are multiple validation datasets:
#1. NOAA GHCN - MAAT [snow]
#2. ISMN - SM, MAGST
#3. HiWAT - SM, MAGST, MAAT [snow]
#4. GNT-P (others) & Wani - MAGT
#5. Zhao 2021 - SM, MAGT (x6 for MAGST and MAAT too)
#6. Ma 2020 - SM, MAGST
#7. ICIMOD - MAGST [snow]

import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import itertools
import geopandas as gpd
from scipy.stats import gaussian_kde
import itertools
import matplotlib.pyplot as plt
import os
import datetime

degree_sign = u"\N{DEGREE SIGN}"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})

def linear_plot(model,insitu,title):
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
        ax.scatter(insitu, model, c=z, s=3)
        ax.plot(insitu, intercept + slope*insitu, label='r: {:.3f}'.format(round(r_value,3)),color='r')
        ax.set_ylabel('Modeled ({}C)'.format(degree_sign))
        ax.set_xlabel('In situ ({}C)'.format(degree_sign))
        ax.legend()
        ax.set_ylim(-10,10)
        ax.set_xlim(-10,10)
        ax.plot([-60,60],[-60,60],'--',color='black')
        ax.text(5,-3,s="count: {}".format(len(insitu)), fontsize=12, ha="left", va="top")
        ax.text(5,-4,s="p-val < 0.001", fontsize=12, ha="left", va="top")
        ax.text(5,-5,s="RMSE: {:.3f}".format(round(rmse,3)), fontsize=12, ha="left", va="top")
        ax.text(5,-6,s="bias: {:.3f}".format(round(bias,3)), fontsize=12, ha="left", va="top")
        ax.set_title(title,weight='bold')
        plt.tight_layout()

        plt.savefig(r'\{}.png'.format(title))

        print(title,round(r_value,3))
        print(len(model),'model points count')
        print(len(insitu),'insitu points count')

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
#CALL INSITU DATASETS (LST & GT)
######################################

# HiWAT - SM, MAGST, MAAT
path= r'\frozen_ground_obs'
xls_files_AWS = sorted(glob.glob(path+'\*\*AWS.xlsx'))

hiwat_stations = pd.read_excel(r'\Station location.xlsx')
xls_files_AWS_station = [[file for file in xls_files_AWS if file.find(id) > 0] for id in sorted(hiwat_stations.station)]

AWS_data = [[pd.read_excel(file).replace(-6999,np.nan) for file in xls_files] for xls_files in xls_files_AWS_station] #6 minutes to run
AWS_data_daily = [[data.set_index(pd.to_datetime(data.TIMESTAMP,'%Y-%m-d')).resample('1D').mean() for data in dataset] for dataset in AWS_data]
AWS_data_stdev = [[data.set_index(pd.to_datetime(data.TIMESTAMP,'%Y-%m-d')).resample('1D').std(ddof=1) for data in dataset] for dataset in AWS_data]

hiwat_soil_temp_annual = [pd.concat([data.filter(like='Ts_') for data in dataset],axis=0) for dataset in AWS_data_daily] #2014-2017: 0,4,10,20,40,80,120cm


#ICIMOD - MAGST, SD (12 seconds)
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
icimod_sd_points = pd.read_csv(r'\ICIMOD\stations.csv')
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


#Others - GTN-P (Wu) & Qin & Wani
others_data = pd.read_csv(r'\Permafrost\others.csv')
Wu_GNTP_dataset = others_data[others_data['Code'].str.contains('CN')].sort_values(by='Code')
Wu_GNTP_MAGT = Wu_GNTP_dataset.iloc[:,15:24].T
Wu_GNTP_depth = Wu_GNTP_dataset.iloc[:,32]
Wu_dataset_GT = others_data[others_data['Event'].str.contains('Wu')].iloc[:,0:27].sort_values(by='Code')
Qin_dataset_GT = others_data[others_data['Event'].str.contains('Qin')].iloc[:,0:29].sort_values(by='Code')


#Zhao_2021 -- MAGT, SM, MAGST 
path= r'\Zhao_2021_ALL'
xls_files = sorted(glob.glob(path+'\*.xlsx'))[:-1]
xls_sheets = [pd.ExcelFile(file).sheet_names for file in xls_files]
xls_data = [pd.read_excel(xls_file,sheet_name=sheet) for sheet,xls_file in zip(xls_sheets,xls_files)]
zhao2021_stations = pd.read_csv(r'\Zhao_2021_ALL\stations.csv')

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


#WANI
wani_files = sorted(glob.glob(r'\Wani_2020\gst\*.csv'))
wani_csv = [pd.read_csv(file,header=None) for file in wani_files]
julian_days = [round(csv.iloc[:,0]) for csv in wani_csv]
gsts = [csv.iloc[:,1] for csv in wani_csv]

def datetime_array_from_julian_day(start_date, julian_days):
    start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    datetime_array = []
    
    for julian_day in julian_days:
        delta = datetime.timedelta(days=julian_day - 1)
        date = start_datetime + delta
        datetime_array.append(date)
    
    return datetime_array

start_date = "2015-09-15"
dates = [datetime_array_from_julian_day(start_date, jul) for jul in julian_days]
wani_gst_ds = [pd.DataFrame([date,gst]).T.rename(columns={0:'dates',1:'gst'}).set_index('dates') for date,gst in zip(dates,gsts)]

#Wu 2015
wu_file = pd.read_csv(r'C:\Users\robin\Box\Data\StudyRegion\HMA\Permafrost\Wu_2015\Wu_2015_insitu.csv')
wu_file.MAGT

#Sun 2018
sun_file = pd.read_csv(r'C:\Users\robin\Box\Data\StudyRegion\HMA\Permafrost\Sun_2018\sun_2018_insitu.csv')

#Wang 2017
wang_file = pd.read_csv(r'C:\Users\robin\Box\Data\StudyRegion\HMA\Permafrost\Wang_2017\Wang_2017_insitu.csv')

#Luo 2018
luo_file = pd.read_csv(r'C:\Users\robin\Box\Data\StudyRegion\HMA\Permafrost\Luo_2018\Luo_2018_insitu.csv')



######################################
#Remotely Sensed ST vs InSitu LST/AT
######################################

#ANNUAL SOIL TEMP (SFG/AL)
#Top Soil Temp (annual)
hiwat_gst = [data.dropna().resample('1Y').mean().iloc[:,0] for data in hiwat_soil_temp_annual]
icimod_gst_I = [icimod_daily_gst.iloc[:,i].dropna().resample('1Y').mean() for i in range(0,len(icimod_daily_gst.keys()))]
icimod_gst_II = [icimod_sd_data[i].filter(like='Ground').dropna().resample('1Y').mean() for i in range(0,len(icimod_sd_data))]
[icimod_gst_I.append(icimod_gst_II[i].iloc[:,0]) for i in range(0,len(icimod_gst_II))]
ismn_gst = [data.dropna().resample('1Y').mean() for data in ts_cm_data[0]]
ma_gst = [df.iloc[:,0].dropna().resample('1Y').mean() for df in ma_soil_temps] 
zhao_gst = [df.iloc[:,0].dropna().resample('1Y').mean() for df in zhao_insitu[0]]
Wani_dataset_ST = others_data[others_data['Code'].str.contains('NIH')].loc[85:96].sort_values(by='Code').iloc[:,[5,-8,-7]] #surface GT

insitu_gt_s = [hiwat_gst,icimod_gst_I,ismn_gst,ma_gst,zhao_gst]
gst_names = sorted(['HiWAT','ICIMOD','ISMN','Ma 2020','Zhao 2021'])

#Deep Soil Temp (annual)
hiwat_gt = [data.dropna().resample('1Y').mean().iloc[:,-1] for data in hiwat_soil_temp_annual]
ismn_gt = [data.dropna().resample('1Y').mean() for data in ts_cm_data[3]] #0 for 5cm [118], 1 for 20cm [233], 2 for 40cm [159] and -1 for 80cm [50]
ma_gt = [df.iloc[:,-1].dropna().resample('1Y').mean() for df in ma_soil_temps] 
zhao_gt = [df.iloc[:,-1].dropna().resample('1Y').mean() for df in zhao_insitu[0]]
insitu_gt_d = [hiwat_gt,ma_gt,zhao_gt]
gt_names = sorted(['HiWAT','Ma 2020','Zhao 2021'])


#MEAN ANNUAL SOIL TEMP (SFG/AL)
#Top Soil Temp (mean annual)
hiwat_gst = [data.dropna().resample('1Y').mean().iloc[:,0].mean() for data in hiwat_soil_temp_annual]
icimod_gst_I = [icimod_daily_gst.iloc[:,i].dropna().resample('1Y').mean().mean() for i in range(0,len(icimod_daily_gst.keys()))]
icimod_gst_II = [icimod_sd_data[i].filter(like='Ground').dropna().resample('1Y').mean().mean() for i in range(0,len(icimod_sd_data))]
[icimod_gst_I.append(icimod_gst_II[i]) for i in range(0,len(icimod_gst_II))]
ismn_gst = [data.dropna().resample('1Y').mean().mean() for data in ts_cm_data[0]]
ma_gst = [df.iloc[:,0].dropna().resample('1Y').mean().mean() for df in ma_soil_temps] 
zhao_gst = [df.iloc[:,0].dropna().resample('1Y').mean().mean() for df in zhao_insitu[0]]
Wani_dataset_ST = others_data[others_data['Code'].str.contains('NIH')].loc[85:96].sort_values(by='Code').iloc[:,[5,-8,-7]] #surface GT
insitu_gt_s = [hiwat_gst,icimod_gst_I,ismn_gst,ma_gst,zhao_gst]
gst_names = sorted(['HiWAT','ICIMOD','ISMN','Ma 2020','Zhao 2021'])

#Deep Soil Temp (mean annual)
hiwat_gt = [data.dropna().resample('1Y').mean().iloc[:,-1].mean() for data in hiwat_soil_temp_annual]
#ismn_gt = [data.dropna().resample('1Y').mean().mean() for data in ts_cm_data[3]] #0 for 5cm [118], 1 for 20cm [233], 2 for 40cm [159] and -1 for 80cm [50]
ma_gt = [df.iloc[:,-1].dropna().resample('1Y').mean().mean() for df in ma_soil_temps] 
zhao_gt = [df.iloc[:,-1].dropna().resample('1Y').mean().mean() for df in zhao_insitu[0]] #include ZNH for MAGT
#insitu_gt_d = [hiwat_gt,ismn_gt,ma_gt,zhao_gt]
insitu_gt_d = [hiwat_gt,ma_gt,zhao_gt]
gt_names = sorted(['HiWAT','Ma 2020','Zhao 2021'])


#Mean Annual Ground (SFG/AL)
hiwat_gst = [data.dropna().interpolate('linear').resample('1Y').mean().iloc[:,-1].mean() for data in hiwat_soil_temp_annual]
icimod_gst_I = [icimod_daily_gst.iloc[:,i].dropna().interpolate('linear').resample('1Y').mean().mean() for i in range(0,len(icimod_daily_gst.keys()))]
icimod_gst_II = [icimod_sd_data[i].filter(like='Ground').dropna().interpolate('linear').resample('1Y').mean().mean() for i in range(0,len(icimod_sd_data))]
[icimod_gst_I.append(icimod_gst_II[i][0]) for i in range(0,len(icimod_gst_II))]
ismn_gst = [data.dropna().interpolate('linear').resample('1Y').mean().mean() for data in ts_cm_data[0]] #0 for 5cm [118], 1 for 20cm [233], 2 for 40cm [159] and -1 for 80cm [50]
ma_gst = [df.iloc[:,-1].dropna().interpolate('linear').resample('1Y').mean().mean() for df in ma_soil_temps] 
zhao_gst = [df.iloc[:,-1].dropna().interpolate('linear').resample('1Y').mean().mean() for df in zhao_insitu[0]]
luo_gst = luo_file.sort_values(by=['Site']).MAGST.reset_index().MAGST
luo_gst = luo_file.MAGST.reset_index().MAGST
wani_gst = [df.iloc[:,0].dropna().resample('1Y').mean().mean() for df in wani_gst_ds]

#Mean Annual Ground Borehole
Wu_GNTP_GT = others_data[others_data['Code'].str.contains('CN')].loc[0:12].sort_values(by='Code').iloc[:,[5,15,16,17,18,19,20,21,22,23,24]]
Wu_GNTP_GT = pd.concat([Wu_GNTP_GT.iloc[:,0],Wu_GNTP_GT.iloc[:,1:].mean(axis=1)],axis=1)
Wu_dataset_GT = others_data[others_data['Event'].str.contains('Wu')].iloc[:,0:27].sort_values(by='Code').iloc[:,[5,-1]]
Wu_all_dataset_GT =  pd.concat([Wu_GNTP_GT,Wu_dataset_GT.rename(columns={'2006_2010':0})],axis=0).sort_values(by='Code')
Qin_dataset_GT = others_data[others_data['Event'].str.contains('Qin')].iloc[:,0:29].sort_values(by='Code').iloc[:,[5,-1]]
others_magt = pd.concat([Wu_all_dataset_GT.iloc[:,1].rename('insitu'),Qin_dataset_GT.iloc[:,-1].rename('insitu')],axis=0).reset_index().insitu

zhao_magt_I = [data.iloc[:,-1].mean() for data in zhao_insitu[2][0]]
zhao_magt_I_sites = [data.Id_site[0] for data in zhao_insitu[2][0]]
zhao_magt_II = [data.iloc[:,1:].mean(axis=1).mean() for data in zhao_insitu[2][1]]
zhao_magt_II_sites = [data.Id_site[0] for data in zhao_insitu[2][1]]
zI_df = pd.DataFrame([zhao_magt_I_sites,zhao_magt_I]).T
zII_df = pd.DataFrame([zhao_magt_II_sites,zhao_magt_II]).T
zhao_magt = pd.concat([zI_df,zII_df],axis=0).sort_values(by=0).rename(columns={0:'site',1:'insitu'}).reset_index().insitu.astype(float)

wu_magt = wu_file.MAGT.reset_index().MAGT
sun_magt = sun_file.MAGT.reset_index().MAGT
wang_magt = wang_file.sort_values(by=['Site']).MAGT.reset_index().MAGT
luo_magt = luo_file.sort_values(by=['Site']).TTOP.dropna().reset_index().TTOP

#All datasets
insitu_magt_AL = [hiwat_gst,icimod_gst_I,ismn_gst,luo_gst,ma_gst,wani_gst,zhao_gst]
insitu_magt_BH = [luo_magt,others_magt,sun_magt,wang_magt,wu_magt,zhao_magt]

AL_names = sorted(['Luo 2018','HiWAT','ICIMOD','ISMN','Ma 2020','Zhao 2021','Wani 2020'])
BH_names = sorted(['Wang 2017','Wu 2015','Sun 2015','Luo 2018','Others','Zhao 2021'])

#################################################################################
#################################################################################
#Choose MGT or MAGT (?)
GT_D = r'\MGT\Deep' #insitu_gt_d
GT_S = r'\MGT\Shallow' #insitu_gt_s
MAGT = r'\MAGT'


insitu_temps = insitu_gt_s
set_names = gst_names
temp = GT_S
savepath = ''


insitu_temps = insitu_gt_d
set_names = gt_names
temp = GT_D
savepath = ''
#################################################################################
#################################################################################


#################################################################################
#SET B - Mean Annual

#BOREHOLES
temp = '\Borehole'
insitu_temps = insitu_magt_BH
set_names = BH_names

gt_1km = r'\multi_elev_lat'+temp
paths_gt_1km = sorted(os.listdir(gt_1km))
set_paths_gt_1km = [sorted(glob.glob(gt_1km+'\\'+path+'\*.csv')) for path in paths_gt_1km]
setB = [[pd.read_csv(file) for file in set_path] for set_path in set_paths_gt_1km]
[[setB[c][i].set_index(pd.to_datetime(setB[c][i].iloc[:,0]),inplace=True) for i in range(0,len(setB[c]))] for c in range(0,len(setB))]

setB_modmyd_airs = [[set.iloc[:,1].resample('1Y').mean().mean() for set in sets] for sets in setB]
setB_modmyd_airs_nival = [[set.iloc[:,2].resample('1Y').mean().mean() for set in sets] for sets in setB]
setB_modmyd_airs_thermal_era = [[set.iloc[:,3].resample('1Y').mean().mean() for set in sets] for sets in setB]
setB_modmyd_airs_thermal_gldas = [[set.iloc[:,4].resample('1Y').mean().mean() for set in sets] for sets in setB]

CCs_all_bh = []
rmses_all_bh = []
bias_all_bh = []
pvals_all_bh = []
all_data_borehole = []

for modis,modis_name in zip([setB_modmyd_airs,setB_modmyd_airs_nival,setB_modmyd_airs_thermal_era,setB_modmyd_airs_thermal_gldas],['MODMYD11_AIRS','MODMYD11_AIRS Nival','MODMYD11_AIRS ERA5-L','MODMYD11_AIRS GLDAS']):

    set_all_data = []
    set_all_metrics = []

    for temp_set,modis_set,set_name in zip(insitu_temps,modis,set_names):

        try:
            set_all_metrics.append(linear_stats(np.array(modis_set),np.array(temp_set),set_name+' {} all'.format(modis_name)))
            set_all_data.append(pd.DataFrame([modis_set,temp_set]).T.rename(columns={0:'modis',1:'insitu'}))

        except IndexError or ValueError:
            continue
        

    #Metrics (CC, rmse, bias, pval)
    all_metrics = pd.concat(set_all_metrics,axis=0)

    CCs_all_bh.append(all_metrics.R)
    rmses_all_bh.append(all_metrics.iloc[:,1])
    bias_all_bh.append(all_metrics.iloc[:,2])
    pvals_all_bh.append(all_metrics.iloc[:,3])

    #Data ()
    all_data_borehole.append(set_all_data)
all_data_borehole = [pd.concat(ds) for ds in all_data_borehole]



#SFG/AL
temp = '\AL'
insitu_temps = insitu_magt_AL
set_names = AL_names

gt_1km = r'\multi_elev_lat'+temp
paths_gt_1km = sorted(os.listdir(gt_1km))
set_paths_gt_1km = [sorted(glob.glob(gt_1km+'\\'+path+'\*.csv')) for path in paths_gt_1km]
setB = [[pd.read_csv(file) for file in set_path] for set_path in set_paths_gt_1km]
[[setB[c][i].set_index(pd.to_datetime(setB[c][i].iloc[:,0]),inplace=True) for i in range(0,len(setB[c]))] for c in range(0,len(setB))]

setB_modmyd_airs = [[set.iloc[:,1].resample('1Y').mean().mean() for set in sets] for sets in setB]
setB_modmyd_airs_nival = [[set.iloc[:,2].resample('1Y').mean().mean() for set in sets] for sets in setB]
setB_modmyd_airs_thermal_era = [[set.iloc[:,3].resample('1Y').mean().mean() for set in sets] for sets in setB]
setB_modmyd_airs_thermal_gldas = [[set.iloc[:,4].resample('1Y').mean().mean() for set in sets] for sets in setB]

CCs_all_al = []
rmses_all_al = []
bias_all_al = []
pvals_all_al = []
all_data_surface = []

for modis,modis_name in zip([setB_modmyd_airs,setB_modmyd_airs_nival,setB_modmyd_airs_thermal_era,setB_modmyd_airs_thermal_gldas],['MAST','MAGST','TTOP ERA5-L','TTOP GLDAS']):

    set_all_data = []
    set_all_metrics = []

    for temp_set,modis_set,set_name in zip(insitu_temps,modis,set_names):

        try:
            set_all_metrics.append(linear_stats(np.array(modis_set),np.array(temp_set),set_name+' {} all'.format(modis_name)))
            set_all_data.append(pd.DataFrame([modis_set,temp_set]).T.rename(columns={0:'modis',1:'insitu'}))

        except IndexError or ValueError:
            continue
        

    #Metrics (CC, rmse, bias, pval)
    all_metrics = pd.concat(set_all_metrics,axis=0)

    CCs_all_al.append(all_metrics.R)
    rmses_all_al.append(all_metrics.iloc[:,1])
    bias_all_al.append(all_metrics.iloc[:,2])
    pvals_all_al.append(all_metrics.iloc[:,3])

    all_data_surface.append(set_all_data)
all_surface_data = [pd.concat(ds) for ds in all_data_surface]


source_names  = ['MAGT-Ia','MAGT-II','MAGT-IIIb','MAGT-IIIa']

#ALL DATA
for i in range(0,len(source_names)):
    for ii in range(0,len(set_names)):
        linear_plot(all_data_borehole[i][ii].modis,all_data_borehole[i][ii].insitu,source_names[i] + ' ' + set_names[ii])

for i in range(0,len(source_names)):
    linear_plot(all_surface_data[i].modis,all_surface_data[i].insitu,source_names[i] + ' SFG/AL')
                
for i in range(0,len(source_names)):
    linear_plot(all_data_borehole[i].modis,all_data_borehole[i].insitu,source_names[i] + ' Borehole')
                

for i in range(0,len(source_names)):
    all_data_model=pd.concat([all_surface_data[i].modis,all_data_borehole[i].modis])
    all_data_insitu=pd.concat([all_surface_data[i].insitu,all_data_borehole[i].insitu])
    
    linear_plot(all_data_model,all_data_insitu,source_names[i])



#################################################################################
#MAGTs 
#Mean Annual Ground AL
hiwat_gst = [data.dropna().interpolate('linear').resample('1Y').mean().iloc[:,-1].mean() for data in hiwat_soil_temp_annual]
icimod_gst_I = [icimod_daily_gst.iloc[:,i].dropna().interpolate('linear').resample('1Y').mean().mean() for i in range(0,len(icimod_daily_gst.keys()))]
icimod_gst_II = [icimod_sd_data[i].filter(like='Ground').dropna().interpolate('linear').resample('1Y').mean().mean() for i in range(0,len(icimod_sd_data))]
[icimod_gst_I.append(icimod_gst_II[i][0]) for i in range(0,len(icimod_gst_II))]
ismn_gst = [data.dropna().interpolate('linear').resample('1Y').mean().mean() for data in ts_cm_data[0]] #0 for 5cm [118], 1 for 20cm [233], 2 for 40cm [159] and -1 for 80cm [50]
ma_gst = [df.iloc[:,-1].dropna().interpolate('linear').resample('1Y').mean().mean() for df in ma_soil_temps] 
zhao_gst = [df.iloc[:,-1].dropna().interpolate('linear').resample('1Y').mean().mean() for df in zhao_insitu[0]] #include ZNH for MAGT
luo_gst = luo_file.sort_values(by=['Site']).MAGST.reset_index().MAGST
wani_gst = [df.iloc[:,0].dropna().resample('1Y').mean().mean() for df in wani_gst_ds]

#Mean Annual Ground Borehole
Wu_GNTP_GT = others_data[others_data['Code'].str.contains('CN')].loc[0:12].sort_values(by='Code').iloc[:,[5,15,16,17,18,19,20,21,22,23,24]]
Wu_GNTP_GT = pd.concat([Wu_GNTP_GT.iloc[:,0],Wu_GNTP_GT.iloc[:,1:].mean(axis=1)],axis=1)
Wu_dataset_GT = others_data[others_data['Event'].str.contains('Wu')].iloc[:,0:27].sort_values(by='Code').iloc[:,[5,-1]]
Wu_all_dataset_GT =  pd.concat([Wu_GNTP_GT,Wu_dataset_GT.rename(columns={'2006_2010':0})],axis=0).sort_values(by='Code')
Qin_dataset_GT = others_data[others_data['Event'].str.contains('Qin')].iloc[:,0:29].sort_values(by='Code').iloc[:,[5,-1]]
others_magt = pd.concat([Wu_all_dataset_GT.iloc[:,1].rename('insitu'),Qin_dataset_GT.iloc[:,-1].rename('insitu')],axis=0).reset_index().insitu

zhao_magt_I = [data.iloc[:,-1].mean() for data in zhao_insitu[2][0]]
zhao_magt_I_sites = [data.Id_site[0] for data in zhao_insitu[2][0]]
zhao_magt_II = [data.iloc[:,1:].mean(axis=1).mean() for data in zhao_insitu[2][1]]
zhao_magt_II_sites = [data.Id_site[0] for data in zhao_insitu[2][1]]
zI_df = pd.DataFrame([zhao_magt_I_sites,zhao_magt_I]).T
zII_df = pd.DataFrame([zhao_magt_II_sites,zhao_magt_II]).T
zhao_magt = pd.concat([zI_df,zII_df],axis=0).sort_values(by=0).rename(columns={0:'site',1:'insitu'}).reset_index().insitu.astype(float)

wu_magt = wu_file.MAGT.reset_index().MAGT
sun_magt = sun_file.MAGT.reset_index().MAGT
wang_magt = wang_file.sort_values(by=['Site']).MAGT.reset_index().MAGT
luo_magt = luo_file.sort_values(by=['Site']).TTOP.dropna().reset_index().TTOP

#All datasets
insitu_magt_AL = [hiwat_gst,icimod_gst_I,ismn_gst,luo_gst,ma_gst,wani_gst,zhao_gst]
insitu_magt_BH = [luo_magt,others_magt,sun_magt,wang_magt,wu_magt,zhao_magt]
insitu_magt_all = [hiwat_gst,icimod_gst_I,ismn_gst,luo_gst,ma_gst,wani_gst,zhao_gst,luo_magt,others_magt,sun_magt,wang_magt,wu_magt,zhao_magt]

AL_names = sorted(['Luo 2018','HiWAT','ICIMOD','ISMN','Ma 2020','Zhao 2021','Wani 2020'])
BH_names = sorted(['Wang 2017','Wu 2015','Sun 2015','Luo 2018','Others','Zhao 2021'])



#################################################################################
#SET B Lit - Mean Annual

lit_magts = sorted(glob.glob(r'\MAGT_litrev\updated\*.csv'))

obu_csv = pd.read_csv(lit_magts[0]).iloc[:,1:]
ran_csv = pd.read_csv(lit_magts[1]).iloc[:,1:]


#AL id:0-116
#BH id:116-297
insitu_magt_all_lit = list(itertools.chain.from_iterable(insitu_magt_all))
insitu_magt_AL_lit = list(itertools.chain.from_iterable(insitu_magt_AL))
insitu_magt_BH_lit = list(itertools.chain.from_iterable(insitu_magt_BH))

#Active Layers
ran_al= np.array(ran_csv.magt[0:225])
obu_al = np.array(obu_csv.magt[0:225])
insitu= np.array(insitu_magt_AL_lit)

#Boreholes
ran_bh= np.array(ran_csv.magt[225:])
obu_bh= np.array(obu_csv.magt[225:])
insitu= np.array(insitu_magt_BH_lit)

#Everything
ran = np.array(ran_csv.magt)
obu = np.array(obu_csv.magt)
insitu= np.array(insitu_magt_all_lit)


for model,insitu,title in zip([ran_al,obu_al,ran_bh,obu_bh,ran,obu],
                              [insitu_magt_AL_lit,insitu_magt_AL_lit,insitu_magt_BH_lit,insitu_magt_BH_lit,insitu_magt_all_lit,insitu_magt_all_lit],
                              ['MAGT-IV (Ran et al., 2022) SFG/AL','MAGT-IIIc (Obu et al., 2019) SFG/AL','MAGT-IV (Ran et al., 2022) Boreholes',
                               'MAGT-IIIc (Obu et al., 2019) Boreholes','MAGT-IV (Ran et al., 2022)','MAGT-IIIc (Obu et al., 2019)']):

        model = np.array(model)
        insitu = np.array(insitu)

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
        ax.scatter(insitu, model, c=z, s=3)
        ax.plot(insitu, intercept + slope*insitu, label='r: {:.3f}'.format(round(r_value,3)),color='r')
        ax.set_ylabel('Modeled ({}C)'.format(degree_sign))
        ax.set_xlabel('In situ ({}C)'.format(degree_sign))
        ax.legend()
        ax.set_ylim(-10,10)
        ax.set_xlim(-10,10)
        ax.plot([-60,60],[-60,60],'--',color='black')
        ax.text(5,-3,s="count: {}".format(len(insitu)), fontsize=12, ha="left", va="top")
        ax.text(5,-4,s="p-val < 0.001", fontsize=12, ha="left", va="top")
        ax.text(5,-5,s="RMSE: {:.3f}".format(round(rmse,3)), fontsize=12, ha="left", va="top")
        ax.text(5,-6,s="bias: {:.3f}".format(round(bias,3)), fontsize=12, ha="left", va="top")
        ax.set_title(title,weight='bold')
        plt.tight_layout()
        print(len(insitu),'insitu point no.')
        print(len(model),'modeled point no.')

        plt.savefig(r'\MAGT\models\{}.png'.format(title))