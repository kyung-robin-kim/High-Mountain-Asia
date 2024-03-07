######################################
#For Validation of LST and SM products
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


def linear_stats(model,insitu,title):
    try:
        nanmask = ~np.isnan(model) & ~np.isnan(insitu)
        model = model[nanmask]
        insitu = insitu[nanmask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(insitu,model)
        modeled_y = intercept + slope*insitu

        bias = np.mean(modeled_y - insitu)
        ub_rmse = np.sqrt(np.sum((modeled_y - insitu)**2)/(len(insitu)-1))
        std_err

        metrics = pd.DataFrame([round(r_value,3),round(ub_rmse,4),round(bias,4),round(p_value,4),title]).rename({0:'R',1:'ub-rmse',2:'bias',3:'pval',4:'label'}).T

        return metrics
    
    except ValueError:
        print('error {}'.format(title))

plt.rcParams["font.family"] = "Times New Roman"

def linear_plot(model,insitu,title,savepath):
    try:
        nanmask = ~np.isnan(model) & ~np.isnan(insitu)
        model = model[nanmask]
        insitu = insitu[nanmask]

        slope, intercept, r_value, std_err = stats.linregress(insitu,model)
        modeled_y = intercept + slope*insitu

        bias = np.mean(modeled_y - insitu)
        ub_rmse = np.sqrt(np.sum((modeled_y - insitu)**2)/(len(insitu)-1))
        std_err

        # Calculate the point density
        xy = np.vstack([insitu,model])
        z = gaussian_kde(xy)(xy)

        fig,ax = plt.subplots()
        ax.scatter(insitu, model, c=z, s=1)
        ax.plot(insitu, intercept + slope*insitu, label='r: {}'.format(round(r_value,3)),color='r')
        ax.set_ylabel('{} VWC (m3/m3)'.format(savepath))
        ax.set_xlabel('In situ VWC (m3/m3)')
        ax.legend()
        ax.set_ylim(0,0.8)
        ax.set_xlim(0,0.8)
        ax.plot([-1,1],[-1,1],'--',color='black')
        ax.text(0.2,0.70,s="count: {}".format(len(insitu)), fontsize=12, ha="left", va="top")
        ax.text(0.2,.60,s="ubRMSE: {}".format(round(ub_rmse,4)), fontsize=10, ha="left", va="top")
        ax.text(0.2,0.55,s="bias: {}".format(round(bias,4)), fontsize=10, ha="left", va="top")
        ax.text(0.2,0.65,s="p-val: {:.3f}", fontsize=10, ha="left", va="top")
        #ax.text(0.2,0.65,s="p-val < 0.001", fontsize=12, ha="left", va="top")

        ax.set_title(title,weight='bold')

        #plt.savefig(r'\validations\SM\{}\{}.png'.format(savepath,title))
        #plt.savefig(r'\validations\SM\{}\{}.pdf'.format(savepath,title))

        print(title,round(r_value,3))

    except ValueError:
        print('error {}'.format(title))



######################################
#CALL INSITU DATASETS (SM)
######################################

# ISMN - SM
ismn_dir = r'\ISMN'
ismn_points = [pd.read_csv(file) for file in sorted(glob.glob(ismn_dir+'\points\*.csv'))]
sm_file_names = [sorted(glob.glob(ismn_dir+r'\MOIST\*_{}cm*.csv'.format(cm))) for cm in ['05','20','40','80']]

sm_cm_data = [[pd.read_csv(file).set_index(pd.to_datetime(pd.read_csv(file).iloc[:,0])).iloc[:,6] for file in cm] for cm in sm_file_names]

mm_sm = [[csv.resample('1M').mean().loc[pd.to_datetime('2003-12-31'):pd.to_datetime('2016-12-31')] for csv in cm] for cm in sm_cm_data]
stdevs = [[csv.resample('1M').std(ddof=1).loc[pd.to_datetime('2003-12-31'):pd.to_datetime('2016-12-31')] for csv in cm] for cm in sm_cm_data]
counts = [[csv.resample('1M').count().loc[pd.to_datetime('2003-12-31'):pd.to_datetime('2016-12-31')] for  csv in cm] for cm in sm_cm_data]
ismn_dicts_sm= [pd.DataFrame([{'in_situ_MM_SM':masm,'in_situ_MM_SM_std':stdev,'in_situ_count':count} for masm,stdev,count in zip(mm_sm_cm,stdevs_cm,counts_cm)])
        for mm_sm_cm,stdevs_cm,counts_cm in zip(mm_sm,stdevs,counts)]

# HiWAT - SM
path= r'\frozen_ground_obs'
xls_files_AWS = sorted(glob.glob(path+'\*\*AWS.xlsx'))
hiwat_stations = pd.read_excel(r'\Station location.xlsx')
xls_files_AWS_station = [[file for file in xls_files_AWS if file.find(id) > 0] for id in sorted(hiwat_stations.station)]
AWS_data = [[pd.read_excel(file).replace(-6999,np.nan) for file in xls_files] for xls_files in xls_files_AWS_station] #6 minutes to run
AWS_data_daily = [[data.set_index(pd.to_datetime(data.TIMESTAMP,'%Y-%m-d')).resample('1D').mean() for data in dataset] for dataset in AWS_data]
AWS_data_stdev = [[data.set_index(pd.to_datetime(data.TIMESTAMP,'%Y-%m-d')).resample('1D').std(ddof=1) for data in dataset] for dataset in AWS_data]
hiwat_soil_moist_annual = [pd.concat([data.filter(like='Ms_') for data in dataset],axis=0) for dataset in AWS_data_daily] #%


#Zhao_2021 - SM
path= r'\Permafrost\Zhao_2021_ALL'
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


#Ma 2020 - SM
import os
mainpath = r'\Ma_ESSD'
ma_2020_stations = sorted(os.listdir(r'\Ma_ESSD')[1:-2])
soil_files = [sorted(glob.glob(mainpath+'\{}\SOIL*\*20*.csv'.format(station))) for station in ma_2020_stations]
soil_data = [[pd.read_csv(i) for i in files] for files in soil_files]
soil_data = [[i.set_index(pd.to_datetime(i.Timestamp)) for i in data] for data in soil_data]
ma_soil_vwc = [pd.concat([i.filter(like='Swc').astype(float) for i in data],axis=0) for data in soil_data] 
ma_soil_vwc[0].Swc_4cm_10MinAve = ma_soil_vwc[0].Swc_4cm_10MinAve/100
ma_soil_vwc[0].Swc_20cm_10MinAve = ma_soil_vwc[0].Swc_20cm_10MinAve/100
ma_soil_vwc[3] = ma_soil_vwc[3]/100
ma_soil_vwc[4] = ma_soil_vwc[4]/100

################################################################################################
#Modeled Soil Moisture
################################################################################################




#############################
#MONTHLY #GLDAS NOAH soil moisture
#############################
main = r'\GLDAS\NOAH\HMA\soil_moisture\validations'
paths = sorted(['\\ISMN','\\Ma2020','\\HiWAT','\\Zhao2021'])
set_paths = [sorted(glob.glob(main+path+'\*.csv')) for path in paths]

gldas_sets = [[pd.read_csv(file) for file in set_path] for set_path in set_paths]
[[set.set_index(pd.to_datetime(set.time),inplace=True) for set in sets] for sets in gldas_sets]

gldas_avg = [[set.iloc[:,1].resample('1M').mean() for set in sets] for sets in gldas_sets]
gldas_swv1 = [[set.iloc[:,2].resample('1M').mean() for set in sets] for sets in gldas_sets]
gldas_swv2 = [[set.iloc[:,3].resample('1M').mean() for set in sets] for sets in gldas_sets]
gldas_swv3 = [[set.iloc[:,4].resample('1M').mean() for set in sets] for sets in gldas_sets]
gldas_swv4 = [[set.iloc[:,5].resample('1M').mean() for set in sets] for sets in gldas_sets]


#Index 0 for average, -1 for deepest layer
insitu_hiwat_top = [(site.resample('1M').mean()/100).iloc[:,0] for site in hiwat_soil_moist_annual]
insitu_ma_top = [(site.resample('1M').mean()).iloc[:,0] for site in ma_soil_vwc]
insitu_zhao_top = [(site.resample('1M').mean()).iloc[:,0] for site in zhao_insitu[1]]
insitu_ismn_top = [(site.resample('1M').mean()) for site in mm_sm[0]]

insitu_hiwat_low = [(site.resample('1M').mean()/100).iloc[:,-1] for site in hiwat_soil_moist_annual]
insitu_ma_low = [(site.resample('1M').mean()).iloc[:,-1] for site in ma_soil_vwc]
insitu_zhao_low = [(site.resample('1M').mean()).iloc[:,-1] for site in zhao_insitu[1]]
insitu_ismn_low = [(site.resample('1M').mean()) for site in mm_sm[-1]]

insitu_sm_top = [insitu_hiwat_top,insitu_ismn_top,insitu_ma_top,insitu_zhao_top]
insitu_sm_low = [insitu_hiwat_low,insitu_ismn_low,insitu_ma_low,insitu_zhao_low]
sm_names = sorted(['HiWAT','ISMN','Ma 2020','Zhao 2021'])

freq = 'Monthly Validation'


#############################
#Select Depth

insitu_sm_set = insitu_sm_low
insitu_sm_set = insitu_sm_top

#############################

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

all_data_concat = []
monsoon_data_concat = []
dry_data_concat = []

for gldas,sm_depth in zip([gldas_avg,gldas_swv1,gldas_swv2,gldas_swv3,gldas_swv4],['Average','0-10 cm','10-40 cm', '40-100 cm','100-200 cm']):
    merged_ids = [[idx1.index.intersection(idx2.index) for idx1,idx2 in zip(sm_set,gldas_set)] for sm_set,gldas_set in zip(insitu_sm_set,gldas)]

    all_metrics = []
    monsoon_metrics = []
    dry_metrics = []

    all_data = []
    monsoon_data = []
    dry_data = []

    for sm_set,gldas_set,merged_id_set,set_name in zip(insitu_sm_set,gldas,merged_ids,sm_names):
            
        set_all_metrics = []
        set_monsoon_metrics = []
        set_dry_metrics = []
        
        set_all_data = []
        set_monsoon_data = []
        set_dry_data = []

        for i in range(0,len(sm_set)):
            try:

                model_set = (gldas_set[i].loc[merged_id_set[i]].ravel()) #all
                model_monsoon_set = (gldas_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([6,7,8])]].ravel()) #monsoon
                model_dry_set = (gldas_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([11,12,1,2])]].ravel()) #dry

                insitu_set = (sm_set[i].loc[merged_id_set[i]].ravel()) #all
                insitu_monsoon_set = (sm_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([6,7,8])]].ravel()) #monsoon
                insitu_dry_set = (sm_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([11,12,1,2])]].ravel()) #dry

                set_all_metrics.append(linear_stats(model_set,insitu_set,set_name+' {} all'.format(sm_depth)))
                set_monsoon_metrics.append(linear_stats(model_monsoon_set,insitu_monsoon_set,set_name+' {} monsoon'.format(sm_depth)))
                set_dry_metrics.append(linear_stats(model_dry_set,insitu_dry_set,set_name+' {} dry '.format(sm_depth)))

                set_all_data.append(pd.DataFrame([model_set.ravel(),insitu_set.ravel()]).T.rename(columns={0:'gldas',1:'insitu'}))
                set_monsoon_data.append(pd.DataFrame([model_monsoon_set.ravel(),insitu_monsoon_set.ravel()]).T.rename(columns={0:'gldas',1:'insitu'}))
                set_dry_data.append(pd.DataFrame([model_dry_set.ravel(),insitu_dry_set.ravel()]).T.rename(columns={0:'gldas',1:'insitu'}))

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
source_names = ['Average','0-10 cm','10-40 cm', '40-100 cm','100-200 cm']


savepath = 'GLDAS'


#ALL DATA
#average
for i in range(0,4):
    linear_plot(all_data_concat[0][i].gldas,all_data_concat[0][i].insitu,sm_names[i] + ' avg')
#289 cm
for i in range(0,4):
    linear_plot(all_data_concat[-1][i].gldas,all_data_concat[-1][i].insitu,sm_names[i] + ' 200cm') 


#For comparing upper soil layers (avg)
linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[1][i].gldas for i in range(0,len(all_data_concat[0]))]))),
            np.array(list(itertools.chain.from_iterable([all_data_concat[1][i].insitu for i in range(0,len(all_data_concat[0]))]))),'Upper SM [0-10cm]',savepath)

linear_plot(np.array(list(itertools.chain.from_iterable([monsoon_data_concat[1][i].gldas for i in range(0,len(all_data_concat[0]))]))),
            np.array(list(itertools.chain.from_iterable([monsoon_data_concat[1][i].insitu for i in range(0,len(all_data_concat[0]))]))),'Upper SM [0-10cm] (Monsoon)',savepath)

linear_plot(np.array(list(itertools.chain.from_iterable([dry_data_concat[1][i].gldas for i in range(0,len(all_data_concat[0]))]))),
            np.array(list(itertools.chain.from_iterable([dry_data_concat[1][i].insitu for i in range(0,len(all_data_concat[0]))]))),'Upper SM [0-10cm] (Non-Monsoon)',savepath)

#For comparing lower soil layers
linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[-1][i].gldas for i in range(0,len(all_data_concat[-1]))]))),
            np.array(list(itertools.chain.from_iterable([all_data_concat[-1][i].insitu for i in range(0,len(all_data_concat[-1]))]))),'Lower SM [100-200cm]',savepath)

linear_plot(np.array(list(itertools.chain.from_iterable([monsoon_data_concat[-1][i].gldas for i in range(0,len(all_data_concat[-1]))]))),
            np.array(list(itertools.chain.from_iterable([monsoon_data_concat[-1][i].insitu for i in range(0,len(all_data_concat[-1]))]))),'Lower SM [100-200cm] (Monsoon)',savepath)

linear_plot(np.array(list(itertools.chain.from_iterable([dry_data_concat[-1][i].gldas for i in range(0,len(all_data_concat[-1]))]))),
            np.array(list(itertools.chain.from_iterable([dry_data_concat[-1][i].insitu for i in range(0,len(all_data_concat[-1]))]))),'Lower SM [100-200cm] (Non-Monsoon)',savepath)



#############################
#ANNUAL
main = r'\validations'
paths = sorted(['\\ISMN','\\Ma2020','\\HiWAT','\\Zhao2021'])
set_paths = [sorted(glob.glob(main+path+'\*.csv')) for path in paths]

gldas_sets = [[pd.read_csv(file) for file in set_path] for set_path in set_paths]
[[set.set_index(pd.to_datetime(set.time),inplace=True) for set in sets] for sets in gldas_sets]

gldas_avg = [[set.iloc[:,1].resample('1Y').mean() for set in sets] for sets in gldas_sets]
gldas_swv1 = [[set.iloc[:,2].resample('1Y').mean() for set in sets] for sets in gldas_sets]
gldas_swv2 = [[set.iloc[:,3].resample('1Y').mean() for set in sets] for sets in gldas_sets]
gldas_swv3 = [[set.iloc[:,4].resample('1Y').mean() for set in sets] for sets in gldas_sets]
gldas_swv4 = [[set.iloc[:,5].resample('1Y').mean() for set in sets] for sets in gldas_sets]

#Index 0 for average, -1 for deepest layer
insitu_hiwat_top = [(site.resample('1Y').mean()/100).iloc[:,0] for site in hiwat_soil_moist_annual]
insitu_ma_top = [(site.resample('1Y').mean()).iloc[:,0] for site in ma_soil_vwc]
insitu_zhao_top = [(site.resample('1Y').mean()).iloc[:,0] for site in zhao_insitu[1]]
insitu_ismn_top = [(site.resample('1Y').mean()) for site in mm_sm[0]]

insitu_hiwat_low = [(site.resample('1Y').mean()/100).iloc[:,-1] for site in hiwat_soil_moist_annual]
insitu_ma_low = [(site.resample('1Y').mean()).iloc[:,-1] for site in ma_soil_vwc]
insitu_zhao_low = [(site.resample('1Y').mean()).iloc[:,-1] for site in zhao_insitu[1]]
insitu_ismn_low = [(site.resample('1Y').mean()) for site in mm_sm[-1]]

insitu_sm_top = [insitu_hiwat_top,insitu_ismn_top,insitu_ma_top,insitu_zhao_top]
insitu_sm_low = [insitu_hiwat_low,insitu_ismn_low,insitu_ma_low,insitu_zhao_low]
sm_names = sorted(['HiWAT','ISMN','Ma 2020','Zhao 2021'])


freq = 'Annual Validation'
#############################
#PLOTS AND METRICS
CCs_all = []
rmses_all = []
bias_all = []
pvals_all = []
all_data_concat = []

for gldas,sm_depth in zip([gldas_avg,gldas_swv1,gldas_swv2,gldas_swv3,gldas_swv4],['Average','0-10 cm','10-40 cm', '40-100 cm','100-200 cm']):
    merged_ids = [[idx1.index.intersection(idx2.index) for idx1,idx2 in zip(sm_set,gldas_set)] for sm_set,gldas_set in zip(insitu_sm_low,gldas)]

    all_metrics = []
    monsoon_metrics = []
    dry_metrics = []

    all_data = []
    monsoon_data = []
    dry_data = []

    for sm_set,gldas_set,merged_id_set,set_name in zip(insitu_sm_low,gldas,merged_ids,sm_names):
            
        set_all_metrics = []
        set_monsoon_metrics = []
        set_dry_metrics = []
        
        set_all_data = []
        set_monsoon_data = []
        set_dry_data = []

        for i in range(0,len(sm_set)):
            try:

                model_set = (gldas_set[i].loc[merged_id_set[i]].ravel()) #all
                insitu_set = (sm_set[i].loc[merged_id_set[i]].ravel()) #all
                set_all_metrics.append(linear_stats(model_set,insitu_set,set_name+' {} all'.format(sm_depth)))
                set_all_data.append(pd.DataFrame([model_set.ravel(),insitu_set.ravel()]).T.rename(columns={0:'gldas',1:'insitu'}))

            except IndexError or ValueError:
                continue

        #Metrics
        set_all_metrics = pd.concat(set_all_metrics,axis=0).reset_index().iloc[:,1:]
        all_metrics.append(set_all_metrics)

        #Data
        set_all_data = pd.concat(set_all_data,axis=0)
        all_data.append(set_all_data)

    #Metrics (CC, rmse, bias, pval)
    all_metrics = pd.concat(all_metrics,axis=0)
    CCs_all.append(all_metrics.R)
    rmses_all.append(all_metrics.iloc[:,1])
    bias_all.append(all_metrics.iloc[:,2])
    pvals_all.append(all_metrics.iloc[:,3])

    #Data ()
    all_data_concat.append(all_data)


titles = ['All Year (Annual)']
source_names = ['Average','0-10 cm','10-40 cm', '40-100 cm','100-200 cm']

#ALL DATA
for i in range(0,4):
    linear_plot(all_data_concat[0][i].gldas,all_data_concat[0][i].insitu,sm_names[i] + ' avg')
for i in range(0,4):
    linear_plot(all_data_concat[-1][i].gldas,all_data_concat[-1][i].insitu,sm_names[i] + ' 200cm') 

#For comparing upper soil layers (avg)
linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[1][i].gldas for i in range(0,len(all_data_concat[0]))]))),
            np.array(list(itertools.chain.from_iterable([all_data_concat[1][i].insitu for i in range(0,len(all_data_concat[0]))]))),'Upper SM [0-10cm] {}'.format(freq))

#For comparing lower soil layers
linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[-1][i].gldas for i in range(0,len(all_data_concat[-1]))]))),
            np.array(list(itertools.chain.from_iterable([all_data_concat[-1][i].insitu for i in range(0,len(all_data_concat[-1]))]))),'Lower SM [100-200cm] {}'.format(freq))






#############################
#MONTHLY ERA5-L
#############################

plt.rcParams["font.family"] = "Times New Roman"
def linear_plot(model,insitu,title,savepath):
    try:
        nanmask = ~np.isnan(model) & ~np.isnan(insitu)
        model = model[nanmask]
        insitu = insitu[nanmask]

        slope, intercept, r_value, p_value, std_err = stats.linregress(insitu,model)
        modeled_y = intercept + slope*insitu

        bias = np.mean(modeled_y - insitu)
        ub_rmse = np.sqrt(np.sum((modeled_y - insitu)**2)/(len(insitu)-1))
        std_err

        # Calculate the point density
        xy = np.vstack([insitu,model])
        z = gaussian_kde(xy)(xy)

        fig,ax = plt.subplots()
        ax.scatter(insitu, model, c=z, s=1)
        ax.plot(insitu, intercept + slope*insitu, label='r: {}'.format(round(r_value,3)),color='r')
        ax.set_ylabel('ERA5-L VWC (m3/m3)')
        ax.set_xlabel('In situ VWC (m3/m3)')
        ax.legend()
        ax.set_ylim(0,0.8)
        ax.set_xlim(0,0.8)
        ax.plot([-1,1],[-1,1],'--',color='black')
        ax.text(0.6,.30,s="ubRMSE: {}".format(round(ub_rmse,4)), fontsize=10, ha="left", va="top")
        ax.text(0.6,0.25,s="bias: {}".format(round(bias,4)), fontsize=10, ha="left", va="top")
        ax.text(0.6,0.35,s="p-val: {}".format(round(p_value,4)), fontsize=10, ha="left", va="top")
        #ax.text(0.6,0.35,s="p-val < 0.001", fontsize=10, ha="left", va="top")
        ax.set_title(title,weight='bold')
        plt.savefig(r'\SM\{}\{}.png'.format(savepath,title))
        #plt.savefig(r'\SM\{}\{}.pdf'.format(savepath,title))

        print(title,round(r_value,3))

    except ValueError:
        print('error {}'.format(title))


main = r'\ERA5\MONTHLY_SM\validations'
paths = sorted(['\\ISMN','\\Ma2020','\\HiWAT','\\Zhao2021'])
set_paths = [sorted(glob.glob(main+path+'\*.csv')) for path in paths]

era5_sets = [[pd.read_csv(file) for file in set_path] for set_path in set_paths]
[[set.set_index(pd.to_datetime(set.time),inplace=True) for set in sets] for sets in era5_sets]

era5_avg = [[set.iloc[:,1].resample('1M').mean() for set in sets] for sets in era5_sets]
era5_swv1 = [[set.iloc[:,2].resample('1M').mean() for set in sets] for sets in era5_sets]
era5_swv2 = [[set.iloc[:,3].resample('1M').mean() for set in sets] for sets in era5_sets]
era5_swv3 = [[set.iloc[:,4].resample('1M').mean() for set in sets] for sets in era5_sets]
era5_swv4 = [[set.iloc[:,5].resample('1M').mean() for set in sets] for sets in era5_sets]

#Index 0 for average, -1 for deepest layer
insitu_hiwat_top = [(site.resample('1M').mean()/100).iloc[:,0] for site in hiwat_soil_moist_annual]
insitu_ma_top = [(site.resample('1M').mean()).iloc[:,0] for site in ma_soil_vwc]
insitu_zhao_top = [(site.resample('1M').mean()).iloc[:,0] for site in zhao_insitu[1]]
insitu_ismn_top = [(site.resample('1M').mean()) for site in mm_sm[0]]

insitu_hiwat_low = [(site.resample('1M').mean()/100).iloc[:,-1] for site in hiwat_soil_moist_annual]
insitu_ma_low = [(site.resample('1M').mean()).iloc[:,-1] for site in ma_soil_vwc]
insitu_zhao_low = [(site.resample('1M').mean()).iloc[:,-1] for site in zhao_insitu[1]]
insitu_ismn_low = [(site.resample('1M').mean()) for site in mm_sm[-1]]

insitu_sm_top = [insitu_hiwat_top,insitu_ismn_top,insitu_ma_top,insitu_zhao_top]
insitu_sm_low = [insitu_hiwat_low,insitu_ismn_low,insitu_ma_low,insitu_zhao_low]
sm_names = sorted(['HiWAT','ISMN','Ma 2020','Zhao 2021'])


#############################
#Select Depth

insitu_sm =insitu_sm_low
insitu_sm =insitu_sm_top

freq = 'Monthly Validation'
#############################


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

all_data_concat = []
monsoon_data_concat = []
dry_data_concat = []

for era5,sm_depth in zip([era5_avg,era5_swv1,era5_swv2,era5_swv3,era5_swv4],['Average','7 cm','28 cm', '100 cm','289 cm']):
    merged_ids = [[idx1.index.intersection(idx2.index) for idx1,idx2 in zip(sm_set,era5_set)] for sm_set,era5_set in zip(insitu_sm,era5)]

    all_metrics = []
    monsoon_metrics = []
    dry_metrics = []

    all_data = []
    monsoon_data = []
    dry_data = []

    for sm_set,era5_set,merged_id_set,set_name in zip(insitu_sm,era5,merged_ids,sm_names):
            
        set_all_metrics = []
        set_monsoon_metrics = []
        set_dry_metrics = []
        
        set_all_data = []
        set_monsoon_data = []
        set_dry_data = []

        for i in range(0,len(sm_set)):
            try:

                model_set = (era5_set[i].loc[merged_id_set[i]].ravel()) #all
                model_monsoon_set = (era5_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([6,7,8])]].ravel()) #monsoon
                model_dry_set = (era5_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([11,12,1,2])]].ravel()) #dry

                insitu_set = (sm_set[i].loc[merged_id_set[i]].ravel()) #all
                insitu_monsoon_set = (sm_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([6,7,8])]].ravel()) #monsoon
                insitu_dry_set = (sm_set[i].loc[merged_id_set[i][merged_id_set[i].month.isin([11,12,1,2])]].ravel()) #dry

                set_all_metrics.append(linear_stats(model_set,insitu_set,set_name+' {} all'.format(sm_depth)))
                set_monsoon_metrics.append(linear_stats(model_monsoon_set,insitu_monsoon_set,set_name+' {} monsoon'.format(sm_depth)))
                set_dry_metrics.append(linear_stats(model_dry_set,insitu_dry_set,set_name+' {} dry '.format(sm_depth)))

                set_all_data.append(pd.DataFrame([model_set.ravel(),insitu_set.ravel()]).T.rename(columns={0:'era5',1:'insitu'}))
                set_monsoon_data.append(pd.DataFrame([model_monsoon_set.ravel(),insitu_monsoon_set.ravel()]).T.rename(columns={0:'era5',1:'insitu'}))
                set_dry_data.append(pd.DataFrame([model_dry_set.ravel(),insitu_dry_set.ravel()]).T.rename(columns={0:'era5',1:'insitu'}))

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

titles = ['All Year','Monsoon Months','Winter Months']
source_names = ['Average','7 cm','28 cm', '100 cm','289 cm']

#ALL DATA
#average
for i in range(0,4):
    linear_plot(all_data_concat[0][i].era5,all_data_concat[0][i].insitu,sm_names[i] + ' avg')
#289 cm
for i in range(0,4):
    linear_plot(all_data_concat[-1][i].era5,all_data_concat[-1][i].insitu,sm_names[i] + ' 289cm') 

savepath='ERA5-L'
#For comparing upper soil layers (avg)
linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[1][i].era5 for i in range(0,len(all_data_concat[0]))]))),
            np.array(list(itertools.chain.from_iterable([all_data_concat[1][i].insitu for i in range(0,len(all_data_concat[0]))]))),'Upper SM [0-7cm]',savepath)

linear_plot(np.array(list(itertools.chain.from_iterable([monsoon_data_concat[1][i].era5 for i in range(0,len(all_data_concat[0]))]))),
            np.array(list(itertools.chain.from_iterable([monsoon_data_concat[1][i].insitu for i in range(0,len(all_data_concat[0]))]))),'Upper SM [0-7cm] (Monsoon)',savepath)

linear_plot(np.array(list(itertools.chain.from_iterable([dry_data_concat[1][i].era5 for i in range(0,len(all_data_concat[0]))]))),
            np.array(list(itertools.chain.from_iterable([dry_data_concat[1][i].insitu for i in range(0,len(all_data_concat[0]))]))),'Upper SM [0-7cm] (Winter)',savepath)

savepath='ERA5-L'
#For comparing lower soil layers
linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[-1][i].era5 for i in range(0,len(all_data_concat[-1]))]))),
            np.array(list(itertools.chain.from_iterable([all_data_concat[-1][i].insitu for i in range(0,len(all_data_concat[-1]))]))),'Lower SM [100-289cm]',savepath)

linear_plot(np.array(list(itertools.chain.from_iterable([monsoon_data_concat[-1][i].era5 for i in range(0,len(all_data_concat[-1]))]))),
            np.array(list(itertools.chain.from_iterable([monsoon_data_concat[-1][i].insitu for i in range(0,len(all_data_concat[-1]))]))),'Lower SM [100-289cm] (Monsoon)',savepath)

linear_plot(np.array(list(itertools.chain.from_iterable([dry_data_concat[-1][i].era5 for i in range(0,len(all_data_concat[-1]))]))),
            np.array(list(itertools.chain.from_iterable([dry_data_concat[-1][i].insitu for i in range(0,len(all_data_concat[-1]))]))),'Lower SM [100-289cm] (Winter)',savepath)



#############################
#ANNUAL
#############################
main = r'\ERA5\MONTHLY_SM\validations'
paths = sorted(['\\ISMN','\\Ma2020','\\HiWAT','\\Zhao2021'])
set_paths = [sorted(glob.glob(main+path+'\*.csv')) for path in paths]

era5_sets = [[pd.read_csv(file) for file in set_path] for set_path in set_paths]
[[set.set_index(pd.to_datetime(set.time),inplace=True) for set in sets] for sets in era5_sets]

era5_avg = [[set.iloc[:,1].resample('1Y').mean() for set in sets] for sets in era5_sets]
era5_swv1 = [[set.iloc[:,2].resample('1Y').mean() for set in sets] for sets in era5_sets]
era5_swv2 = [[set.iloc[:,3].resample('1Y').mean() for set in sets] for sets in era5_sets]
era5_swv3 = [[set.iloc[:,4].resample('1Y').mean() for set in sets] for sets in era5_sets]
era5_swv4 = [[set.iloc[:,5].resample('1Y').mean() for set in sets] for sets in era5_sets]

#Index 0 for average, -1 for deepest layer
insitu_hiwat_top = [(site.resample('1Y').mean()/100).iloc[:,0] for site in hiwat_soil_moist_annual]
insitu_ma_top = [(site.resample('1Y').mean()).iloc[:,0] for site in ma_soil_vwc]
insitu_zhao_top = [(site.resample('1Y').mean()).iloc[:,0] for site in zhao_insitu[1]]
insitu_ismn_top = [(site.resample('1Y').mean()) for site in mm_sm[0]]

insitu_hiwat_low = [(site.resample('1Y').mean()/100).iloc[:,-1] for site in hiwat_soil_moist_annual]
insitu_ma_low = [(site.resample('1Y').mean()).iloc[:,-1] for site in ma_soil_vwc]
insitu_zhao_low = [(site.resample('1Y').mean()).iloc[:,-1] for site in zhao_insitu[1]]
insitu_ismn_low = [(site.resample('1Y').mean()) for site in mm_sm[-1]]

insitu_sm_top = [insitu_hiwat_top,insitu_ismn_top,insitu_ma_top,insitu_zhao_top]
insitu_sm_low = [insitu_hiwat_low,insitu_ismn_low,insitu_ma_low,insitu_zhao_low]
sm_names = sorted(['HiWAT','ISMN','Ma 2020','Zhao 2021'])


#############################

insitu_sm =insitu_sm_low
insitu_sm =insitu_sm_top


freq = 'Annual Validation'
#############################


CCs_all = []
rmses_all = []
bias_all = []
pvals_all = []
all_data_concat = []

for era5,sm_depth in zip([era5_avg,era5_swv1,era5_swv2,era5_swv3,era5_swv4],['Average','7 cm','28 cm', '100 cm','289 cm']):
    merged_ids = [[idx1.index.intersection(idx2.index) for idx1,idx2 in zip(sm_set,era5_set)] for sm_set,era5_set in zip(insitu_sm,era5)]

    all_metrics = []
    monsoon_metrics = []
    dry_metrics = []

    all_data = []
    monsoon_data = []
    dry_data = []

    for sm_set,era5_set,merged_id_set,set_name in zip(insitu_sm,era5,merged_ids,sm_names):
            
        set_all_metrics = []
        set_monsoon_metrics = []
        set_dry_metrics = []
        
        set_all_data = []
        set_monsoon_data = []
        set_dry_data = []

        for i in range(0,len(sm_set)):
            try:

                model_set = (era5_set[i].loc[merged_id_set[i]].ravel()) #all
                insitu_set = (sm_set[i].loc[merged_id_set[i]].ravel()) #all
                set_all_metrics.append(linear_stats(model_set,insitu_set,set_name+' {} all'.format(sm_depth)))
                set_all_data.append(pd.DataFrame([model_set.ravel(),insitu_set.ravel()]).T.rename(columns={0:'era5',1:'insitu'}))

            except IndexError or ValueError:
                continue

        #Metrics
        set_all_metrics = pd.concat(set_all_metrics,axis=0).reset_index().iloc[:,1:]
        all_metrics.append(set_all_metrics)

        #Data
        set_all_data = pd.concat(set_all_data,axis=0)
        all_data.append(set_all_data)

    #Metrics (CC, rmse, bias, pval)
    all_metrics = pd.concat(all_metrics,axis=0)
    CCs_all.append(all_metrics.R)
    rmses_all.append(all_metrics.iloc[:,1])
    bias_all.append(all_metrics.iloc[:,2])
    pvals_all.append(all_metrics.iloc[:,3])

    #Data ()
    all_data_concat.append(all_data)


titles = ['All Year (Annual)']
source_names = ['Average','7 cm','28 cm', '100 cm','289 cm']


#ALL DATA
#average
for i in range(0,4):
    linear_plot(all_data_concat[0][i].era5,all_data_concat[0][i].insitu,sm_names[i] + ' avg')
#289 cm
for i in range(0,4):
    linear_plot(all_data_concat[-1][i].era5,all_data_concat[-1][i].insitu,sm_names[i] + ' 289cm') 


#For comparing upper soil layers (avg)
linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[1][i].era5 for i in range(0,len(all_data_concat[0]))]))),
            np.array(list(itertools.chain.from_iterable([all_data_concat[1][i].insitu for i in range(0,len(all_data_concat[0]))]))),'Upper SM [0-7cm] {}'.format(freq))

#For comparing lower soil layers
linear_plot(np.array(list(itertools.chain.from_iterable([all_data_concat[-1][i].era5 for i in range(0,len(all_data_concat[-1]))]))),
            np.array(list(itertools.chain.from_iterable([all_data_concat[-1][i].insitu for i in range(0,len(all_data_concat[-1]))]))),'Lower SM [100-289cm] {}'.format(freq))


