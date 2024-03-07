import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
from statsmodels.tsa.stattools import adfuller
import glob
from statsmodels.tsa.stattools import kpss
from rasterstats import zonal_stats
import scipy.stats as stats

def read_file(file):
    with rio.open(file) as src:
        return(src.read())

degree_sign = u"\N{DEGREE SIGN}"

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


#########################
#Annual
#########################

shpname = r'/MTRanges.shp'
regions = gpd.read_file(shpname).to_crs('EPSG:4326')
lst_affine = xr.open_rasterio(r'\2003_2016_MODMYD11AIRS.tif').transform

lakes = xr.open_mfdataset(r'\lake_mask_1km_nearest.tif').band_data[0]
glaciers = xr.open_mfdataset(r'\glacier_mask_1km_nearest.tif').band_data[0]

file_name = sorted(glob.glob(r"\MAST\*.tif"))
stats_all = [zonal_stats(regions,file,affine = lst_affine,stats='mean std',all_touched=True) for file in file_name]


regional_dfs = []
for mtnrange in (range(0,15)):
        mtnrange_means = [stats_all[year][mtnrange]['mean'] for year in range(0,14)]
        mtnrange_stdevs = [stats_all[year][mtnrange]['std'] for year in range(0,14)]

        df = pd.DataFrame([mtnrange_means,mtnrange_stdevs]).T
        df = df.set_index(i for i in range(2003,2017)).set_axis(['Mean','Stdev'], axis=1, inplace=False)
        regional_dfs.append(df)

means = [i.mean(axis=0)['Mean'] for i in regional_dfs]
sorted_order = np.argsort(means)

sorted_regional_dfs = [regional_dfs[i] for i in sorted_order[::-1]]
sortedregion = regions.Region[sorted_order][::-1]

##################
anomalies = [(sorted_regional_dfs[i]-sorted_regional_dfs[i].mean()) for i in range(0,15)]
residuals = [(sorted_regional_dfs[i].mean() - anomalies[i]) for i in range(0,15)]

#ADF
adf_stats = []
p_values = []
cv_1 = []
cv_5 = []
cv_10 = []
test_outcomes = []
for mtnrange,name in zip(anomalies,sortedregion):
    result = adfuller(mtnrange['Mean'],regression='c')
    adf_stats.append(result[0])
    p_values.append(result[1])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    cv_1.append(result[4]['1%'])
    cv_5.append(result[4]['5%'])
    cv_10.append(result[4]['5%']) 
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[0] > result[4]["5%"]:
        print ("Failed to Reject Ho - Time Series is Non-Stationary in {}\n".format(name))
        test_outcomes.append('Non-Stationary (failed to reject H0)')
    else:
        print ("Reject Ho - Time Series is Stationary in {}\n".format(name))
        test_outcomes.append('Stationary (rejected H0)')

adf_df = pd.DataFrame([test_outcomes,adf_stats,p_values,cv_1,cv_5,cv_10],index=['outcome','adf_stat','pval','Cv 1%','CV 5%','CV 10%']).set_axis(sortedregion, axis=1, inplace=False).T

stationary_mtn_ranges = adf_df[adf_df.loc[:,'outcome']=='Stationary (rejected H0)']
nonnstationary_mtn_ranges = adf_df[adf_df.loc[:,'outcome']=='Non-Stationary (failed to reject H0)']

#sorted_regional_dfs 
#sortedregion



plt.rcParams["font.family"] = "Times New Roman"
degree_sign = u"\N{DEGREE SIGN}"
fig, ax = plt.subplots(figsize=(13,7))
for region,name in zip(sorted_regional_dfs,sortedregion):
    if  (name == 'TibetanPlateau'):
        ax.plot([i for i in range(2003,2017)],region['Mean'],'--',linewidth=2,color='#abcae4',label='{}'.format(name))
    elif  (name == 'BayanHar') | (name == 'Tanggula') | (name == 'Gandise') | (name == 'Altun') | (name == 'TienShan')| (name == 'Karakoram') | (name == 'Qilian') | (name == 'Himalaya') | (name == 'Kunlun'):
        ax.plot([i for i in range(2003,2017)],region['Mean'],linewidth=1.5,color='#3d84bf',label='{}'.format(name))
    elif (name == 'HissarAlay') | (name == 'Pamir') | (name == 'HinduKush'):
        ax.plot([i for i in range(2003,2017)],region['Mean'],linewidth=1,color='#abcae4',label='{}'.format(name))
    elif (name == 'Hengduan')| (name == 'Nyainqentanglha'):
        ax.plot([i for i in range(2003,2017)],region['Mean'],'--',linewidth=2,color='orangered',label='{}'.format(name))
ax.set_ylabel('Temperature ({0}C)'.format(degree_sign),weight='bold',fontsize=19)
ax.set_xlabel('Year',weight='bold',fontsize=19)
ax.set_ylim(-5,13)
ax.set_xlim(2003,2016)
ax.set_title('Regional MAGT-Ia Time Series',weight='bold',fontsize=20)
ax.legend(bbox_to_anchor=(1,0.85), fontsize=15)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
fig.tight_layout()
plt.savefig(r'\annual_ts.pdf')


plt.rcParams["font.family"] = "Times New Roman"
degree_sign = u"\N{DEGREE SIGN}"
fig, ax = plt.subplots(figsize=(13,7))
for region,name in zip(anomalies,sortedregion):

    if  (name == 'TibetanPlateau'):
        ax.plot([i for i in range(2003,2017)],region['Mean'],'--',linewidth=2,color='#abcae4',label='{}'.format(name))
    elif  (name == 'BayanHar') | (name == 'Tanggula') | (name == 'Gandise') | (name == 'Altun') | (name == 'TienShan')| (name == 'Karakoram') | (name == 'Qilian') | (name == 'Himalaya') | (name == 'Kunlun') :
        ax.plot([i for i in range(2003,2017)],region['Mean'],linewidth=1.5,color='#3d84bf',label='{}'.format(name))
    elif (name == 'HissarAlay') | (name == 'Pamir') | (name == 'HinduKush'):
        ax.plot([i for i in range(2003,2017)],region['Mean'],linewidth=1,color='#abcae4',label='{}'.format(name))
    elif (name == 'Hengduan')| (name == 'Nyainqentanglha'):
        ax.plot([i for i in range(2003,2017)],region['Mean'],'--',linewidth=2,color='orangered',label='{}'.format(name))

ax.set_ylabel('Temperature Anomaly ({0}C)'.format(degree_sign),weight='bold',fontsize=19)
ax.set_xlabel('Year',weight='bold',fontsize=19)
ax.set_ylim(-1.5,1.5)
ax.axhline(0, color='black',linewidth=1)
ax.set_title('Regional MAGT-Ia (2003 - 2016)',weight='bold',fontsize=20)
ax.legend(bbox_to_anchor=(1,0.85), fontsize=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.tight_layout()


#Spearman's Rank Correlation
correlations = []
p_values = []
for i in range(0,15):
    correlation, p_value = stats.spearmanr(regional_dfs[i].Mean, range(0,len(regional_dfs[i].Mean)))
    print("Spearman's correlation coefficient:", correlation)
    print("p-value:", p_value)
    print(regions.Region.iloc[i])
    correlations.append(correlation)
    p_values.append(p_value)

spearman_df = pd.DataFrame([correlations,p_values],index=['corr','p-val']).set_axis(regions.Region, axis=1, inplace=False).T


#KPSS Test
kpss_stats = []
p_values = []
lags=[]
cv_1 = []
cv_5 = []
cv_10 = []
cv_25=[]
test_outcomes = []
for i,name in zip(range(0,15),regions.Region):
    kpsstest = kpss(anomalies[i]['Mean'], regression='c',nlags=0)
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value

    kpss_stats.append(kpss_output[0])
    p_values.append(kpss_output[1])
    lags.append(kpss_output[2])
    print('KPSS Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    cv_10.append(kpss_output[3])
    cv_5.append(kpss_output[4])
    cv_25.append(kpss_output[5])
    cv_1.append(kpss_output[6]) 
    if kpss_output[0] < kpss_output[4]:
        print ("Failed to Reject Ho - Time Series is Stationary in {}\n".format(name))
        test_outcomes.append('Stationary (rejected H0)')
    else:
        print ("Reject Ho - Time Series is Non-Stationary in {}\n".format(name))
        test_outcomes.append('Non-Stationary (failed to reject H0)')

kpss_df = pd.DataFrame([test_outcomes,kpss_stats,lags,p_values,cv_1,cv_5,cv_10],index=['outcome','kpss_stat','lags','pval','Cv 1%','CV 5%','CV 10%']).set_axis(regions.Region, axis=1, inplace=False).T