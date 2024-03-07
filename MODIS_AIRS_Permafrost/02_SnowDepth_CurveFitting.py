#Robin Kim (kk7xv@virginia.edu), 07-14-21
#Curve Fitting of Smith & Riseborough (2002) nf parameters

from scipy.optimize import curve_fit
from scipy import special
import csv
import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams["font.family"] = "Times New Roman"
degree_sign = u"\N{DEGREE SIGN}"

nf_points_dir = r'\csv\nf_SmithRiseborough'
nf_points_files = sorted(glob.glob(nf_points_dir+"/*.csv"))

#All (double_exponential)
nf_5 = pd.read_csv(nf_points_files[8],delimiter=',', header=None)
nf_2 = pd.read_csv(nf_points_files[7],delimiter=',', header=None)
nf_0 = pd.read_csv(nf_points_files[6],delimiter=',', header=None)
nf_n2 = pd.read_csv(nf_points_files[2],delimiter=',', header=None)
nf_n4 = pd.read_csv(nf_points_files[3],delimiter=',', header=None)
nf_n6 = pd.read_csv(nf_points_files[4],delimiter=',', header=None)
nf_n8 = pd.read_csv(nf_points_files[5],delimiter=',', header=None)
nf_n10 = pd.read_csv(nf_points_files[0],delimiter=',', header=None)
nf_n12 = pd.read_csv(nf_points_files[1],delimiter=',', header=None)


#Define exponential functions:
def double_exponential(x,a,b,c,d):
    return np.exp(a*np.exp(b*x)) + np.exp(c*np.exp(d*x))
def double_exponential_single(x,a,b):
    return np.exp(a*np.exp(b*x))
def exponential(x,a,b):
    return (a*np.exp(b*x))
def linear(x,m,b):
    return(m*x + b)

nfs = [nf_5, nf_2, nf_0, nf_n2, nf_n4, nf_n6, nf_n8, nf_n10, nf_n12]

params = []
r2s = []
maat = [5,2,0,-2,-4,-6,-8,-10,-12]


for nf,t in zip(nfs,maat):

    pars,cov = curve_fit(f=double_exponential, xdata = nf[0], ydata=nf[1],
    p0 = [0,0,0,0], bounds=(-np.inf, np.inf))

    ss_res = np.sum((nf[1] - double_exponential(nf[0],*pars)) ** 2)
    ss_tot = np.sum((nf[1] -np.mean(nf[1])) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    r2s.append(r2)

    params.append(pars)


params = pd.DataFrame(data=params).rename(columns={0:'a',1:'b',2:'c',3:'d'})
params['MAAT'] = maat
params['R2'] = r2s

#Fitted Curves
fig = plt.figure(figsize=(6,4))
ax = fig.add_axes([0,0,1,1])
#ax.set_yscale('log')
limits = [0,1,2,3,4,5,6,7,8]
maats = [5,2,0,-2,-4,-6,-8,-10,-12]
for maat,limit,i in zip(maats,limits,range(0,len(maats))):
    x = np.arange(0,1.21,0.01)
    nfs = np.exp(params.loc[limit][0]*np.exp(params.loc[limit][1]*x)) + np.exp(params.loc[limit][2]*np.exp(params.loc[limit][3]*x))
    ax.plot(x,nfs,label='{}'.format(maat))
    print(nfs[0])
ax.set_ylabel('nf',weight='bold')
ax.set_xlabel('Snow Depth (m)',weight='bold')
ax.set_title('Snow Depth Factor for MAATs',weight='bold')
ax.set_xlim(0,1.2)
ax.set_ylim(0,1)
ax.grid()
fig.legend(loc='upper right',borderaxespad=3, fontsize=12)
plt.savefig(r'\fitted_curves.png',bbox_inches='tight',dpi = 150)



#Params: a,b,c,d
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1,figsize=(8, 12))
plt.rcParams.update({'font.size': 16})

parameter = 'a'
#ax1 = fig.add_axes([0,0,1,1])
ax1.set_ylabel('Parameter {}'.format(parameter))
ax1.set_xlabel('MAAT ({}C)'.format(degree_sign))
ax1.scatter(maat,params['{}'.format(parameter)],s=40,color='black')

parameter = 'b'
#ax2 = fig.add_axes([0,0,1,1])
ax2.set_ylabel('Parameter {}'.format(parameter))
ax2.set_xlabel('MAAT ({}C)'.format(degree_sign))
ax2.scatter(maat,params['{}'.format(parameter)],s=40,color='black')

parameter = 'c'
#ax3 = fig.add_axes([0,0,1,1])
ax3.set_ylabel('Parameter {}'.format(parameter))
ax3.set_xlabel('MAAT ({}C)'.format(degree_sign))
ax3.scatter(maat,params['{}'.format(parameter)],s=40,color='black')
parameter = 'd'
#ax4 = fig.add_axes([0,0,1,1])
ax4.set_ylabel('Parameter {}'.format(parameter))
ax4.set_xlabel('MAAT ({}C)'.format(degree_sign))
ax4.scatter(maat,params['{}'.format(parameter)],s=40,color='black')
fig.tight_layout()
