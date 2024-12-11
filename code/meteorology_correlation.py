import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import datetime as dt
from datetime import timedelta
from scipy import stats

#get temperature data
dataframe1 = pd.read_pickle("wpt_kuivajarvi.pkl")
dataframe1=dataframe1.reset_index()

dataframe1['timestamp']=pd.to_datetime(dataframe1['timestamp'])

#get radiation data
dataframe_radiation=pd.read_pickle("data_radiation_data.pkl")
dataframe_radiation=dataframe_radiation.reset_index()

#mean temperature of turnovers
mean2013s=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2013, 4, 27)) & (dataframe1['timestamp'] < dt.datetime(2013, 5, 8)), 'KVJ_EDDY.av_t'].mean()
mean2015s=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2015, 4, 20)) & (dataframe1['timestamp'] < dt.datetime(2015, 5, 18)), 'KVJ_EDDY.av_t'].mean()
mean2016s=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2016, 4, 7)) & (dataframe1['timestamp'] < dt.datetime(2016, 5, 2)), 'KVJ_EDDY.av_t'].mean()
mean2017s=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2017, 3, 29)) & (dataframe1['timestamp'] < dt.datetime(2017, 5, 19)), 'KVJ_EDDY.av_t'].mean()
mean2018s=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2018, 4, 23)) & (dataframe1['timestamp'] < dt.datetime(2018, 5, 7)), 'KVJ_EDDY.av_t'].mean()
mean2019s=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2019, 4, 23)) & (dataframe1['timestamp'] < dt.datetime(2019, 5, 12)), 'KVJ_EDDY.av_t'].mean()
mean2021s=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2021, 4, 20)) & (dataframe1['timestamp'] < dt.datetime(2021, 5, 12)), 'KVJ_EDDY.av_t'].mean()
mean2022s=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2022, 5, 7)) & (dataframe1['timestamp'] < dt.datetime(2022, 5, 19)), 'KVJ_EDDY.av_t'].mean()
mean2023s=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2023, 4, 21)) & (dataframe1['timestamp'] < dt.datetime(2023, 5, 7)), 'KVJ_EDDY.av_t'].mean()

mean2012a=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2012, 10, 6)) & (dataframe1['timestamp'] < dt.datetime(2012, 11, 28)), 'KVJ_EDDY.av_t'].mean()
mean2013a=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2013, 10, 12)) & (dataframe1['timestamp'] < dt.datetime(2013, 11, 25)), 'KVJ_EDDY.av_t'].mean()
mean2014a=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2014, 9, 27)) & (dataframe1['timestamp'] < dt.datetime(2014, 11, 30)), 'KVJ_EDDY.av_t'].mean()
mean2015a=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2015, 10, 3)) & (dataframe1['timestamp'] < dt.datetime(2015, 11, 28)), 'KVJ_EDDY.av_t'].mean()
mean2016a=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2016, 10, 6)) & (dataframe1['timestamp'] < dt.datetime(2016, 11, 28)), 'KVJ_EDDY.av_t'].mean()
mean2017a=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2017, 10, 3)) & (dataframe1['timestamp'] < dt.datetime(2017, 12, 14)), 'KVJ_EDDY.av_t'].mean()
mean2018a=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2018, 10, 3)) & (dataframe1['timestamp'] < dt.datetime(2018, 11, 25)), 'KVJ_EDDY.av_t'].mean()
mean2022a=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2022, 10, 7)) & (dataframe1['timestamp'] < dt.datetime(2022, 11, 23)), 'KVJ_EDDY.av_t'].mean()

#choose turnovers from radiation data
s2013rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2013, 4, 27)) & (dataframe_radiation['timestamp'] < dt.datetime(2013, 5, 8))]
s2015rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2015, 4, 20)) & (dataframe_radiation['timestamp'] < dt.datetime(2015, 5, 18))]
s2016rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2016, 4, 7)) & (dataframe_radiation['timestamp'] < dt.datetime(2016, 5, 2))]
s2017rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2017, 3, 29)) & (dataframe_radiation['timestamp'] < dt.datetime(2017, 5, 19))]
s2018rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2018, 4, 23)) & (dataframe_radiation['timestamp'] < dt.datetime(2018, 5, 7))]
s2019rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2019, 4, 23)) & (dataframe_radiation['timestamp'] < dt.datetime(2019, 5, 12))]
s2021rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2021, 4, 20)) & (dataframe_radiation['timestamp'] < dt.datetime(2021, 5, 12))]
s2022rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2022, 5, 7)) & (dataframe_radiation['timestamp'] < dt.datetime(2022, 5, 19))]
s2023rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2023, 4, 21)) & (dataframe_radiation['timestamp'] < dt.datetime(2023, 5, 7))]

a2012rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2012, 10, 6)) & (dataframe_radiation['timestamp'] < dt.datetime(2012, 11, 28))]
a2013rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2013, 10, 12)) & (dataframe_radiation['timestamp'] < dt.datetime(2013, 11, 25))]
a2014rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2014, 9, 27)) & (dataframe_radiation['timestamp'] < dt.datetime(2014, 11, 30))]
a2015rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2015, 10, 3)) & (dataframe_radiation['timestamp'] < dt.datetime(2015, 11, 28))]
a2016rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2016, 10, 6)) & (dataframe_radiation['timestamp'] < dt.datetime(2016, 11, 28))]
a2017rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2017, 10, 3)) & (dataframe_radiation['timestamp'] < dt.datetime(2017, 12, 14))]
a2018rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2018, 10, 3)) & (dataframe_radiation['timestamp'] < dt.datetime(2018, 11, 25))]
a2022rad=dataframe_radiation.loc[(dataframe_radiation['timestamp'] >= dt.datetime(2022, 10, 7)) & (dataframe_radiation['timestamp'] < dt.datetime(2022, 11, 23))]

#mean radiation of turnovers
s2013rad=s2013rad['KVJ_META.Glob'].mean()
s2015rad=s2015rad['KVJ_META.Glob'].mean()
s2016rad=s2016rad['KVJ_META.Glob'].mean()
s2017rad=s2017rad['KVJ_META.Glob'].mean()
s2018rad=s2018rad['KVJ_META.Glob'].mean()
s2019rad=s2019rad['KVJ_META.Glob'].mean()
s2021rad=s2021rad['KVJ_META.Glob'].mean()
s2022rad=s2022rad['KVJ_META.Glob'].mean()
s2023rad=s2023rad['KVJ_META.Glob'].mean()

a2012rad=a2012rad['KVJ_META.Glob'].mean()
a2013rad=a2013rad['KVJ_META.Glob'].mean()
a2014rad=a2014rad['KVJ_META.Glob'].mean()
a2015rad=a2015rad['KVJ_META.Glob'].mean()
a2016rad=a2016rad['KVJ_META.Glob'].mean()
a2017rad=a2017rad['KVJ_META.Glob'].mean()
a2018rad=a2018rad['KVJ_META.Glob'].mean()
a2022rad=a2022rad['KVJ_META.Glob'].mean()

#plot lengths and temperature and radiation and make regression lines
fig, ax =plt.subplots(2,2,figsize=(10,8))
name = "tab20"
cmap = plt.colormaps[name]  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list
x=np.array([11,28,25,51,14,19,22,12,16])
x2=np.array([11,28,25,14,19,22,12,16])
y=np.array([mean2013s,mean2015s,mean2016s,mean2017s,mean2018s,mean2019s,mean2021s,mean2022s,mean2023s])
y2=np.array([mean2013s,mean2015s,mean2016s,mean2018s,mean2019s,mean2021s,mean2022s,mean2023s])
#parametrit_1, cov_mat_1 = curve_fit(suora,x,y)
res = stats.linregress(x, y)
res2 = stats.linregress(x2, y2)

ax[0,0].scatter(11,mean2013s,s=39,color='#009e73')
ax[0,0].scatter(28,mean2015s,s=39,color='#009e73')
ax[0,0].scatter(25,mean2016s,s=39,color='#009e73')
ax[0,0].scatter(51,mean2017s,s=39,color='#009e73')
ax[0,0].scatter(14,mean2018s,s=39,color='#009e73')
ax[0,0].scatter(19,mean2019s,s=39,color='#009e73')
ax[0,0].scatter(22,mean2021s,s=39,color='#009e73')
ax[0,0].scatter(12,mean2022s,s=39,color='#009e73')
ax[0,0].scatter(16,mean2023s,s=39,color='#009e73')
ax[0,0].plot(x,res.intercept+res.slope*x,label=f"$R^2$: {res.rvalue**2:.5f} \n Slope: {res.slope:.5f} \n p-value: {res.pvalue:.5f}")
ax[0,0].plot(x2,res2.intercept+res.slope*x2,label=f"$R^2$: {res2.rvalue**2:.5f} \n Slope: {res2.slope:.5f} \n p-value: {res2.pvalue:.5f}")
ax[0,0].set_title("Spring",fontsize=18)
ax[0,0].set_xlabel('Length of the turnover (days)',fontsize=18)
ax[0,0].set_ylabel("Mean air temperature \n during the turnover ($\mathrm{\degree C}$)",fontsize=18)
ax[0,0].tick_params(axis='x', labelsize=20)
ax[0,0].tick_params(axis='y', labelsize=20)
ax[0,0].legend(fontsize=10)

x=np.array([53,44,64,56,53,72,53,47])
y=np.array([mean2012a,mean2013a,mean2014a,mean2015a,mean2016a,mean2017a,mean2018a,mean2022a])
res = stats.linregress(x, y)

ax[0,1].scatter(53,mean2012a,color='#e69f00')
ax[0,1].scatter(44,mean2013a,color='#e69f00')
ax[0,1].scatter(64,mean2014a,color='#e69f00')
ax[0,1].scatter(56,mean2015a,color='#e69f00')
ax[0,1].scatter(53,mean2016a,color='#e69f00')
ax[0,1].scatter(72,mean2017a,color='#e69f00')
ax[0,1].scatter(53,mean2018a,color='#e69f00')
ax[0,1].scatter(47,mean2022a,color='#e69f00')
ax[0,1].plot(x,res.intercept+res.slope*x,label=f"$R^2$: {res.rvalue**2:.5f} \n Slope: {res.slope:.5f} \n p-value: {res.pvalue:.5f}")
ax[0,1].set_title("Autumn",fontsize=18)
ax[0,1].set_xlabel('Length of the turnover (days)',fontsize=18)
ax[0,1].set_ylabel("Mean air temperature \n during the turnover ($\mathrm{\degree C}$)",fontsize=18)
ax[0,1].tick_params(axis='x', labelsize=20)
ax[0,1].tick_params(axis='y', labelsize=20)
ax[0,1].legend(fontsize=10)
#ax[1,0].set_prop_cycle(color=colors)
x=np.array([11,28,25,51,14,19,22,12,16])
y=np.array([s2013rad,s2015rad,s2016rad,s2017rad,s2018rad,s2019rad,s2021rad,s2022rad,s2023rad])
res = stats.linregress(x, y)

ax[1,0].scatter(11,s2013rad,color='#009e73')
ax[1,0].scatter(28,s2015rad,color='#009e73')
ax[1,0].scatter(25,s2016rad,color='#009e73')
ax[1,0].scatter(51,s2017rad,color='#009e73')
ax[1,0].scatter(14,s2018rad,color='#009e73')
ax[1,0].scatter(19,s2019rad,color='#009e73')
ax[1,0].scatter(22,s2021rad,color='#009e73')
ax[1,0].scatter(12,s2022rad,color='#009e73')
ax[1,0].scatter(16,s2023rad,color='#009e73')
ax[1,0].plot(x,res.intercept+res.slope*x,label=f"$R^2$: {res.rvalue**2:.5f} \n Slope: {res.slope:.5f} \n p-value: {res.pvalue:.5f}")
ax[1,0].set_title("Spring",fontsize=18)
ax[1,0].set_xlabel('Length of the turnover (days)',fontsize=18)
ax[1,0].set_ylabel("Mean global radiation \n during the turnover ($\mathrm{W m^{-2}}$)",fontsize=18)
ax[1,0].tick_params(axis='x', labelsize=20)
ax[1,0].tick_params(axis='y', labelsize=20)
ax[1,0].legend(fontsize=10)
#ax[1,1].set_axis_off()
x=np.array([53,44,64,56,53,72,53,47])
y=np.array([a2012rad,a2013rad,a2014rad,a2015rad,a2016rad,a2017rad,a2018rad,a2022rad])
res = stats.linregress(x, y)

ax[1,1].scatter(53,a2012rad,color='#e69f00')
ax[1,1].scatter(44,a2013rad,color='#e69f00')
ax[1,1].scatter(64,a2014rad,color='#e69f00')
ax[1,1].scatter(56,a2015rad,color='#e69f00')
ax[1,1].scatter(53,a2016rad,color='#e69f00')
ax[1,1].scatter(72,a2017rad,color='#e69f00')
ax[1,1].scatter(53,a2018rad,color='#e69f00')
ax[1,1].scatter(47,a2022rad,color='#e69f00')
ax[1,1].plot(x,res.intercept+res.slope*x,label=f"$R^2$: {res.rvalue**2:.5f} \n Slope: {res.slope:.5f} \n p-value: {res.pvalue:.5f}")
ax[1,1].set_title("Autumn",fontsize=18)
ax[1,1].set_xlabel('Length of the turnover (days)',fontsize=18)
ax[1,1].set_ylabel("Mean global radiation \n during the turnover ($\mathrm{W m^{-2}}$)",fontsize=18)
ax[1,1].tick_params(axis='x', labelsize=20)
ax[1,1].tick_params(axis='y', labelsize=20)
ax[1,1].legend(fontsize=10,loc='upper right')
fig.tight_layout()


plt.show()
