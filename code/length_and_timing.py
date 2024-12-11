"""
Analysis of atmosphere-surface interactions and feedbacks / Hyytiälä 2024
analysis scripts

@author: Anni Karvonen (anni.karvonen@helsinki.fi)

Plotting script for turnover lengths and timings.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import datetime as dt
from datetime import timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates


fig_path = 'KVJ/image/'
data_in = 'KVJ/data/data_in/'
data_out = 'KVJ/data/data_out/'

#get data
dataframe1 = pd.read_pickle(data_out +"wpt_kuivajarvi.pkl")
dataframe1=dataframe1.reset_index()

#get spring turnovers
s2013=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2013, 4, 27)) & (dataframe1['timestamp'] < dt.datetime(2013, 5, 8))]
s2014=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2014, 3, 26)) & (dataframe1['timestamp'] < dt.datetime(2014, 5, 14))]
s2015=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2015, 4, 20)) & (dataframe1['timestamp'] < dt.datetime(2015, 5, 18))]
s2016=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2016, 4, 7)) & (dataframe1['timestamp'] < dt.datetime(2016, 5, 2))]
s2017=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2017, 3, 29)) & (dataframe1['timestamp'] < dt.datetime(2017, 5, 19))]
s2018=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2018, 4, 23)) & (dataframe1['timestamp'] < dt.datetime(2018, 5, 7))]
s2019=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2019, 4, 23)) & (dataframe1['timestamp'] < dt.datetime(2019, 5, 12))]
s2021=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2021, 4, 20)) & (dataframe1['timestamp'] < dt.datetime(2021, 5, 12))]
s2022=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2022, 5, 7)) & (dataframe1['timestamp'] < dt.datetime(2022, 5, 19))]
s2023=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2023, 4, 21)) & (dataframe1['timestamp'] < dt.datetime(2023, 5, 7))]

#get autumn turnovers
a2012=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2012, 10, 6)) & (dataframe1['timestamp'] < dt.datetime(2012, 11, 28))]
a2013=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2013, 10, 12)) & (dataframe1['timestamp'] < dt.datetime(2013, 11, 25))]
a2014=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2014, 9, 27)) & (dataframe1['timestamp'] < dt.datetime(2014, 11, 30))]
a2015=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2015, 10, 3)) & (dataframe1['timestamp'] < dt.datetime(2015, 11, 28))]
a2016=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2016, 10, 6)) & (dataframe1['timestamp'] < dt.datetime(2016, 11, 28))]
a2017=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2017, 10, 3)) & (dataframe1['timestamp'] < dt.datetime(2017, 12, 14))]
a2018=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2018, 10, 3)) & (dataframe1['timestamp'] < dt.datetime(2018, 11, 25))]
a2021=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2021, 9, 24)) & (dataframe1['timestamp'] < dt.datetime(2021, 11, 23))]
a2022=dataframe1.loc[(dataframe1['timestamp'] >= dt.datetime(2022, 10, 7)) & (dataframe1['timestamp'] < dt.datetime(2022, 11, 23))]

#set turnover lengths for plotting
s2013['days']=11
s2014['days']=50.5
s2015['days']=28
s2016['days']=25
s2017['days']=51
s2018['days']=14
s2019['days']=19
s2021['days']=22
s2022['days']=12
s2023['days']=16

a2012['days']=53
a2013['days']=44
a2014['days']=64
a2015['days']=56
a2016['days']=52.5
a2017['days']=72
a2018['days']=53.5
a2021['days']=60
a2022['days']=47

#calculating coefficient of variation
spring_list=[11,51,28,25,51,14,19,22,12,16]
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
print(cv(spring_list))

autumn_list=[53,44,64,56,53,72,53,60,47]
print(cv(autumn_list))

#getting start dates for plotting
start_date_2013s = np.min(s2013['timestamp'])  # First date in the time column
start_date_2014s = np.min(s2014['timestamp'])  # First date in the time column
start_date_2015s = np.min(s2015['timestamp'])  # First date in the time column
start_date_2016s = np.min(s2016['timestamp'])  # First date in the time column
start_date_2017s = np.min(s2017['timestamp'])  # First date in the time column
start_date_2018s = np.min(s2018['timestamp'])  # First date in the time column
start_date_2019s = np.min(s2019['timestamp'])  # First date in the time column
start_date_2021s = np.min(s2021['timestamp'])  # First date in the time column
start_date_2022s = np.min(s2022['timestamp'])  # First date in the time column
start_date_2023s = np.min(s2023['timestamp'])  # First date in the time column

start_date_2012a = np.min(a2012['timestamp'])  # First date in the time column
start_date_2013a = np.min(a2013['timestamp'])  # First date in the time column
start_date_2014a = np.min(a2014['timestamp'])  # First date in the time column
start_date_2015a = np.min(a2015['timestamp'])  # First date in the time column
start_date_2016a = np.min(a2016['timestamp'])  # First date in the time column
start_date_2017a = np.min(a2017['timestamp'])  # First date in the time column
start_date_2018a = np.min(a2018['timestamp'])  # First date in the time column
start_date_2021a = np.min(a2021['timestamp'])  # First date in the time column
start_date_2022a = np.min(a2022['timestamp'])  # First date in the time column

#plotting start date and length
fig0,ax0 = plt.subplots(1,2,figsize=[13,8])
ax0[0].plot(start_date_2013s.replace(year=2020),11,marker='o',label='2013',markersize=12,color='#009e73')  # use a fixed year, e.g., 2020
ax0[0].plot(start_date_2014s.replace(year=2020),51,marker='o',label='2014',markersize=12,color='#009e73')
ax0[0].plot(start_date_2015s.replace(year=2020),28,marker='o',label='2015',markersize=12,color='#009e73')
ax0[0].plot(start_date_2016s.replace(year=2020),25,marker='o',label='2016',markersize=12,color='#009e73')
ax0[0].plot(start_date_2017s.replace(year=2020),51,marker='o',label='2017',markersize=12,color='#009e73')
ax0[0].plot(start_date_2018s.replace(year=2020),14,marker='o',label='2018',markersize=12,color='#009e73')
ax0[0].plot(start_date_2019s.replace(year=2020),19,marker='o',label='2019',markersize=12,color='#009e73')
ax0[0].plot(start_date_2021s.replace(year=2020),22,marker='o',label='2021',markersize=12,color='#009e73')
ax0[0].plot(start_date_2022s.replace(year=2020),12,marker='o',label='2022',markersize=12,color='#009e73')
ax0[0].plot(start_date_2023s.replace(year=2020),16,marker='o',label='2023',markersize=12,color='#009e73')
# Format the x-axis to show only the month and day
ax0[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax0[0].set_title('Spring',fontsize=20)
ax0[0].set_xlabel('Starting date',fontsize=18)
ax0[0].set_ylabel('Length (days)',fontsize=18)
ax0[0].tick_params(axis='y', labelsize=20)
ax0[0].tick_params(axis='x', labelsize=20,labelrotation=25)
ax0[0].grid()
interval = 6
ax0[0].xaxis.set_major_locator(ticker.MultipleLocator(interval))

ax0[1].plot(start_date_2012a.replace(year=2020),53,marker='o',label='2012',markersize=12,color='#e69f00')  # use a fixed year, e.g., 2020
ax0[1].plot(start_date_2013a.replace(year=2020),44,marker='o',label='2013',markersize=12,color='#e69f00')
ax0[1].plot(start_date_2014a.replace(year=2020),64,marker='o',label='2014',markersize=12,color='#e69f00')
ax0[1].plot(start_date_2015a.replace(year=2020),56,marker='o',label='2015',markersize=12,color='#e69f00')
ax0[1].plot(start_date_2016a.replace(year=2020),53,marker='o',label='2016',markersize=12,color='#e69f00')
ax0[1].plot(start_date_2017a.replace(year=2020),72,marker='o',label='2017',markersize=12,color='#e69f00')
ax0[1].plot(start_date_2018a.replace(year=2020),53,marker='o',label='2018',markersize=12,color='#e69f00')
ax0[1].plot(start_date_2021a.replace(year=2020),60,marker='o',label='2021',markersize=12,color='#e69f00')
ax0[1].plot(start_date_2022a.replace(year=2020),47,marker='o',label='2022',markersize=12,color='#e69f00')
# Format the x-axis to show only the month and day
ax0[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax0[1].set_title('Autumn',fontsize=20)
ax0[1].set_xlabel('Starting date',fontsize=18)
ax0[1].set_ylabel('Length (days)',fontsize=18)
ax0[1].tick_params(axis='y', labelsize=20)
ax0[1].tick_params(axis='x', labelsize=20,labelrotation=25)
ax0[1].grid()
interval = 4
ax0[1].xaxis.set_major_locator(ticker.MultipleLocator(interval))
fig0.tight_layout()
plt.savefig(fig_path + 'turnover_lengths.png', dpi=400)
plt.show()


#plotting time and length
fig, ax = plt.subplots(2,figsize=[12,9])
ax[0].plot([d.replace(year=2020) for d in s2013['timestamp']], s2013['days'],label='2013',color='#009e73',marker='o')  # use a fixed year, e.g., 2020
ax[0].plot([d.replace(year=2020) for d in s2014['timestamp']], s2014['days'],marker='o',label='2014',color='#009e73')
ax[0].plot([d.replace(year=2020) for d in s2015['timestamp']], s2015['days'],marker='o',label='2015',color='#009e73')
ax[0].plot([d.replace(year=2020) for d in s2016['timestamp']], s2016['days'],marker='o',label='2016',color='#009e73')
ax[0].plot([d.replace(year=2020) for d in s2017['timestamp']], s2017['days'],marker='o',label='2017',color='#009e73')
ax[0].plot([d.replace(year=2020) for d in s2018['timestamp']], s2018['days'],marker='o',label='2018',color='#009e73')
ax[0].plot([d.replace(year=2020) for d in s2019['timestamp']], s2019['days'],marker='o',label='2019',color='#009e73')
ax[0].plot([d.replace(year=2020) for d in s2021['timestamp']], s2021['days'],marker='o',label='2021',color='#009e73')
ax[0].plot([d.replace(year=2020) for d in s2022['timestamp']], s2022['days'],marker='o',label='2022',color='#009e73')
ax[0].plot([d.replace(year=2020) for d in s2023['timestamp']], s2023['days'],marker='o',label='2023',color='#009e73')
# Format the x-axis to show only the month and day
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax[0].set_title('Spring',fontsize=20)
ax[0].set_xlabel('Date',fontsize=18)
ax[0].set_ylabel('Length of the \n turnover (days)',fontsize=18)
ax[0].tick_params(axis='x', labelsize=20,labelrotation=25)
ax[0].tick_params(axis='y', labelsize=20)
ax[0].grid()
interval = 5
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(interval))


ax[1].plot([d.replace(year=2020) for d in a2012['timestamp']], a2012['days'],marker='o',label='2012',color='#e69f00')  # use a fixed year, e.g., 2020
ax[1].plot([d.replace(year=2020) for d in a2013['timestamp']], a2013['days'],marker='o',label='2013',color='#e69f00')
ax[1].plot([d.replace(year=2020) for d in a2014['timestamp']], a2014['days'],marker='o',label='2014',color='#e69f00')
ax[1].plot([d.replace(year=2020) for d in a2015['timestamp']], a2015['days'],marker='o',label='2015',color='#e69f00')
ax[1].plot([d.replace(year=2020) for d in a2016['timestamp']], a2016['days'],marker='o',label='2016',color='#e69f00')
ax[1].plot([d.replace(year=2020) for d in a2017['timestamp']], a2017['days'],marker='o',label='2017',color='#e69f00')
ax[1].plot([d.replace(year=2020) for d in a2018['timestamp']], a2018['days'],marker='o',label='2018',color='#e69f00')
ax[1].plot([d.replace(year=2020) for d in a2021['timestamp']], a2021['days'],marker='o',label='2021',color='#e69f00')
ax[1].plot([d.replace(year=2020) for d in a2022['timestamp']], a2022['days'],marker='o',label='2022',color='#e69f00')
# Format the x-axis to show only the month and day
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax[1].set_title('Autumn',fontsize=20)
ax[1].set_xlabel('Date',fontsize=18)
ax[1].set_ylabel('Length of the \n turnover (days)',fontsize=18)
ax[1].tick_params(axis='x', labelsize=20,labelrotation=25)
ax[1].tick_params(axis='y', labelsize=20)
ax[1].grid()
interval = 10
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(interval))
fig.tight_layout()
plt.savefig(fig_path + 'turnover_timings_scatter.png', dpi=400)
plt.show()
