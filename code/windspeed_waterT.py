"""
Analysis of atmosphere-surface interactions and feedbacks / Hyytiälä 2024
analysis scripts

@author: Eevi Silvennoinen (eevi.silvennoinen@helsinki.fi)

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reading in data
wind_speed = pd.read_csv(rf"wpt_kuivajarvi.csv", parse_dates= ['timestamp'])
dates = pd.read_csv(rf"turnover_dates.csv", header=None, names= ['year','month','day','hour', 'minute'])
water = pd.read_csv(rf"Hyytiala/waterT_0.2m.csv")

water['datetime'] = pd.to_datetime(water['datetime'])

# creating one datetime column for turnover dates instead of multiple columns
dates['datetime'] = pd.to_datetime(dates[['year', 'month', 'day', 'hour', 'minute']])

#extracting date part from turnover dates
dates['date'] = dates['datetime'].dt.date

#extracting date, year, month and day from the wind speed data
wind_speed['date'] = wind_speed['timestamp'].dt.date
wind_speed['year'] = wind_speed['timestamp'].dt.year
wind_speed['month'] = wind_speed['timestamp'].dt.month
wind_speed['day'] = wind_speed['timestamp'].dt.day
wind_speed['month_day'] = wind_speed['timestamp'].dt.strftime('%m-%d')

water['date'] = water['datetime'].dt.date


# taking daily averages of wind speed for the whole dataset
average_U = wind_speed.groupby('date')['KVJ_EDDY.U'].mean().reset_index()

#removing invalid values (way too large) from the daily averages:
remove =[3651,3652,3653,3654,3655,3656,3657,3658,3659,3660,3661,3662,3663,3664]
avg_U =average_U.drop(remove)

# taking daily averages
avg_w = water.groupby('date')['values'].mean().reset_index()


# defining a function to get the turnover dates
def get_turnover_dates(dataframe):
    dates_in_str = pd.to_datetime(dataframe['date'].values).strftime("%d-%m")
    dates_to_doy = pd.to_datetime(dataframe.date.values).dayofyear
    ref_dates = pd.to_datetime(dates_to_doy-1, unit='D', origin=str(2024))
    #print(dates_to_doy)
    return dates_to_doy

'''
# saving daily averages to a new csv file
avg_U.to_csv('/home/skekeevi/Hyytiala/daily_avg_U.csv', index=False)
'''

########################################################################################


# exactly turnover dates 
U_A_2012x = avg_U[(avg_U['date'] >= pd.to_datetime('2012-10-06').date()) & (avg_U['date'] <= pd.to_datetime('2012-11-27').date())]
U_A_2013x = avg_U[(avg_U['date'] >= pd.to_datetime('2013-10-12').date()) & (avg_U['date'] <= pd.to_datetime('2013-11-24').date())]
U_A_2014x = avg_U[(avg_U['date'] >= pd.to_datetime('2014-09-27').date()) & (avg_U['date'] <= pd.to_datetime('2014-11-29').date())]
U_A_2017x = avg_U[(avg_U['date'] >= pd.to_datetime('2017-10-03').date()) & (avg_U['date'] <= pd.to_datetime('2017-12-13').date())]
U_A_2015x = avg_U[(avg_U['date'] >= pd.to_datetime('2015-10-03').date()) & (avg_U['date'] <= pd.to_datetime('2015-11-27').date())]
U_A_2016x = avg_U[(avg_U['date'] >= pd.to_datetime('2016-10-07').date()) & (avg_U['date'] <= pd.to_datetime('2016-11-27').date())]
U_A_2018x = avg_U[(avg_U['date'] >= pd.to_datetime('2018-10-03').date()) & (avg_U['date'] <= pd.to_datetime('2018-11-24').date())]
# NOT ENOUGH DATA U_2021 = avg_U[(avg_U['date'] >= pd.to_datetime('2021-09-24').date()) & (avg_U['date'] <= pd.to_datetime('2021-11-22').date())]
U_A_2022x = avg_U[(avg_U['date'] >= pd.to_datetime('2022-10-07').date()) & (avg_U['date'] <= pd.to_datetime('2022-11-22').date())]

# three weeks after turnover period
U_A_2013plus = avg_U[(avg_U['date'] >= pd.to_datetime('2013-11-24').date()) & (avg_U['date'] <= pd.to_datetime('2013-12-15').date())]
U_A_2014plus = avg_U[(avg_U['date'] >= pd.to_datetime('2014-11-29').date()) & (avg_U['date'] <= pd.to_datetime('2014-12-20').date())]
U_A_2017plus = avg_U[(avg_U['date'] >= pd.to_datetime('2017-12-13').date()) & (avg_U['date'] <= pd.to_datetime('2017-12-31').date())]
U_A_2016plus = avg_U[(avg_U['date'] >= pd.to_datetime('2016-11-27').date()) & (avg_U['date'] <= pd.to_datetime('2016-12-18').date())]


# searching for the dates of the turnovers
dates_2012_Ax = get_turnover_dates(U_A_2012x)
dates_2013_Ax = get_turnover_dates(U_A_2013x)
dates_2014_Ax = get_turnover_dates(U_A_2014x)
dates_2015_Ax = get_turnover_dates(U_A_2015x)
dates_2016_Ax = get_turnover_dates(U_A_2016x)
dates_2017_Ax = get_turnover_dates(U_A_2017x)
dates_2018_Ax = get_turnover_dates(U_A_2018x)
dates_2022_Ax = get_turnover_dates(U_A_2022x)

dates_2013_Aplus = get_turnover_dates(U_A_2013plus)
dates_2014_Aplus = get_turnover_dates(U_A_2014plus)
dates_2016_Aplus = get_turnover_dates(U_A_2016plus)
dates_2017_Aplus = get_turnover_dates(U_A_2017plus)


# plotting wind speed for 2013 and 2017 turnover periods
fig_u = plt.figure(figsize=(10,6))
plt.plot(dates_2013_Ax, U_A_2013x['KVJ_EDDY.U'],label= '2013 short', color= 'deeppink')
plt.plot(dates_2017_Ax, U_A_2017x['KVJ_EDDY.U'],label= '2017 long', color= 'green')

plt.plot(dates_2013_Aplus,U_A_2013plus['KVJ_EDDY.U'], label='2013 three weeks after',ls='--',color='deeppink')
plt.plot(dates_2017_Aplus,U_A_2017plus['KVJ_EDDY.U'], label='2017 three weeks after',ls='--',color='green')

plt.xlabel('Day of year',fontsize=18)
plt.ylabel('Wind speed (m/s)',fontsize =18)

plt.axvline(x= 328,linewidth=2,color='black') # when short 2013 turnover period ends
plt.axvline(x= 347,linewidth=2,color='blue') # when long 2017 turnover period ends
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=12, loc='upper left',bbox_to_anchor=(0.8,1.0), borderaxespad=0.)

# adjusting layout to prevent cutting off the legend
plt.subplots_adjust(right=0.80)

fig_u.savefig('u_comparison.png')


#############################################################################
# surface water temperature and daily mean wind speed during turnover periods

w_A_2013x = avg_w[(avg_w['date'] >= pd.to_datetime('2013-10-12').date()) & (avg_w['date'] <= pd.to_datetime('2013-11-24').date())]
w_A_2017x = avg_w[(avg_w['date'] >= pd.to_datetime('2017-10-03').date()) & (avg_w['date'] <= pd.to_datetime('2017-12-13').date())]
# turnover dates plus 3 weeks
w_A_2013plus = avg_w[(avg_w['date'] >= pd.to_datetime('2013-11-24').date()) & (avg_w['date'] <= pd.to_datetime('2013-12-15').date())]
w_A_2017plus = avg_w[(avg_w['date'] >= pd.to_datetime('2017-12-13').date()) & (avg_w['date'] <= pd.to_datetime('2017-12-31').date())]

# plotting surface water temperature for autumn turnover periods 2013 and 2017
fig_w = plt.figure(figsize=(10,6))
plt.plot(dates_2013_Ax, w_A_2013x['values'],label= '2013 short', color= 'deeppink')
plt.plot(dates_2017_Ax, w_A_2017x['values'],label= '2017 long', color= 'green')

# three weeks after turnover period ends
plt.plot(dates_2013_Aplus, w_A_2013plus['values'],label='2013 three weeks after', ls='--',color='deeppink')
plt.plot(dates_2017_Aplus, w_A_2017plus['values'],label='2017 three weeks after', ls='--',color='green')

plt.xlabel('Day of year',fontsize=16)
plt.ylabel('Water temperature (°C)',fontsize =16)
plt.yticks(np.arange(-1,11),fontsize=18)

plt.axvline(x= 328,linewidth=2,color='black') # when short turnover period ends
plt.axvline(x= 347,linewidth=2,color='blue') # when long turnover period ends
plt.xticks(fontsize=18)
plt.axhline(y=0, color='black')
plt.axhline(y=4, color='red')
plt.legend(fontsize= 12, loc='upper right', bbox_to_anchor=(1.2,1.0), borderaxespad=0.)

# adjusting layout to prevent cutting off the legend
plt.subplots_adjust(right=0.8)
plt.show()

fig_w.savefig('wT_comparison.png')
