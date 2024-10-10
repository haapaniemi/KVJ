import datetime
import pandas as pd 
import requests
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import time
import xarray as xr

from matplotlib import colors
import pickle

import math
import cmocean
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy
import cartopy.io.shapereader as shpreader

#https://smear-backend.2.rahtiapp.fi//search/timeseries/csv?tablevariable=KVJ_META.Tw15&from=2024-10-08T00%3A00%3A00.000&to=2024-10-09T23%3A59%3A59.999&quality=ANY&aggregation=NONE&interval=1

varname = 'Tw15'
current_datetime = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

def GET_temperature_data():
    """
    Function for fetching data from SmartMet for given station name and 
    time period. Returns the data as pandas DataFrame.
    
    Parameters
    ----------

    Returns
    -------
    df : pd DataFrame
        fetched data
    """
    url = "https://smear-backend.2.rahtiapp.fi//search/timeseries/csv?tablevariable=KVJ_META.Tw15&from=2024-10-08T00%3A00%3A00.000&to=2024-10-09T23%3A59%3A59.999&quality=ANY&aggregation=NONE&interval=1"
   # url_2m = "https://smear-backend.2.rahtiapp.fi//search/timeseries/csv?tablevariable=KVJ_META.Tw20&from=2024-10-08T00%3A00%3A00.000&to=2024-10-09T23%3A59%3A59.999&quality=ANY&aggregation=NONE&interval=1"

#    url = 'http://smartmet.fmi.fi/timeseries'
#    payload = {
#        "fmisid" : "{}".format(station_id),
#        "producer": "observations_fmi",
#        "precision": "auto", 			#automatic precision
#        "tz":"utc",
#        "param": "stationname, stationlat, stationlon," \
#            "fmisid," \
#            "utctime," \
#             'TW_PT1M_AVG',\
#        "starttime": "{}".format(start_dt.strftime("%Y-%m-%dT%H:%M:%S")), 
#        "endtime": "{}".format(end_dt.strftime("%Y-%m-%dT%H:%M:%S")),
#        "timestep": "data",
#        "format": "json"
#        }
    running = True
    while running:
        try:
            r = requests.get(url)
            print(r.url)
            running = False
        except: 
            print("Connection refused by the server (Max retries exceeded)")
            print("Taking a nap...")
            print("ZZzzzz...")
            time.sleep(10)
            print("Slept for 10 secods, now continuing...")
    
    dictr = r.json() 
   
    df = pd.json_normalize(dictr)
  #  print(df)
    try:
        #df['time']= pd.to_datetime(df['time'])
        print('ok')
        #df.to_csv('/work/data/haapanie/2024/WT_'+savename+\
        #          '_{}.csv'.format(station_id), index=False)
        return df
    except:
        print('empty dataframe')
        #df.to_csv('/work/data/haapanie/2024/WT_'+savename+\
        #          '_{}.csv'.format(station_id), index=False)
        return  df
    
if __name__ == '__main__':
    path = '/home/haapanie/hyytiala/'
#    dat = pd.read_csv(path + 'waterT_SMEAR_Kuivajarvi_120101_201231_30min.csv',sep=';')
    dat = pd.read_csv(path + 'waterT_SMEAR_Kuivajarvi_120101_230630_30min.csv',sep=',')
    print(dat.dropna())
    print(dat)
    
    dat['datetime'] = pd.to_datetime(dat[['Year','Month','Day','Hour','Minute','Second']])
    dat = dat.set_index('datetime')
    dat = dat.drop(columns=['Year','Month','Day','Hour','Minute','Second'])
   #print(dat)
    
   # print(dat.columns)

#    dat02 = dat['KVJ_META.Tw02']
 #   print(dat02)
    
   # depths = ['0.2', '0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5','5', '6', '7' ,'8', '10', '12']
   # vals = [dat['KVJ_META.Tw02'], dat['KVJ_META.Tw05'], dat['KVJ_META.Tw10'],\
   #         dat['KVJ_META.Tw15'], dat['KVJ_META.Tw20'], dat['KVJ_META.Tw25'],\
   #         dat['KVJ_META.Tw30'], dat['KVJ_META.Tw35'], dat['KVJ_META.Tw40'],\
   #         dat['KVJ_META.Tw45'], dat['KVJ_META.Tw50'], dat['KVJ_META.Tw60'],\
   #         dat['KVJ_META.Tw70'], dat['KVJ_META.Tw80'], dat['KVJ_META.Tw100'],\
   #         dat['KVJ_META.Tw120']]

    times = dat.index.values

    df = dat.reset_index().melt(id_vars=['datetime'], var_name='depth', value_name='values')   
    df['depth'] = df['depth'].str.extract('(\d+)').astype(int)
    df['depth'] = df['depth']/10
    

    df= df.set_index(['datetime', 'depth'])
    df= df.sort_index(level='depth')
    
    
# =============================================================================
#     
# =============================================================================

    for d in np.unique(df.index.get_level_values(1)):
        surf_temp = df[df.index.get_level_values(1) == d]
    
        plt.plot(surf_temp.index.get_level_values(0), surf_temp.values,label='depth = '+str(d))
    plt.legend()
    plt.show()
    
   #na print(k)
   # surf_temp.to_csv('waterT_0.2m.csv')   
    
#    data['T'] = data['T'] -273.15
    
# =============================================================================
#     
# =============================================================================

    t = len(np.unique(df.index.get_level_values(0)))
    d = 16
    
    z = df.values.reshape(d, t)
    x=np.unique(df.index.get_level_values(0))
    y=np.unique(df.index.get_level_values(1))
    X, Y = np.meshgrid(x,-y)
    
    vmin, vmax = 0, 25
    
    cmap = cmocean.cm.thermal
    plt.figure(figsize=(15,7))
    plt.contourf(X, Y, z,cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title('Water temperature')
    plt.xlabel('Time')
    plt.ylabel('Depth')
#    a = plt.contour(X, Y, z,cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()
    
    print(df)


        
    cmap = cmocean.cm.thermal
    plt.figure(figsize=(15,7))
    a = plt.contour(X, Y, z,cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(a)
    #cbar.set_ticks(bounds)
    #cbar.set_ticklabels(names)
    
    
    plt.show()
    
# =============================================================================
#     
# =============================================================================

    times = df.index.get_level_values(0).strftime('%m-%d')
    print(times)
    
    # GROUPED TIME [M-D, DEPTHS]
    mds = df.groupby([df.index.get_level_values(0).strftime('%m-%d'),\
                      df.index.get_level_values(1)]).mean()
    print(mds)
    
    # -----------------------------------
    
    dates = mds.index.get_level_values(0)
    year = '2024'
    dt = pd.to_datetime([f'{year}-{date}' for date in dates], format='%Y-%m-%d')
    
    mds = mds.reset_index()
    print(mds)
    
    mds['time'] = dt
    print(mds)
    mds = mds[['time','depth', 'values']]
    mds = mds.set_index(['time','depth'])
    
    t = len(np.unique(mds.index.get_level_values(0)))
    d = 16
    
    z = mds.values.reshape(d, t)
    print(z.shape)
    print(z)
    
    x=np.unique(mds.index.get_level_values(0))
    y=np.unique(mds.index.get_level_values(1))
    X, Y = np.meshgrid(x,-y)
    
    print(X.shape)
    print(Y.shape)
    
    cmap = cmocean.cm.thermal
    plt.figure(figsize=(15,7))
    a = plt.contour(X, Y, z,cmap=cmap)
    cbar = plt.colorbar(a)

    plt.show()
    
    
    
    
#    mds = df.index.get_level_values(0).dt.strftime('%m%d')
#    print(mds)
#    avg_year = df.groupby().mean()
    #print(avg_year)
    
    
    
