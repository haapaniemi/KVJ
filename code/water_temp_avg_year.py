import datetime
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import xarray as xr

from matplotlib import colors
import cmocean


if __name__ == '__main__':
    path = '/home/haapanie/hyytiala/'
#    dat = pd.read_csv(path + 'waterT_SMEAR_Kuivajarvi_120101_201231_30min.csv',sep=';')
    dat = pd.read_csv(path + 'waterT_SMEAR_Kuivajarvi_120101_230630_30min.csv',sep=',')
    
    dat['datetime'] = pd.to_datetime(dat[['Year','Month','Day','Hour','Minute','Second']])
    dat = dat.set_index('datetime')
    dat = dat.drop(columns=['Year','Month','Day','Hour','Minute','Second'])
    times = dat.index.values
    print(times)

    df = dat.reset_index().melt(id_vars=['datetime'], var_name='depth', value_name='values')   
    df['depth'] = df['depth'].str.extract('(\d+)').astype(int)
    df['depth'] = df['depth']/10


    df= df.set_index(['datetime', 'depth'])
    df= df.sort_index(level='depth')


    # ----------------------------------------------------------------------------------------------------

  
    avg_year = df.reset_index()
    avg_year = avg_year[avg_year.datetime.dt.strftime('%m-%d') != '02-29']
    avg_year['doy'] = avg_year.datetime.dt.dayofyear.values
    
    avg_year = avg_year[['doy','depth','values']]
    print(avg_year)
    avg_year = avg_year.groupby(['doy','depth']).mean()
    avg_year = avg_year.sort_index(level='depth')
    
    for d in np.unique(avg_year.index.get_level_values(1)):
        plt.title(str(d))
        level_temp = avg_year[avg_year.index.get_level_values(1) == d]
        plt.plot(level_temp.index.get_level_values(0), level_temp.values,label='depth = '+str(d))
        plt.ylim(0,25)
        plt.show()
    
    t = len(np.unique(avg_year.index.get_level_values(0)))
    d = 16
    z = avg_year.values.reshape(d, t)
    x=np.unique(avg_year.index.get_level_values(0))
    y=np.unique(avg_year.index.get_level_values(1))
    X, Y = np.meshgrid(x,y)
     
    cmap = cmocean.cm.thermal
    plt.figure(figsize=(10,7))
    levels = np.linspace(0,30,10)
    a = plt.contourf(X, Y, z, cmap=cmap, levels=levels)
    plt.title('Water temperature',fontsize=16)
    plt.xlabel('Time',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylabel('Depth [m]',fontsize=14)
    plt.ylim(12,0)
    
    # Format the x-axis to show months
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Major ticks on the first day of each month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Month names (e.g., Jan, Feb)
    
    cbar = plt.colorbar(a)
    cbar.set_label('Water temperature [$^\circ$C]',fontsize=16)
    plt.show()
