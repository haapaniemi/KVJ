#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis of atmosphere-surface interactions and feedbacks / Hyytiälä 2024
analysis scripts

@author: Veera Haapaniemi (veera.haapaniemi@fmi.fi)

Water column heat content calculated based on openly available SmartSmear/AVAA
thermistor chain measuremets. Thermocline depth not taken into account in 
heat content calculation.

Wind speeds plotted agains turnover lengths.

"""

import numpy as np
import pandas as pd
from scipy import stats    
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean
import datetime

# =============================================================================
# DEFINE PATHS FOR READING AND PLOTTING
# =============================================================================
fig_path = 'KVJ/image/'
data_in = 'KVJ/data/data_in/'
data_out = 'KVJ/data/data_out/'

# =============================================================================
# COLORMAPS AND CONSTANTS
# =============================================================================

colors=['#e69f00', '#009e73']
cmap = cmocean.cm.thermal

density=1000
specific_heat = 3.999
q_4degC = density * specific_heat * 12 *4 / 1e3

# =============================================================================
# DEFINE FUNCTIONS
# =============================================================================
def format_temp_timeseries():
    # Read in water temperature measurement dataset downloaded from SmartSmear
    dat = pd.read_csv(data_in + 'waterT_SMEAR_Kuivajarvi_120101_230630_30min.csv',sep=',')
    
    # Format timestamps
    dat['datetime'] = pd.to_datetime(dat[['Year','Month','Day','Hour','Minute','Second']])
    
    # Set time as index
    dat = dat.set_index('datetime')
    dat = dat.drop(columns=['Year','Month','Day','Hour','Minute','Second'])
    print('Water temperature data read in and formatted')
    
    # Formatting the depth column
    df = dat.reset_index().melt(id_vars=['datetime'], var_name='depth', value_name='values')   
    df['depth'] = df['depth'].str.extract('(\d+)').astype(int)
    df['depth'] = df['depth']/10
    df= df.set_index(['datetime', 'depth'])
    df= df.sort_index(level='depth')
    df.to_pickle(data_out+'waterT_processed.pkl')
    print('Water temperatures saved to pickle')
    
    return df


def fit(x,y):
    """ Return R^2 where x and y are array-like."""
    return stats.linregress(x, y)
def myfunc(x):
    return slope * x + intercept

def plot_avg_year(avg_year):
    
    fig, axs = plt.subplots(1,2, sharex=True, figsize=(16,8))

# =============================================================================
#     UPPER PLOT    
# =============================================================================
    #plt.figure(figsize=(10,10))
    plt.suptitle('Average year')
    for d in np.unique(avg_year.index.get_level_values(1)):
        level_temp = avg_year[avg_year.index.get_level_values(1) == d]
        axs[0].plot(level_temp.index.get_level_values(0), level_temp.values,label='depth = '+str(d))

    axs[0].set_ylim(0,25)
    axs[0].legend()
    
# =============================================================================
#   LOWER PLOT
# =============================================================================
    # Time and depth levels for reshaping the data for plotting        
    t = len(np.unique(avg_year.index.get_level_values(0)))
    d = 16
    z = avg_year.values.reshape(d, t)
    
        
    # Levels for discrete plotting
    levels = np.linspace(0,25,14)
    
    # create x, y mesh for plotting
    x=np.unique(avg_year.index.get_level_values(0))
    y=np.unique(avg_year.index.get_level_values(1))
    X, Y = np.meshgrid(x,y)


    a = axs[1].contourf(X, Y, z, cmap=cmap, levels=levels)    
    axs[1].set_xlabel('Day of Year',fontsize=14)
    axs[1].set_ylabel('Depth [m]',fontsize=14)

    # Set x and y limits for values
    axs[1].set_xlim(0, 364.99999)
    axs[1].set_ylim(12,0.2)

    # Add colorbar
    cbar = plt.colorbar(a)
    cbar.set_label('Water temperature [$^\circ$C]',fontsize=16)
    plt.show()
    return


def plot_all_years(df):
    
    fig, axs = plt.subplots(2,1, figsize=(30,15))

    for d in np.unique(df.index.get_level_values(1)):
        surf_temp = df[df.index.get_level_values(1) == d]
        axs[0].plot(surf_temp.index.get_level_values(0), surf_temp.values,label='depth = '+str(d))
    axs[0].legend()
    #plt.show()
    
    t = len(np.unique(df.index.get_level_values(0)))
    d = 16
    z = df.values.reshape(d, t)
    
    x=np.unique(df.index.get_level_values(0))
    y=np.unique(df.index.get_level_values(1))
    X, Y = np.meshgrid(x,-y)
    
    levels = np.linspace(0,25,14)
    a = axs[1].contourf(X, Y, z, cmap=cmap, levels=levels)

    # Create colorbar
    cbar = plt.colorbar(a)
    cbar.set_label('Water temperature [$^\circ$C]',fontsize=16)
    plt.show()
    return

def compute_heat_content():
    dat = pd.read_csv(data_in + 'waterT_SMEAR_Kuivajarvi_120101_230630_30min.csv',sep=',')
    
    dat['datetime'] = pd.to_datetime(dat[['Year','Month','Day','Hour','Minute','Second']])
    dat = dat.set_index('datetime')
    dat = dat.drop(columns=['Year','Month','Day','Hour','Minute','Second'])
    
    df = dat.reset_index().melt(id_vars=['datetime'], var_name='depth', value_name='values')   
    df['depth'] = df['depth'].str.extract('(\d+)').astype(int)
    df['depth'] = df['depth']/10
    

    df= df.set_index(['datetime', 'depth'])
    df= df.sort_index(level='depth')        
            
    dz = [0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 2, 2]

    depths = np.unique(df.index.get_level_values(1))
    q_slices = []
    
    
    print(df)
    
#    plt.title('Heat content in each slice')
    for i in range(len(depths)):
        d = depths[i]
        z = dz[i]
        print('depth = ', d)
        print('dz = ', z)
        
        temp_series = df[df.index.get_level_values(1) == d]
        print(temp_series)
        
        temp_series['Q/m2'] = specific_heat * density * z * np.copy(temp_series.values)
        print(temp_series)
        
        temp_series = temp_series.reset_index()
        q_slices.append(temp_series)
        
      #  plt.plot(temp_series.index.get_level_values(0), temp_series['Q/m2'],\
      #           label='depth = '+str(d))

    print('here')
    q = pd.concat(q_slices)
    q = q.groupby('datetime').sum()
    q = q[['Q/m2']]
    
    # MASK ZERO VALUES
    q = q[q.values >0]
    # from kJ to MJ
    q = q / 1e3
    # ADD NAN FOR MISSING TIMES
    q = q.asfreq('30min', method=None)  # Create missing dates with NaN
    
    # Saving the computed heat contents
    q.to_csv(data_out +'heat_content.csv')
    
    return q



def plot_heat_contents(q, turnover_periods):
    fig, ax = plt.subplots(figsize=(13, 6))
    

    ax.set_xlabel("Time",fontsize=16)
    ax.set_title('Water column heat content',fontsize=16)
    
    # Plot horizontal line for isothermal 4 degree water colun
    ax.hlines(y=q_4degC, xmin=q.index.values[0], xmax=q.index.values[-1],\
               linewidth=2, color='r',label='H = H(T=4$^\circ$C)')
        
    print(q.values)
    ax.plot(q.index.values, q.values, c='k', linewidth=2)
        

    for i in range(0,len(turnover_periods),2):
        start_period = turnover_periods['datetime'].iloc[i]
        end_period = turnover_periods['datetime'].iloc[i + 1]
        if pd.notna(start_period):
            if i == 0:
                ax.axvspan(start_period, end_period,\
                           color=colors[int(np.floor(i%4)/2)],\
                           label="Autumn turnover", alpha=0.5)
            elif i == 2:
                ax.axvspan(start_period, end_period,\
                           color=colors[int(np.floor(i%4)/2)],\
                           label="Spring turnover", alpha=0.5)
            else:
                ax.axvspan(start_period, end_period,\
                           color=colors[int(np.floor(i%4)/2)], alpha=0.5)

    ax.set_ylim(-50, 750)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    
    #plt.xlim(pd.to_datetime('2017-03-20'), pd.to_datetime('2017-12-31'))
    plt.ylabel('Enthalpy [MJ/m$^2$]',fontsize=16)
    plt.legend(fontsize=12, loc='lower left')
    plt.savefig(fig_path + 'heat_content_all_years.png',dpi=700)
    plt.show()
    
    return

def plot_2panel_heat_contents(q, turnover_periods):
    """
    Plot heat content time series together with turnover period timings
    and heat fluxes.
    """
    
    dat = pd.read_pickle("hyytiala/flux_data_kuivajarvi.pkl")
    
    dat = dat[dat['KVJ_EDDY.Qc_H'] != 2]
    dat = dat[dat['KVJ_EDDY.Qc_LE_LI72'] != 2]
    dat = dat[dat['KVJ_EDDY.LE_LI72'] < 500]
    
    rad_dat = pd.read_pickle("hyytiala/data_radiation_data.pkl")
    rad_dat = rad_dat[rad_dat['KVJ_META.Glob'] <550]

    # Set time averaging 
    time_averaging = '1D'    
    dat_resampled = dat.resample(time_averaging).mean()
    rad_dat = rad_dat.resample(time_averaging).mean()
    
    net_rad = rad_dat['KVJ_META.Glob'] - rad_dat['KVJ_META.RGlob'] \
                + rad_dat['KVJ_META.LWin'] - rad_dat['KVJ_META.LWout'] \
                + dat_resampled['KVJ_EDDY.H'] + dat_resampled['KVJ_EDDY.LE_LI72']
               
# =============================================================================
    
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(13, 10))
    
    ax[0].set_xlabel("Time",fontsize=16)
    ax[0].set_title('Water column heat content',fontsize=16)
    
    # Plot horizontal line for isothermal 4 degree water colun
    ax[0].hlines(y=q_4degC, xmin=q.index.values[0], xmax=q.index.values[-1],\
               linewidth=2, color='r',label='H = H(T=4$^\circ$C)')
        
    print(q.values)
    ax[0].plot(q.index.values, q.values, c='k', linewidth=2)
        

    for i in range(0,len(turnover_periods),2):
        start_period = turnover_periods['datetime'].iloc[i]
        end_period = turnover_periods['datetime'].iloc[i + 1]
        if pd.notna(start_period):
            if i == 0:
                ax[0].axvspan(start_period, end_period,\
                           color=colors[int(np.floor(i%4)/2)],\
                           label="Autumn turnover", alpha=0.5)
            elif i == 2:
                ax[0].axvspan(start_period, end_period,\
                           color=colors[int(np.floor(i%4)/2)],\
                           label="Spring turnover", alpha=0.5)
            else:
                ax[0].axvspan(start_period, end_period,\
                           color=colors[int(np.floor(i%4)/2)], alpha=0.5)

    ax[0].set_ylim(-50, 750)
    ax[0].set_xlim(pd.to_datetime('2012-06-01'), pd.to_datetime('2023-07-31'))    
    ax[0].set_ylabel('Enthalpy [MJ/m$^2$]',fontsize=16)
    ax[0].legend(fontsize=12, loc='lower left')

# =============================================================================
#     add 2nd panel; energy budget terms together with heat content ts
# =============================================================================    
    
    for i in range(0,len(turnover_periods),2):
        start_period = turnover_periods['datetime'].iloc[i]
        end_period = turnover_periods['datetime'].iloc[i + 1]
        if pd.notna(start_period):
            if i == 0:
                ax[1].axvspan(start_period, end_period,\
                           color=colors[int(np.floor(i%4)/2)],\
                           label="Autumn turnover", alpha=0.5)
            elif i == 2:
                ax[1].axvspan(start_period, end_period,\
                           color=colors[int(np.floor(i%4)/2)],\
                           label="Spring turnover", alpha=0.5)
            else:
                ax[1].axvspan(start_period, end_period,\
                           color=colors[int(np.floor(i%4)/2)], alpha=0.5)
    ax[1].set_xlim(pd.to_datetime('2012-06-01'), pd.to_datetime('2023-07-31'))
#    ax[1].set_xlim(pd.to_datetime('2017-03-20'), pd.to_datetime('2017-12-31'))
    ax[1].plot(rad_dat.index, rad_dat['KVJ_META.Glob'])
    ax[1].plot(net_rad.index, net_rad, linewidth=0.7, color='k')    
    plt.show()
    
    return

def plot_whole_turnovers_with_dates(turnover_periods, dat):
    spring_frames = []
    autumn_frames = []
    spring_lengths = []
    autumn_lengths = []

    for i in range(0,len(turnover_periods),2):
        start_period = pd.to_datetime(turnover_periods['datetime'].iloc[i]).date() 
        end_period =  pd.to_datetime(turnover_periods['datetime'].iloc[i + 1]).date()
        length = (end_period - start_period).days 
    
        try:
            if start_period.strftime('%m') in ['02','03','04','05','06','07']:
                sel_spring = dat[(dat.index > start_period) & \
                                 (dat.index < end_period)]
            
                spring_frames.append(sel_spring)
                spring_lengths.append(length)
            else:
                sel_autumn = dat[(dat.index > start_period) & \
                                 (dat.index < end_period)]
                      
                autumn_frames.append(sel_autumn)
                autumn_lengths.append(length)
            
        except ValueError:
            print('probs nan')

    springs = pd.concat(spring_frames)
    autumns = pd.concat(autumn_frames)    

    springs = springs.sort_index(level='year')
    autumns = autumns.sort_index(level='year')

    return spring_frames, autumn_frames, springs, autumns, spring_lengths, autumn_lengths




def compute_turnover_variables(turnover_periods, q):
    starts, ends = [], []
    lengths = []

    aut_starts, aut_ends = [], []
    
    spr_starts, spr_ends = [], []
    spr_lengths, aut_lengths = [], []

    spring_Hinit, aut_Hinit = [], []
    spring_Hend, aut_Hend = [], []
    spring_dH, aut_dH = [], []

    store_spr_turnover_q, store_aut_turnover_q = [], []
    
    for i in range(0,len(turnover_periods),2):
    
        start_period = pd.to_datetime(turnover_periods['datetime'].iloc[i])
        end_period =  pd.to_datetime(turnover_periods['datetime'].iloc[i + 1]) 
        length = (end_period - start_period).days 
        print(start_period, end_period)
    
        try:
            if start_period.strftime('%m') in ['02','03','04','05','06','07']:
                spr_starts.append(start_period)
                spr_ends.append(end_period)
                spr_lengths.append(length)
                
                # --- heat content calculation ----------------------------
                s = pd.to_datetime(start_period).date()
                e = pd.to_datetime(end_period).date()
            
                store_spr_turnover_q.append(q[(q.index > s) & (q.index < e)])
                H_init = q[q.index == s].values
                H_end = q[q.index == e].values
                change_in_heat_content = H_end - H_init
                print('dH = ', change_in_heat_content)    
                
                
                spring_dH.append(change_in_heat_content.item())
                spring_Hinit.append(H_init.item())
                spring_Hend.append(H_end.item())            
            
            else:
                aut_starts.append(start_period)
                aut_ends.append(end_period)
                aut_lengths.append(length)
            
                # --- heat content calculation ----------------------------
                s = pd.to_datetime(start_period).date()
                e = pd.to_datetime(end_period).date()
                
                store_aut_turnover_q.append(q[(q.index > s) & (q.index < e)])
                H_init = q[q.index == s].values
                H_end = q[q.index == e].values
       
                change_in_heat_content = H_end - H_init
                print('dH = ', change_in_heat_content)       
                
                
                aut_dH.append(change_in_heat_content.item())
                aut_Hinit.append(H_init.item())
                aut_Hend.append(H_end.item())
                
       
            starts.append(start_period)
            ends.append(end_period)
            lengths.append(length)
        
        except ValueError:
            print('probs nan')     
        
    return spr_starts, aut_starts, spr_ends, aut_ends, \
        spr_lengths, aut_lengths, spring_dH, aut_dH, spring_Hinit, aut_Hinit,\
        spring_Hend, aut_Hend



def add_boxes(sel_spring, start_turnover, end_turnover, \
              strong_events_counts, season):
    if season=='s':
        color=colors[1]
    else:
        color = colors[0]
        
    df_spring = sel_spring.to_frame(name='winds')
    # Convert to periods and then to datetime
    df_spring['week'] = df_spring.index.to_period('W')
    df_spring['week_start'] = df_spring['week'].apply(lambda x: x.start_time)
    
    # Prepare data for the boxplot: grouping by 'week_start'
    boxplot_data = [df_spring[df_spring['week_start'] == week]['winds'].values\
                    for week in df_spring['week_start'].unique()]

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Add the shaded region for spring turnover
   # plt.axvspan(turnover_start_index - 0.5, turnover_end_index + 0.5, 
   #             color='gray', label="Spring turnover", alpha=0.4)

    # Plot the boxplot with the 'patch_artist' argument set to True to fill the boxes
    box = plt.boxplot(boxplot_data, positions=np.arange(len(boxplot_data)), widths=0.6, 
                      patch_artist=True,  # This fills the boxes with color
                      boxprops=dict(color='k', linewidth=2),  # Box edge color
                      whiskerprops=dict(color='k'),  # Whisker color
                      capprops=dict(color='k'),  # Cap color
                      medianprops=dict(color='k'),  # Median color
                      flierprops=dict(marker='o', markersize=5, linestyle='none'))  # Outlier color

    # Set the color for the fill
    for patch in box['boxes']:
        patch.set_facecolor(color)

    # Add the shaded region for spring turnover
    turnover_start_index = np.where(df_spring['week_start'].values == start_turnover)[0]
    turnover_end_index = np.where(df_spring['week_start'].values == end_turnover)[0]
    
    # Format x-axis as datetime
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator())
    
    # Titles and labels
    plt.suptitle('Frequency of Strong Wind Events')
    plt.xlabel('Time')
    plt.ylabel('Number of Strong Wind Events')
    plt.title('Strong Wind Events by Week')
    
    plt.xticks(rotation=45)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Wind speed')
    plt.ylim(0,20)
    plt.show()
    
    
    # HISTOGRAM with turnover timing ------------------------------
    plt.figure(figsize=(12,6))
    plt.bar(strong_events_counts.index, strong_events_counts.values, width=4,\
            color='k')
    #strong_events_counts.plot(kind='bar', color=colors[1])
    if season=='s':
        plt.axvspan(start_turnover, end_turnover,\
                    color=colors[1],\
                    label="Spring turnover", alpha=0.4)
    else:
        plt.axvspan(start_turnover, end_turnover,\
                    color=colors[0],\
                    label="Autumn turnover", alpha=0.4)
    plt.suptitle(str(end_turnover.year))
    plt.xlabel('Time')
    plt.ylabel('Number of Strong Wind Events')
    plt.title('Frequency of Strong Wind Events')
    plt.ylim(0,100)
    plt.show()
    

    return
    

def plot_whole_turnovers_with_dates2(turnover_periods, dat, pctl, box_plots):
    spring_frames, autumn_frames = [], []
    spring_mins, autumn_mins = [], []
    spring_means, autumn_means = [], []
    spring_medians, autumn_medians = [],[]
    spring_maxs, autumn_maxs = [],[]
    
    spr_p90s, aut_p90s = [], []
    spring_lengths, autumn_lengths = [], []
    spr_strong_event_counts, aut_strong_event_counts = [], []
    
    #dat = dat.resample('2H').max()
    dat = dat[dat.values < 22]
    
    for i in range(0,len(turnover_periods),2):
        start_turnover = pd.to_datetime(turnover_periods['datetime'].iloc[i])
        start_period = start_turnover - datetime.timedelta(days = 21)
  
        end_turnover = pd.to_datetime(turnover_periods['datetime'].iloc[i + 1])
        end_period = end_turnover + datetime.timedelta(days = 21)
      
        length = (end_turnover - start_turnover).days 
    
        try:
            if start_period.strftime('%m') in ['02','03','04','05','06','07']:
                sel_spring = dat[(dat.index > start_period) & \
                                 (dat.index < end_period)]
                
                
                if sel_spring.empty:
                    spring_mins.append(np.nan)
                    spring_means.append(np.nan)
                    spring_medians.append(np.nan)
                    spring_maxs.append(np.nan)
                    
                    spring_frames.append(np.nan)
                    spr_p90s.append(np.nan)
                    spr_strong_event_counts.append(np.nan)
                    spring_lengths.append(np.nan)

                else:
                    print(sel_spring)
                    spring_min = np.nanmin(sel_spring.values)
                    spring_mean = np.nanmean(sel_spring.values)
                    spr_median = np.nanmedian(sel_spring.values)
                    spring_max = np.nanmax(sel_spring.values)
                
                    spring_mins.append(spring_min)
                    spring_means.append(spring_mean)
                    spring_medians.append(spr_median)
                    spring_maxs.append(spring_max)
                    
                    spr_p90 = np.percentile(sel_spring.values, pctl)
                    spr_p90s.append(spr_p90)
                    
                    spring_frames.append(sel_spring)
                    spring_lengths.append(length)
                    
                    # Resample to count strong wind events (e.g., hourly)
                    strong_events = sel_spring[sel_spring.values >= 5].resample('3D')
                    strong_events_counts = strong_events.size()
                    spr_strong_event_counts.append(strong_events_counts.sum())     
                    strong_events_counts.index = pd.to_datetime(strong_events_counts.index)
                    

                    if box_plots:
                        add_boxes(sel_spring, start_turnover, end_turnover,\
                                  strong_events_counts,'s')
                    
            else:
                sel_autumn = dat[(dat.index > start_period) & \
                                 (dat.index < end_period)]
               
             
                if sel_autumn.empty:
                    autumn_mins.append(np.nan)
                    autumn_means.append(np.nan)
                    autumn_medians.append(np.nan)
                    autumn_maxs.append(np.nan)
                    
                    autumn_frames.append(np.nan)
                    aut_p90s.append(np.nan)
                    autumn_lengths.append(np.nan)
                    aut_strong_event_counts.append(np.nan)

                else:
                    autumn_min = np.nanmin(sel_autumn.values)
                    autumn_mean = np.nanmean(sel_autumn.values)    
                    aut_median = np.nanmedian(sel_autumn.values)
                    autumn_max = np.nanmax(sel_autumn.values)
                    
                    autumn_mins.append(autumn_min)
                    autumn_means.append(autumn_mean)
                    autumn_medians.append(aut_median)
                    autumn_maxs.append(autumn_max)
                    
                    aut_p90 = np.percentile(sel_autumn.values, pctl)
                    aut_p90s.append(aut_p90)
                    
                    autumn_frames.append(sel_autumn)
                    autumn_lengths.append(length)


                    # Resample to count strong wind events (e.g., hourly)
                    strong_events = sel_autumn[sel_autumn.values >= 5].resample('3D')                   
                    strong_events_counts = strong_events.size()
                    aut_strong_event_counts.append(strong_events_counts.sum())
                    strong_events_counts.index = pd.to_datetime(strong_events_counts.index)                        
                   
                    if box_plots:
                        add_boxes(sel_autumn, start_turnover, end_turnover,\
                                  strong_events_counts, 'a')

            
        except ValueError:
            print('probs nan')
        
    
    plt.figure()
    plt.title('Wind speed events vs turnover length')
    plt.scatter(autumn_lengths, aut_strong_event_counts, c=colors[0], label='Autumn')
    plt.scatter(spring_lengths, spr_strong_event_counts, c=colors[1], label='Spring')
    plt.show()

    plt.title('Turnover length vs min wind')
    plt.scatter(autumn_lengths, autumn_mins, c=colors[0], label='Autumn')
    plt.scatter(spring_lengths, spring_mins, c=colors[1], label='Spring')
    plt.ylabel('Wind speed [m/s]')
    plt.xlabel('Length of the mixing period (days)')
    plt.legend()
    plt.ylim(0,25)
    plt.show()

    plt.title('Turnover length vs mean wind')
    plt.scatter(autumn_lengths, autumn_means, c=colors[0], label='Autumn')
    plt.scatter(spring_lengths, spring_means, c=colors[1], label='Spring')
    plt.ylabel('Wind speed [m/s]')
    plt.xlabel('Length of the mixing period (days)')
    plt.legend()
    plt.ylim(0,25)
    plt.show()
    
    
    fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,4))
    plt.suptitle('Turnover length - wind conditions')
    res = stats.linregress(autumn_lengths+spring_lengths, autumn_means+spring_means)
    x = np.linspace(1,75,100)
    axs[0].plot(x,res.intercept+res.slope*x,\
          label=f"$R^2$: {res.rvalue**2:.5f} \n Slope: {res.slope:.5f} \n p-value: {res.pvalue:.5f}")

    axs[0].set_title('Mean wind')
    axs[0].scatter(autumn_lengths, autumn_means, c=colors[0], label='Autumn')
    axs[0].scatter(spring_lengths, spring_means, c=colors[1], label='Spring')
    axs[0].set_ylabel('Wind speed [m/s]')
    axs[0].set_xlabel('Length of the mixing period (days)')
    axs[0].legend()
    axs[0].set_ylim(-4,23)
    
    axs[1].set_title('Maximum wind')
    res = stats.linregress(autumn_lengths+spring_lengths, autumn_maxs+spring_maxs)
    axs[1].plot(x,res.intercept+res.slope*x,\
            label=f"$R^2$: {res.rvalue**2:.5f} \n Slope: {res.slope:.5f} \n p-value: {res.pvalue:.5f}")

    axs[1].scatter(autumn_lengths, autumn_maxs, c=colors[0], label='Autumn')
    axs[1].scatter(spring_lengths, spring_maxs, c=colors[1], label='Spring')
 #   axs[1].set_ylabel('Wind speed [m/s]')
    axs[1].set_xlabel('Length of the mixing period (days)')
    axs[1].legend(loc='lower right')
    axs[1].set_ylim(-4,23)
    plt.tight_layout()
    plt.savefig(fig_path +'wind_length_correlation.png', dpi=500)
    plt.show()

    return 

def plot_wind_pctl(turnover_periods, dat, pctl):
    spring_frames, autumn_frames = [], []
    spr_p90s, aut_p90s = [], []
    spring_lengths, autumn_lengths = [], []
    spr_strong_event_counts, aut_strong_event_counts = [], []
    
    #dat = dat.resample('2H').max()
    dat = dat[dat.values < 22]
    
    for i in range(0,len(turnover_periods),2):
        start_turnover = pd.to_datetime(turnover_periods['datetime'].iloc[i])
        start_period = start_turnover# - datetime.timedelta(days = 21)
  
        end_turnover = pd.to_datetime(turnover_periods['datetime'].iloc[i + 1])
        end_period = end_turnover #+ datetime.timedelta(days = 21)
      
        length = (end_turnover - start_turnover).days 
    
        try:
            if start_period.strftime('%m') in ['02','03','04','05','06','07']:
                sel_spring = dat[(dat.index > start_period) & \
                                 (dat.index < end_period)]
                
                
                if sel_spring.empty:
                    spring_frames.append(np.nan)
                    spr_p90s.append(np.nan)
                    spr_strong_event_counts.append(np.nan)
                    spring_lengths.append(np.nan)

                else:
                    print(sel_spring)
                    spr_p90 = np.percentile(sel_spring.values, pctl)
                    spr_p90s.append(spr_p90)
                    
                    spring_frames.append(sel_spring)
                    spring_lengths.append(length)
                    
                    
            else:
                sel_autumn = dat[(dat.index > start_period) & \
                                 (dat.index < end_period)]
               
             
                if sel_autumn.empty:
                    autumn_frames.append(np.nan)
                    aut_p90s.append(np.nan)
                    autumn_lengths.append(np.nan)
                    aut_strong_event_counts.append(np.nan)

                else:
                    aut_p90 = np.percentile(sel_autumn.values, pctl)
                    aut_p90s.append(aut_p90)
                    
                    autumn_frames.append(sel_autumn)
                    autumn_lengths.append(length)
            
        except ValueError:
            print('probs nan')
        
    
    plt.title('Turnover length vs wind pctl='+str(pctl))
    plt.scatter(autumn_lengths, aut_p90s, c=colors[0], label='Autumn')
    plt.scatter(spring_lengths, spr_p90s, c=colors[1], label='Spring')
    plt.ylabel('Wind speed [m/s]')
    plt.xlabel('Length of the mixing period (days)')
    plt.legend()
    plt.ylim(0,25)
    plt.show()

    return 

def plot_specs_only(turnover_periods, dat, pctl, aut_len, spr_len):
    dat = dat[dat.values < 22]
    plt.plot(dat.index, dat.values)
    

    spr_cumsum, aut_cumsum = [], []
    
    fig, axs = plt.subplots(1,1)
    
    for i in range(0,len(turnover_periods),2):
        start_turnover = pd.to_datetime(turnover_periods['datetime'].iloc[i])
        start_period = start_turnover# - datetime.timedelta(days = 21)
    
        end_turnover = pd.to_datetime(turnover_periods['datetime'].iloc[i + 1])
        end_period = end_turnover# + datetime.timedelta(days = 21)
        length = (end_turnover - start_turnover).days 

        try:
            if start_period.strftime('%m') in ['02','03','04','05','06','07']:
                sel_spring = dat[(dat.index > start_period) & \
                                 (dat.index < end_period)]
                if sel_spring.empty:
                    print('empty')
                    spr_cumsum.append(np.nan)
                else:
                    data = sel_spring.copy()
                    data_cumsum = np.cumsum(data.values**2)
                    spr_cumsum.append(data_cumsum[-1])
                    print(data_cumsum)
                    axs.plot(data.index, data_cumsum, \
                             label="Wind Speed Cumulative Sum",\
                             c=colors[1])
            else:
                sel_autumn = dat[(dat.index > start_period) & \
                                 (dat.index < end_period)]
                if sel_autumn.empty:
                    print('empty')
                    aut_cumsum.append(np.nan)
                    print(sel_autumn)
                else:
                    data = sel_autumn.copy()
                    #data = data.resample('2H').mean()
                    data_cumsum = np.cumsum(data.values**2)
                    aut_cumsum.append(data_cumsum[-1])
                    axs.plot(data.index, data_cumsum,\
                             label="Wind Speed Cumulative Sum",\
                             c=colors[0])
        except ValueError:
            print('probs nan')
            
    plt.title("Wind Speed Cumulative Sum")
    axs.set_xlabel("Time")
    axs.set_ylabel("Wind")
    axs.grid(True, which="both", linestyle="--", linewidth=0.5)       
    plt.show()
    
    plt.title('Turnover length vs cumulative sum of wind speed')
    plt.scatter(aut_lengths, aut_cumsum, c=colors[0], label='Autumn')
    plt.scatter(spr_lengths, spr_cumsum, c=colors[1], label='Spring')
    plt.ylabel('Cumulative sum of wind speed [m/s]')
    plt.xlabel('Length of the mixing period (days)')
    plt.legend()
    plt.show()
    
        
    return 

def plot_correlations_simple(spr_lengths, aut_lengths, spring_dH, aut_dH,\
                             spr_rate, aut_rate, mymodel):
    
    fig = plt.figure(figsize=(8, 4))
    #axs = plt.subplots(1,2, figsize=(12,5))
    plt.title('Water column heat content change [MJ / m$^2$]')
    plt.scatter(aut_lengths, [abs(num) for num in aut_dH], c=colors[0], label='Autumn')
    plt.scatter(spr_lengths, [abs(num) for num in spring_dH], c=colors[1], label='Spring')
    plt.ylabel('Absolute change in heat content [MJ / m$^2$]')
    plt.xlabel('Length of the mixing period (days)')
    plt.plot(x, mymodel, label='R$^2$ = 0.85, p=2e-8' )
    plt.legend()
    plt.savefig(fig_path + 'heat_content_length_correlation.png', dpi=500)
    plt.show()
    
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    axs[0].set_title('Water column heat content change [MJ / m$^2$]')
    axs[0].scatter(aut_lengths, [abs(num) for num in aut_dH], c=colors[0], label='Autumn')
    axs[0].scatter(spr_lengths, [abs(num) for num in spring_dH], c=colors[1], label='Spring')
    #axs[0].scatter(aut_lengths, aut_Hend, c=colors[0])
    #axs[0].scatter(spr_lengths, spring_Hend, c=colors[1])
    axs[0].set_ylabel('Absolute change in heat content [MJ / m$^2$]')
    axs[0].set_xlabel('Length of the mixing period (days)')
    
    #plt.scatter(x, y)
    axs[0].plot(x, mymodel, label='R$^2$ = 0.85, p=2e-8' )
    axs[0].legend()
    
    
    axs[1].scatter(aut_lengths, [abs(num) for num in aut_rate], c=colors[0], label='Autumn')
    axs[1].scatter(spr_lengths, spr_rate, c=colors[1], label='Spring')
    axs[1].set_title('Rate of change in heat content [MJ / m$^2$ / d]')
    axs[1].set_xlabel('Length of the mixing period (days)')
    axs[1].set_ylabel('Rate of change in heat content [MJ / m$^2$ / d]')
    axs[1].legend()
    plt.savefig(fig_path + 'heat_content_length_correlation_n_rate.png', dpi=500)
    plt.show()
    return

def plot_dH_correlations(spr_lengths, aut_lengths, spring_dH, aut_dH,\
                         spr_Hinit, aut_Hinit, spr_Hend, aut_Hend,\
                         mymodel):
    
    fig, axs = plt.subplots(1,3, figsize=(12,5), sharex=True)
    axs[1].set_title('Water column heat content change [MJ / m$^2$]')
    axs[1].scatter(aut_lengths, [abs(num) for num in aut_dH], c=colors[0], label='Autumn')
    axs[1].scatter(spr_lengths, [abs(num) for num in spring_dH], c=colors[1], label='Spring')
    #axs[0].scatter(aut_lengths, aut_Hend, c=colors[0])
    #axs[0].scatter(spr_lengths, spring_Hend, c=colors[1])
    axs[1].set_ylabel('Absolute change in heat content [MJ / m$^2$]',fontsize=14)
    axs[1].set_xlabel('Length of the mixing period (days)', fontsize=14)
    
    axs[1].plot(x, mymodel, label='R$^2$ = 0.85, p=2e-8' )
    axs[1].set_ylim(0,520)
    axs[1].legend(fontsize=12)
    
    axs[0].scatter(aut_lengths, aut_Hinit, c=colors[0], label='Autumn')
    axs[0].scatter(spr_lengths, spr_Hinit, c=colors[1], label='Spring')
    axs[0].set_title('Initial heat content [MJ / m$^2$]')
    axs[0].set_xlabel('Length of the mixing period (days)',fontsize=12)
    axs[0].set_ylabel('Initial heat content [MJ / m$^2$]',fontsize=12)
    axs[0].set_ylim(0,520)
    axs[0].legend(fontsize=12)
    
    axs[2].scatter(aut_lengths, aut_Hend, c=colors[0], label='Autumn')
    axs[2].scatter(spr_lengths, spr_Hend, c=colors[1], label='Spring')
    axs[2].set_title('Final heat content [MJ / m$^2$]')
    axs[2].set_xlabel('Length of the mixing period (days)',fontsize=12)
    axs[2].set_ylabel('Final heat content [MJ / m$^2$]',fontsize=12)
    axs[2].legend(fontsize=12)
    axs[2].set_ylim(0,520)

    
    plt.tight_layout()
    plt.savefig(fig_path + 'heat_content_length_correlations_3.png', dpi=500)
    plt.show()
    return


def compute_days_to_turnover(spring_frames, autumn_frames, q_4degC,\
                             aut_lengths, spr_lengths):
    spr_d_before, aut_d_before = [], []
    spr_d_after, aut_d_after = [], []
    spring_fracs = []
    autumn_fracs = []
    for s in spring_frames:
        #print(s)
        before_4degC = s[s.values < q_4degC]
        after_4degC = s[s.values > q_4degC]
       # print('days before = ', len(before_4degC))
       # print('days after = ', len(after_4degC))
       # print('frac = ', len(before_4degC) / len(after_4degC))
        
        spr_d_before.append(len(before_4degC))
        spr_d_after.append(len(after_4degC))
        spring_fracs.append(len(before_4degC) / len(after_4degC))
        
    for a in autumn_frames:
        #print(a)
        before_4degC = a[a.values > q_4degC]
        after_4degC = a[a.values < q_4degC]
       # print('days before = ', len(before_4degC))
       # print('days after = ', len(after_4degC))
       # print('frac = ', len(before_4degC) / len(after_4degC))
        
        aut_d_before.append(len(before_4degC))
        aut_d_after.append(len(after_4degC))
        autumn_fracs.append(len(before_4degC) / len(after_4degC))
        
        # =============================================================================
        # add correlation against starting date doy
        # =============================================================================
        
    plt.figure(figsize=(7,3))
    plt.title('Fraction of turnover days before / after 4deg water column vs timing')
    plt.scatter(aut_lengths, aut_d_before, c=colors[0])
    plt.scatter(spr_lengths, spr_d_before, c=colors[1])
    plt.ylabel('Days before 4 degree column')
    plt.xlabel('Length of the mixing period (days)')
    plt.show()
      
    return spr_d_before, aut_d_before, spr_d_after, aut_d_after,\
        spring_fracs, autumn_fracs


if __name__ == '__main__':
# =============================================================================
#   WATER TEMPERATURES
# =============================================================================
    df = format_temp_timeseries()
    
    # Define avg year
    avg_year = df.reset_index()
    avg_year = avg_year[avg_year.datetime.dt.strftime('%m-%d') != '02-29']
    avg_year['doy'] = avg_year.datetime.dt.dayofyear.values
    
    avg_year = avg_year[['doy','depth','values']]
    avg_year = avg_year.groupby(['doy','depth']).mean()
    avg_year = avg_year.sort_index(level='depth')
    
    # Plotting
    plot_avg_year(avg_year)
    plot_all_years(df)

# =============================================================================
#   WATER COLUMN HEAT CONTENTS
# =============================================================================
  
    # Compute water column heat content
    q = compute_heat_content()
    
    # Read in previously computed heat content time series for plottng
    q = pd.read_csv(data_out+ 'heat_content.csv')
    
    q['datetime'] = pd.to_datetime(q['datetime'])
    q = q.groupby(q['datetime'].dt.date)['Q/m2'].mean().reset_index()
    q = q.set_index('datetime')
    
    # Read in turnover timings for plotting
    turnover_periods = pd.read_pickle(data_in + 'turnover_periods.pkl')

    plot_heat_contents(q, turnover_periods)
    plot_2panel_heat_contents(q, turnover_periods)
    
# =============================================================================
#   COMPUTE HEAT CONTENT CHANGES DURING TURNOVERS
# =============================================================================
    spr_starts, aut_starts, spr_ends, aut_ends, \
        spr_lengths, aut_lengths, spring_dH, aut_dH, spr_Hinit, aut_Hinit,\
        spr_Hend, aut_Hend = compute_turnover_variables(turnover_periods, q)

    aut_starts1 = pd.Series(aut_starts).dt.date
    aut_ends1 = pd.Series(aut_ends).dt.date
    
    # Ensure all values are in datetime format (already are in this case)
    aut_starts = pd.to_datetime(aut_starts1).values#.date
    aut_ends = pd.to_datetime(aut_ends1).values#.date
    
    spr_starts1 = pd.Series(spr_starts).dt.date
    spr_ends1 = pd.Series(spr_ends).dt.date
    
    # Ensure all values are in datetime format (already are in this case)
    spr_starts = pd.to_datetime(spr_starts1).values#.date
    spr_ends = pd.to_datetime(spr_ends1).values#.date


# =============================================================================
#   COMPUTE LINEAR REGRESSION FOR HEAT CONTENT CHANGES
# =============================================================================  
    X = spr_lengths + aut_lengths
    Y =  [abs(num) for num in spring_dH] + [abs(num) for num in aut_dH]
    
    slope, intercept, r_value, p_value, sd_err = fit(X, Y)
    print('Slope = ', slope)
    print('Intercept', intercept)
    print('R2', r_value**2)
    print('p', p_value)
    print('sd', sd_err)
    
    x = np.linspace(5,75,100)
    mymodel = list(map(myfunc, x))

    aut_rate = [x/y for x, y in zip(map(int, aut_dH), map(int, aut_lengths))]
    spr_rate = [x/y for x, y in zip(map(int, spring_dH), map(int, spr_lengths))]


# =============================================================================
#   PLOT WATER COLUMN HEAT CONTENT CORRELATION PLOTS
# =============================================================================
    plot_correlations_simple(spr_lengths, aut_lengths, spring_dH, aut_dH, \
                             spr_rate, aut_rate, mymodel)
    plot_dH_correlations(spr_lengths, aut_lengths, spring_dH, aut_dH, \
                         spr_Hinit, aut_Hinit, spr_Hend, aut_Hend,\
                         mymodel)
        
    spring_frames, autumn_frames, springs, autumns, spring_lengths, autumn_lengths = \
        plot_whole_turnovers_with_dates(turnover_periods, q)

    # Get data to consider days to turnovers
    spr_d_before, aut_d_before, spr_d_after, aut_d_after,\
        spring_fracs, autumn_fracs = compute_days_to_turnover(spring_frames,\
                                        autumn_frames, q_4degC, autumn_lengths, \
                                        spring_lengths)
            
# =============================================================================
#    PLOT WIND SPEED CORRELATIONS
# =============================================================================
    
    u_dat = pd.read_pickle(data_out + 'wpt_kuivajarvi.pkl')
    print(u_dat.columns)
    
    u = u_dat['KVJ_EDDY.av_u']
    v = u_dat['KVJ_EDDY.av_v']
   
    u_mod = np.sqrt(u**2 + v**2)
    u_mod = u_dat['KVJ_EDDY.U']
    
    plot_specs_only(turnover_periods, u_mod, 90, autumn_lengths,\
                    spring_lengths)
    plot_whole_turnovers_with_dates2(turnover_periods, u_mod, 90, False)
    
    for pctl in [5,10,20,50,70,90,95]:
        plot_wind_pctl(turnover_periods, u_mod, pctl)