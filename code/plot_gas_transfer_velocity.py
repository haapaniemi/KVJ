#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis of atmosphere-surface interactions and feedbacks / Hyytiälä 2024
analysis scripts

@author: Veera Haapaniemi (veera.haapaniemi@fmi.fi)

Gas transfer velocity computation + plotting.

"""
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import matplotlib.pyplot as plt

fig_path = 'KVJ/image/'
data_in = 'KVJ/data/data_in/'
data_out = 'KVJ/data/data_out/'

turnover_periods = pd.read_pickle(data_out + "turnover_periods.pkl") 

# =============================================================================
# COLORS AND CONSTANTS
# =============================================================================

colors=['#e69f00', '#009e73']

g = 9.81
alpha = 0.0007 #207*10e-6

#0.0007
density= 1000
specific_heat = 3.999*1e3
 
c1 = 0.56
c2 = 0.77
c3 = 0.6
kappa = 0.41
z = 0.15
 
c4 = 0.5
nu = 1.002*1e-6
 
# -----------
size=80

# =============================================================================
# FUNCTIONS
# =============================================================================

def plot_whole_turnovers_with_dates(turnover_periods, dat):
    spring_frames = []
    autumn_frames = []
    spring_lengths = []
    autumn_lengths = []

    for i in range(0,len(turnover_periods),2):
        start_period = pd.to_datetime(turnover_periods['datetime'].iloc[i]) - \
            datetime.timedelta(days = 60)
        end_period =  pd.to_datetime(turnover_periods['datetime'].iloc[i + 1]) + \
            datetime.timedelta(days = 60)
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

def plot_stripes():    
    
    fig, ax = plt.subplots(figsize=(14,7))    
    for i in range(0,len(turnover_periods),2):
        start_period = turnover_periods['datetime'].iloc[i]
        end_period = turnover_periods['datetime'].iloc[i + 1]
        if pd.notna(start_period):
            if i == 0:
                ax.axvspan(start_period, end_period, color=colors[int(np.floor(i%4)/2)],label="Fall turnover", alpha=0.5)
            elif i == 2:
                ax.axvspan(start_period, end_period, color=colors[int(np.floor(i%4)/2)],label="Spring turnover", alpha=0.5)
            else:
                ax.axvspan(start_period, end_period, color=colors[int(np.floor(i%4)/2)], alpha=0.5)
    return
def plot_buoyancy(beta_heat_in, beta_heat_out):
    fig, ax = plt.subplots(figsize=(14,7))

    
    plt.xlabel("Time",fontsize=20)
    plt.title('Buoyancy flux',fontsize=20)
    plt.grid(color='gray', alpha=0.5, linestyle='--')

    plt.plot(beta_heat_in.index, beta_heat_in.values, c='m', linewidth=1,\
             label='Lake warming')
    plt.plot(beta_heat_out.index, beta_heat_out.values, c='c', linewidth=1,\
             label='Lake cooling')

    plot_stripes()

    plt.ylim(-50, 250)
    plt.xlim(pd.to_datetime('2012-05-01'), pd.to_datetime('2023-12-31'))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    
    #plt.xlim(pd.to_datetime('2017-03-20'), pd.to_datetime('2017-12-31'))
    plt.ylabel('Buoyancy',fontsize=20)
    plt.legend(fontsize=14, loc='upper right')
    plt.show()

    return

def plot_gas_transfer_velocity(avg_y_gas_transfer, doy=False):

    plt.figure(figsize=(13,6))    
    for i in range(0,len(turnover_periods),2):
        
        if doy:
            start_period = turnover_periods['datetime'].iloc[i].dayofyear
            end_period = turnover_periods['datetime'].iloc[i + 1].dayofyear
        else:
            start_period = turnover_periods['datetime'].iloc[i]
            end_period = turnover_periods['datetime'].iloc[i + 1]
            
        if pd.notna(start_period):
            if i == 0:
                plt.axvspan(start_period, end_period, color=colors[int(np.floor(i%4)/2)],label="Fall turnover", alpha=0.2)
            elif i == 2:
                plt.axvspan(start_period, end_period, color=colors[int(np.floor(i%4)/2)],label="Spring turnover", alpha=0.2)
            else:
                plt.axvspan(start_period, end_period, color=colors[int(np.floor(i%4)/2)], alpha=0.2)
    
    plt.title('Average gas transfer velocity',fontsize=16)
    plt.plot(avg_y_gas_transfer.index, avg_y_gas_transfer.values,linewidth=0.2,c='k')
    plt.scatter(avg_y_gas_transfer.index, avg_y_gas_transfer.values, s=50, edgecolor='k',c='lightblue')
    
    
    if doy:     
        avg_y_gas_transfer['dt'] = pd.to_datetime(avg_y_gas_transfer.index, unit='D', origin='2024')
        dat = avg_y_gas_transfer.reset_index().set_index('dt')
        dat.columns.values[1] = 'k_vals'
    else:
        dat = avg_y_gas_transfer.copy()
        dat.columns.values[0] = 'k_vals'

    print(dat)
    
    resampled = dat.resample('1ME').mean()
    print(resampled)
    
    if doy:
        plt.plot(resampled.index.dayofyear, resampled.k_vals)
    else:
        plt.plot(resampled.index, resampled.values)
        
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('$k_{TE}$ [m/s]',fontsize=16)
    plt.xlabel('DOY', fontsize=16)
    plt.legend(fontsize=14)
    plt.xlim(0,365)
    
    plt.savefig(fig_path+'gas_transfer_velocity_avg_year.png', dpi=700)
    plt.show()
    return

def turnover_gas_transfer_velocity(gtv, doy=False):

    spr_gtv, aut_gtv = [], []
    spr_starts, aut_starts = [], []
    spr_ends, aut_ends = [], []
    spr_lengths, aut_lengths = [], []
    
    
    for i in range(0,len(turnover_periods),2):       
        start_period = pd.to_datetime(turnover_periods['datetime'].iloc[i])
        end_period =  pd.to_datetime(turnover_periods['datetime'].iloc[i + 1]) 
        length = (end_period - start_period).days 
        
        try:
            if start_period.strftime('%m') in ['02','03','04','05','06','07']:
                spr_starts.append(start_period)
                spr_ends.append(end_period)
                spr_lengths.append(length)
                
                s = pd.to_datetime(start_period)
                e = pd.to_datetime(end_period)
                turnover_gtv = gtv[(gtv.index > s) & (gtv.index < e)]
                            
                mean_gtv = turnover_gtv.mean().item()
                spr_gtv.append(mean_gtv)
                
            else:
                aut_starts.append(start_period)
                aut_ends.append(end_period)
                aut_lengths.append(length)
                
                s = pd.to_datetime(start_period)
                e = pd.to_datetime(end_period)
                turnover_gtv = gtv[(gtv.index > s) & (gtv.index < e)]
                mean_gtv = turnover_gtv.mean().item()
                aut_gtv.append(mean_gtv)
        
        except ValueError:
            print('probs nan')     
            

    return spr_gtv, aut_gtv, spr_lengths, aut_lengths

def plot_all_years_separately(turnover_periods, gas_transfer_velo_cooling,\
                              gas_transfer_velo_warming):
    
    all_gas_transfer_velocities = pd.concat([gas_transfer_velo_warming, gas_transfer_velo_cooling])

    spring_frames, autumn_frames, springs, autumns, spring_lengths, autumn_lengths = \
        plot_whole_turnovers_with_dates(turnover_periods, gas_transfer_velo_cooling) 
            
    frames = spring_frames + autumn_frames
        
    for s in frames:   
        try:
            start, end = s.index.values[0], s.index.values[-1]
            plot_stripes()  
                
            plt.xlabel("Time",fontsize=20)    
            plt.title('Gas transfer velocity')
            
            plt.grid(linestyle='--', color='gray',alpha=0.3)
            
            print(np.nanmean(gas_transfer_velo_cooling.values))
            print(np.nanmean(gas_transfer_velo_warming.values))
                
                
            plt.scatter(gas_transfer_velo_cooling.index, gas_transfer_velo_cooling.values,\
                    label='Lake cooling', s=size, edgecolor='k', c='lightskyblue')
            plt.scatter(gas_transfer_velo_warming.index, gas_transfer_velo_warming.values,\
                    label='Lake warming', c='red', s=size, edgecolor='k')
         
            plt.plot(gas_transfer_velo_cooling.index, gas_transfer_velo_cooling.values,\
                     label='Lake cooling',alpha=0.3)
            plt.plot(gas_transfer_velo_warming.index, gas_transfer_velo_warming.values,\
                     label='Lake warming', c='k',alpha=0.3)
            
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20, rotation=45)
            plt.ylabel('$k_{TE}$', fontsize=20)
            #plt.ylim(-0.01,0.1)
#                plt.ylim(-0.01, 1000)
                    
            plt.legend(fontsize=14, loc='upper right')
            plt.xlim(start, end)
            plt.show()
                
        except IndexError:
            print('Next')
    return
def plot_gas_transfer_velocities(gas_transfer_velo_cooling, gas_transfer_velo_warming):
    time_averaging = '1ME'
    gas_transfer_velo_cooling =  gas_transfer_velo_cooling.resample(time_averaging).mean()
    gas_transfer_velo_warming =  gas_transfer_velo_warming.resample(time_averaging).mean()
    
    avg_gas_transfer = pd.concat([gas_transfer_velo_cooling, gas_transfer_velo_warming])
    avg_m_gas_transfer = avg_gas_transfer.resample('1m').mean()
    
    plot_stripes()
    plt.xlabel("Time",fontsize=20)    
    plt.title('Gas transfer velocity')
    plt.grid(linestyle='--', color='gray',alpha=0.3)
    
    
    plt.scatter(gas_transfer_velo_cooling.index, gas_transfer_velo_cooling.values,\
            label='Lake cooling',s=size, edgecolor='k',c='lightblue')
        
    plt.scatter(gas_transfer_velo_warming.index, gas_transfer_velo_warming.values,\
            label='Lake warming', s=size, edgecolor='k', color='r')
    #plt.ylim(-50, 250)
    plt.plot(avg_gas_transfer.index, avg_gas_transfer.values, c='k',\
                linewidth=0.6)
    
    plt.scatter(avg_m_gas_transfer.index, avg_m_gas_transfer.values, c='k',\
                edgecolor='k', s=size)
    
    plt.xlim(pd.to_datetime('2012-05-01'), pd.to_datetime('2023-12-31'))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.ylabel('$k_{TE}$', fontsize=20)
    # plt.ylim(-0.01, 200)
    plt.legend(fontsize=14, loc='upper right')
    #   plt.xlim(pd.to_datetime('2017-03-20'), pd.to_datetime('2017-12-31'))
    plt.show()
    return    
def plot_dissipation_rates_yearly(turnover_periods, beta,\
                                  due_to_turbulence_1, \
                                  due_to_turbulence_2,\
                                  due_to_buoyancy):
    
    spring_frames, autumn_frames, springs, autumns, spring_lengths, autumn_lengths = \
        plot_whole_turnovers_with_dates(turnover_periods, beta) 
        
    frames = spring_frames + autumn_frames
    for s in frames:
      
        try:
            print(s)
            start = s.index.values[0]
            end = s.index.values[-1]
            print(start, end)
                
            plot_stripes()  
            plt.xlabel("Time",fontsize=20)    
            plt.title('Turbulent dissipation rate')
            
            plt.grid(linestyle='--', color='gray',alpha=0.3)
            
            plt.scatter(due_to_turbulence_2.index, due_to_turbulence_2.values,\
                        label='Warming / Wind',color='orangered',edgecolor='k',s=80)
            plt.scatter(due_to_turbulence_1.index, due_to_turbulence_1.values,\
                     label='Cooling / Wind',color='skyblue',edgecolor='k',s=80)
            plt.scatter(turbulent_dissipation_heat_out.index, \
                    turbulent_dissipation_heat_out.values,\
                    label='Cooling (sum)',color='blue',edgecolor='k',s=80)
           
            plt.scatter(due_to_buoyancy.index, due_to_buoyancy.values,\
                        label='Cooling / due to buoyancy',color='lightblue',edgecolor='k',s=80)
                
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20, rotation=45)
            plt.ylabel('$\epsilon_{TE}$', fontsize=20)
                        
            plt.legend(fontsize=14, loc='upper right')
            plt.xlim(start, end)

            plt.show()
                
        except IndexError:
            print('Next')
    return
def plot_dissipation_rates_all(turbulent_dissipation_heat_in,\
                               turbulent_dissipation_heat_out,\
                               due_to_turbulence_1,\
                               due_to_turbulence_2,\
                               due_to_buoyancy):
    plot_stripes()  
    plt.xlabel("Time",fontsize=20)    
    plt.title('Turbulent dissipation rate')
    
    plt.grid(linestyle='--', color='gray',alpha=0.3)
    #all_to_same = pd.concat([turbulent_dissipation_heat_out,\
    #                         turbulent_dissipation_heat_in])    
    
    plt.scatter(due_to_turbulence_2.index, due_to_turbulence_2.values,\
                label='Warming / Wind',color='orangered',edgecolor='k',s=80)
    plt.scatter(due_to_turbulence_1.index, due_to_turbulence_1.values,\
             label='Cooling / Wind',color='skyblue',edgecolor='k',s=80)
    plt.scatter(turbulent_dissipation_heat_out.index, \
            turbulent_dissipation_heat_out.values,\
            label='Cooling (sum)',color='blue',edgecolor='k',s=80)
    plt.scatter(due_to_buoyancy.index, due_to_buoyancy.values,\
                label='Cooling / due to buoyancy',color='lightblue',s=20)
    
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.ylabel('$\epsilon_{TE}$', fontsize=20)
    plt.legend(fontsize=14, loc='upper right')
   # plt.xlim(start, end)
    plt.show()
    return
def more_plots(turbulent_dissipation_heat_in, turbulent_dissipation_heat_out):
    plot_stripes()
    plt.xlabel("Time",fontsize=20)    
    plt.title('Turbulent dissipation rate')
    plt.ylabel('$\epsilon_{TE}$')
    plt.grid(linestyle='--', color='gray',alpha=0.3)

    plt.scatter(turbulent_dissipation_heat_out.index, turbulent_dissipation_heat_out.values,\
             label='Lake cooling', s=2)
    plt.scatter(turbulent_dissipation_heat_in.index, turbulent_dissipation_heat_in.values,\
             label='Lake warming', c='k', s=2)
    full_df = pd.concat([turbulent_dissipation_heat_out, turbulent_dissipation_heat_in])
    
    full_df =  full_df.resample('1m').mean()
    
    plt.scatter(full_df.index, full_df.values, c='r')
    plt.xlim(pd.to_datetime('2012-05-01'), pd.to_datetime('2023-12-31'))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
        
    plt.ylim(0, 1e-4)    #plt.xlim(pd.to_datetime('2017-03-20'), pd.to_datetime('2017-12-31'))
    plt.ylabel('$\epsilon_{TE}$',fontsize=20)
    plt.legend(fontsize=14, loc='upper right')
    plt.show()

    return full_df

if __name__ == '__main__':
    """
    DOWNLOAD DATA FROM SMARTSMEAR / AVAA
    
    FLUX DATA should contain following columns:
    ['KVJ_EDDY.LE_LI75', 'KVJ_EDDY.LE_LI72', 'KVJ_EDDY.F_CO2_LI75',
           'KVJ_EDDY.F_CO2_PICA', 'KVJ_EDDY.F_CO2_LI72', 'KVJ_EDDY.MO_length',
           'KVJ_EDDY.LE_PICA', 'KVJ_EDDY.H', 'KVJ_EDDY.Qc_CO2_LI75',
           'KVJ_EDDY.Qc_CO2_LI72', 'KVJ_EDDY.Qc_CO2_PICA', 'KVJ_EDDY.tau',
           'KVJ_EDDY.Qc_CH4_PICA', 'KVJ_EDDY.Qc_H', 'KVJ_EDDY.Qc_LE_PICA',
           'KVJ_EDDY.F_H2O_LI72', 'KVJ_EDDY.F_H2O_PICA', 'KVJ_EDDY.F_CH4_PICA',
           'KVJ_EDDY.Qc_LE_LI75', 'KVJ_EDDY.F_H2O_LI75', 'KVJ_EDDY.Qc_tau',
           'KVJ_EDDY.Qc_LE_LI72', 'KVJ_EDDY.u_star']
    
    RADIATION DATA should contain following columns:
        ['KVJ_META.LWout', 'KVJ_META.Glob', 'KVJ_META.RGlob', 'KVJ_META.LWin']
    
    """
    
# =============================================================================
#   READ IN DATASETS 
# =============================================================================
    dat = pd.read_pickle(data_out + "flux_data_kuivajarvi.pkl")
    rad_dat = pd.read_pickle(data_out +"data_radiation_data.pkl")


# =============================================================================
#   Quality control
# =============================================================================
    dat = dat[dat['KVJ_EDDY.Qc_H'] != 2]
    dat = dat[dat['KVJ_EDDY.Qc_LE_LI72'] != 2]
    dat = dat[dat['KVJ_EDDY.LE_LI72'] < 500]
    rad_dat = rad_dat[rad_dat['KVJ_META.Glob'] <550]

    # Set time averaging 
    time_averaging = '1D'    
    dat_resampled = dat.resample(time_averaging).mean()
    rad_dat = rad_dat.resample(time_averaging).mean()
    
    
# =============================================================================
#   COMPUTE NET RADIATION FOR GAS TRANSFER VELOCITY COMPUTATION
# =============================================================================
    net_rad = rad_dat['KVJ_META.Glob'] - rad_dat['KVJ_META.RGlob'] \
                + rad_dat['KVJ_META.LWin'] - rad_dat['KVJ_META.LWout'] \
                + dat_resampled['KVJ_EDDY.H'] + dat_resampled['KVJ_EDDY.LE_LI72']
        
    # Compute buoyancy flux [m**2 / s**3]
    beta = g * alpha * net_rad / (density * specific_heat)
    
    
    df = pd.DataFrame(beta, columns=['beta'])
    df['wstar'] = dat_resampled['KVJ_EDDY.u_star'] / 28
    
    # Mask warming / cooling values
    beta_heat_in = df[df.beta >= 0]
    beta_heat_out = df[df.beta < 0] 

    # Compute turbulent dissipation rates
    # ---terms separately
    due_to_buoyancy = c2 * abs(beta_heat_out.beta)
    due_to_turbulence_1 = (c1 * beta_heat_out.wstar**3) / (kappa*z)
    due_to_turbulence_2 = (c3 * beta_heat_in.wstar**3) / (kappa*z)
    
    # --- dissipation rates separately for heat in and out
    turbulent_dissipation_heat_out = due_to_turbulence_1 + due_to_buoyancy
    turbulent_dissipation_heat_in = due_to_turbulence_2.copy()

    # --- save the buoyancy term to file
    due_to_buoyancy.to_csv(data_out + 'due_to_buoyancy.csv')
    

# =============================================================================
#   PLOT TURBULENT DISSIPATION RATES
# =============================================================================
    # Plotting full time series
    plot_dissipation_rates_all(turbulent_dissipation_heat_in,\
                                   turbulent_dissipation_heat_out,\
                                   due_to_turbulence_1,\
                                   due_to_turbulence_2,\
                                   due_to_buoyancy)
    # Plotting years separately
    plot_dissipation_rates_yearly(turnover_periods, beta,\
                                 due_to_turbulence_1,\
                                 due_to_turbulence_2,\
                                 due_to_buoyancy)
        
    full_df = more_plots(turbulent_dissipation_heat_in, turbulent_dissipation_heat_out)
    
# =============================================================================
#     COMPUTE GAS TRANSFER VELOCITIES
# ============================================================================= 
    # Keeping cooling and warming separately
    gas_transfer_velo_cooling = c4 * (turbulent_dissipation_heat_out * nu)**(1/4)*\
          (1036)**(-1/2) * 3.6 * 1e5
    gas_transfer_velo_warming = c4 * (turbulent_dissipation_heat_in * nu)**(1/4)*\
          (1036)**(-1/2) * 3.6 * 1e5

# =============================================================================
#     CONCAT
# =============================================================================   
    ds_cool = gas_transfer_velo_cooling.copy()
    ds_cool.to_csv(data_out+'gas_transfer_velo_cool_vals.csv')
    
    ds_warm = gas_transfer_velo_warming.copy()
    ds_warm.to_csv(data_out+'gas_transfer_velo_warm_vals.csv')

    # Merging cooling and warming subsets to same dataset
    full_y = pd.concat([gas_transfer_velo_cooling, gas_transfer_velo_warming]).reset_index()
    
    ds = full_y.copy()
    ds['timestamp'] = pd.to_datetime(full_y.timestamp)
    ds = ds.set_index('timestamp')
    
    # Saving gas transfer velocities to file
    ds.to_csv(data_out + 'gas_transfer_velo_vals.csv')
    
# =============================================================================
#   PLOTTING GAS TRANSFER VELOCITY
# =============================================================================
    plot_gas_transfer_velocity(ds, False)
    spr, aut, slen, alen = turnover_gas_transfer_velocity(ds, False)
# =============================================================================
#     PLOT GAS TRANSFER VELOCITY AVG YEAR
# =============================================================================

    full_y['doy'] = full_y.timestamp.dt.dayofyear
    full_y = full_y.drop(columns='timestamp')
    
    avg_y_gas_transfer = full_y.groupby('doy').mean()
    
    print(avg_y_gas_transfer.index.values)
    avg_y_gas_transfer = avg_y_gas_transfer[(avg_y_gas_transfer.index.values > 40) & \
                                            (avg_y_gas_transfer.index.values < 360)]
    print(avg_y_gas_transfer)
   # print(k)

    plot_gas_transfer_velocity(avg_y_gas_transfer, True)    
    plot_gas_transfer_velocities(gas_transfer_velo_cooling, gas_transfer_velo_warming)

# =============================================================================
    plot_all_years_separately(turnover_periods, gas_transfer_velo_cooling,\
                              gas_transfer_velo_warming)
        
        