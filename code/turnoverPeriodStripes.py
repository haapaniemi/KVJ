"""
Analysis of atmosphere-surface interactions and feedbacks / Hyytiälä 2024
analysis scripts

@author: Gunnar Thorsen Liahjell (gunnartl@uio.no)

Plotting script for turnover period stripes on calendar year.
"""
turnover_periods = pd.read_pickle("data/turnover_periods.pkl") #this is the easiest to use
#turnover_periods = pd.read_read("data/turnover_periods.csv")  # this needs to be structured so that it hase a column of datetime.objects

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot("TIMESERIES DATA HERE")
colors = ["#F18C0E","#0EF18C"]
plt.xlabel("Time [Years since birth of Jesus Christ - Gregorian calendar]",fontsize=20)

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
