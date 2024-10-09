import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    
    df = pd.read_csv('waterT_0.2m.csv')
    print(df)
    df['rolling_mean'] = df['values'].rolling(window=48).mean()
    
    print(df)
    
   # df = df.dropna()
    
    instant = df['values']
    rolling = df['rolling_mean']
    
    
    print(rolling)
    print(instant)
    
    
    times = df.datetime.values
    print(times)
    
    print(len(rolling))
    print(len(instant))
    print(len(times))
    

    #plt.scatter(pd.to_datetime(times), instant, marker='o',s=0.005,c='k',alpha=0.7)
    #plt.scatter(pd.to_datetime(times), rolling, marker='x',s=0.005, c='r',alpha=0.7)
    plt.subplots(2,1, sharex=True, sharey=True)
    
    plt.plot(pd.to_datetime(times), instant, alpha=0.7, c='k',label='30min values')
    plt.plot(pd.to_datetime(times), rolling, alpha=0.7, c='r',label='daily')
    plt.legend()
    plt.show()
    
    