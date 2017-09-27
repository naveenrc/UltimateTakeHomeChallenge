import os
import pandas as pd
from pandas.io.json import json_normalize
import json
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18, 8)


def logins_df():
    curdir = os.path.dirname(__file__)
    logins_path = os.path.join(curdir, './data/') + 'logins.json'
    # Read file
    str_data = json.load(open(logins_path))
    logins = json_normalize(str_data, record_path='login_time')
    logins.columns = ['date']
    logins['val'] = 1
    logins['date'] = pd.to_datetime(logins['date'])
    logins.set_index('date', inplace=True)
    return logins


def resample(df, interval):
    temp_resample = df.resample(interval).count()
    return temp_resample


def custom_plot(df, axis, title, y_label=''):
    plt.subplot(axis)
    plt.plot(df['val'])
    plt.ylabel(y_label)
    plt.title(title)


logins = logins_df()
min15 = resample(logins, '15Min')
hourly = resample(logins, '60Min')
daily = resample(logins, 'D')
monthly = resample(logins, 'M')

# 15 Min
custom_plot(min15, 221, "15 min time interval", "# of logins")
# Hourly
custom_plot(hourly, 222, "Hourly time interval")
# Daily
custom_plot(daily, 223, "Daily time interval", "# of logins")
# Monthly
custom_plot(monthly, 224, "Monthly time interval")
plt.show()

# 15 Min
custom_plot(min15[97:192], 221, "15 min time interval from Jan-2 to Jan-3", "# of logins")
# Hourly
custom_plot(hourly[:72], 222, "Hourly time interval from Jan-2 to Jan-4")
# Daily
custom_plot(daily[:14], 223, "Daily time interval from Jan-2 to Jan-14", "# of logins")
# Monthly
custom_plot(monthly, 224, "Monthly time interval")
plt.show()

# auto-correlation
print('Correlation between 15min, 15min lag(previous 15min)')
print(pd.DataFrame({'actual': min15['val'], 'lag': min15['val'].shift()}).corr())
