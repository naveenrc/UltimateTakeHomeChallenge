import os
import pandas as pd
import json
import datetime
import numpy as np
from matplotlib.pylab import plt
plt.rcParams['figure.figsize'] = (18, 6)


# get data
def data_df():
    curdir = os.path.dirname(__file__)
    path = os.path.join(curdir, './data/') + 'ultimate_data_challenge.json'
    dicts = json.load(open(path))
    df = pd.DataFrame(dicts)
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    return df


# clean data
def transform_data(df):
    df.dropna(inplace=True)
    # cities to int
    cities = {'Winterfell': 1, 'Astapor': 2, "King's Landing": 3}
    df['city'] = df['city'].map(lambda x: cities[x])
    # devices to int
    phones = {'iPhone': 1, 'Android': 2}
    df['phone'] = df['phone'].map(lambda i: phones[i])
    df['ultimate_black_user'] = df['ultimate_black_user'].map(lambda x: int(x == 'True'))
    df['active'] = ((df['last_trip_date'] - df['signup_date']) / np.timedelta64(1, 'M')).\
        map(lambda x: 1 if x > 5 else 0)
    df.drop(['last_trip_date', 'signup_date', 'surge_pct', 'ultimate_black_user'], axis=1, inplace=True)
    return df


# calculate retention
def retentions_df(df):
    # preciding 30 days
    from_day = max(df['last_trip_date'])
    days = [from_day - datetime.timedelta(days=x) for x in range(0, 30)]
    df['retention'] = df['last_trip_date'].map(lambda x: int(x in days))
    return df


if __name__ == '__main__':
    data = data_df()
    data_clean = transform_data(data)
    ret_df = retentions_df(data)
    frac_ret = len(ret_df[ret_df['retention'] == 1])/len(ret_df)
    print("Fraction of users retained: " + str(frac_ret))
    clean_ret = retentions_df(data_clean)
    print(clean_ret.corr())

    # plt.scatter(clean_ret['avg_dist'], clean_ret['avg_surge'])
    # plt.scatter(clean_ret['trips_in_first_30_days'], clean_ret['avg_surge'])
    # plt.scatter(clean_ret['trips_in_first_30_days'], clean_ret['avg_dist'])
    # plt.show()
