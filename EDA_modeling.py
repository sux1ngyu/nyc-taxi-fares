import pandas as pd
import numpy as np
import pandas_profiling
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pyproj import Geod
import scipy

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: '%.4f' % x)


TRAIN_PATH = './input/train.csv'
TEST_PATH = './input/test.csv'

test = pd.read_csv(TEST_PATH)

def distance(lon1,lat1,lon2,lat2):
    az12,az21,dist = Geod(ellps='WGS84').inv(lon1,lat1,lon2,lat2)
    return dist
def direction(lon1,lat1,lon2,lat2):
    az12,az21,dist = Geod(ellps='WGS84').inv(lon1,lat1,lon2,lat2)
    return az12

def shared_features(df):
    """adds features that will be used by both the modelling and EDA dataframes"""
    rows = len(df)
    #these long/lat values are needed as lists to hand to the distance function
    nyc_long, nyc_lat = [-74.001541]*rows, [40.724944]*rows    
    jfk_long, jfk_lat = [-73.785937]*rows, [40.645494]*rows
    lga_long, lga_lat = [-73.872067]*rows, [40.774071]*rows
    nla_long, nla_lat = [-74.177721]*rows, [40.690764]*rows
    chp_long, chp_lat = [-73.137393]*rows, [41.366138]*rows
    exp_long, exp_lat = [-74.0375]*rows, [40.736]*rows
    pickup_long = df.pickup_longitude.tolist()
    pickup_lat = df.pickup_latitude.tolist()
    dropoff_long = df.dropoff_longitude.tolist()
    dropoff_lat = df.dropoff_latitude.tolist()


    #add features to the data
    df = df.assign(
        #time features
        year=df.pickup_datetime.dt.year,
        dayofyear=df.pickup_datetime.dt.dayofyear,
        weekday=df.pickup_datetime.dt.dayofweek,
        time=(df.pickup_datetime.dt.hour+df.pickup_datetime.dt.minute/5),
        
        #distance between pickup and dropoff, and bearing from pickup to dropoff
        distance=distance(pickup_long, pickup_lat, dropoff_long, dropoff_lat),
        direction=direction(pickup_long, pickup_lat, dropoff_long, dropoff_lat),
        #distance from locations
        pickup_dist_nyc=pd.Series(distance(pickup_long, pickup_lat, nyc_long, nyc_lat)),
        dropoff_dist_nyc=pd.Series(distance(dropoff_long, dropoff_lat, nyc_long, nyc_lat)),
        pickup_dist_jfk=pd.Series(distance(pickup_long, pickup_lat, jfk_long, jfk_lat)),
        dropoff_dist_jfk=pd.Series(distance(dropoff_long, dropoff_lat, jfk_long, jfk_lat)),
        pickup_dist_lga=pd.Series(distance(pickup_long, pickup_lat, lga_long, lga_lat)),
        dropoff_dist_lga=pd.Series(distance(dropoff_long, dropoff_lat, lga_long, lga_lat)),
        pickup_dist_nla=pd.Series(distance(pickup_long, pickup_lat, nla_long, nla_lat)),
        dropoff_dist_nla=pd.Series(distance(dropoff_long, dropoff_lat, nla_long, nla_lat)),
        pickup_dist_chp=pd.Series(distance(pickup_long, pickup_lat, chp_long, chp_lat)),
        dropoff_dist_chp=pd.Series(distance(dropoff_long, dropoff_lat, chp_long, chp_lat)),
        pickup_dist_exp=pd.Series(distance(pickup_long, pickup_lat, exp_long, exp_lat)),
        dropoff_dist_exp=pd.Series(distance(dropoff_long, dropoff_lat, exp_long, exp_lat))
    )
    return df



def exploration_features(df):
    """adds features for use in the EDA section"""
    df = shared_features(df)
    df = (
        df
        .assign(
            hour=df.pickup_datetime.dt.hour,
            close_to_airport='No',
            fare_per_km=df.fare_amount*1000/df.distance,
            direction_bucket = pd.cut(df.direction, np.linspace(-180, 180, 37)),

            #small location buckets
            pickup_long_bucket=pd.cut(df.pickup_longitude, bins=2550, labels=False),
            pickup_lat_bucket=pd.cut(df.pickup_latitude, bins=2200, labels=False),
            dropoff_long_bucket=pd.cut(df.dropoff_longitude, bins=2550, labels=False),
            dropoff_lat_bucket=pd.cut(df.dropoff_latitude, bins=2200, labels=False),


            #large location buckets
            pickup_long_bucket_big=pd.cut(df.pickup_longitude, bins=255, labels=False),
            pickup_lat_bucket_big=pd.cut(df.pickup_latitude, bins=220, labels=False),
            dropoff_long_bucket_big=pd.cut(df.dropoff_longitude, bins=255, labels=False),
            dropoff_lat_bucket_big=pd.cut(df.dropoff_latitude, bins=220, labels=False)
        )
        .drop(columns='pickup_datetime')
        .query("0 < distance")
    )

    df.loc[((df['pickup_dist_jfk']<1500) | (df['dropoff_dist_jfk']<1500)), 'close_to_airport'] = 'JFK'
    df.loc[((df['pickup_dist_lga']<1500) | (df['dropoff_dist_lga']<1500)), 'close_to_airport'] = 'LaGuardia'
    df.loc[((df['pickup_dist_nla']<1500) | (df['dropoff_dist_nla']<1500)), 'close_to_airport'] = 'Newark'  
    return df


def clean_df(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 15)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    #reverse incorrectly assigned longitude/latitude values
    df = df.assign(rev=df.dropoff_latitude<df.dropoff_longitude)
    idx = (df['rev'] == 1)
    df.loc[idx,['dropoff_longitude','dropoff_latitude']] = df.loc[idx,['dropoff_latitude','dropoff_longitude']].values
    df.loc[idx,['pickup_longitude','pickup_latitude']] = df.loc[idx,['pickup_latitude','pickup_longitude']].values
    
    #remove data points outside appropriate ranges
    criteria = (
    " 0 < fare_amount <= 500"
    " and 0 < passenger_count <= 6 "
    " and -75 <= pickup_longitude <= -72 "
    " and -75 <= dropoff_longitude <= -72 "
    " and 40 <= pickup_latitude <= 42 "
    " and 40 <= dropoff_latitude <= 42 "
    )
    df = (df
          .dropna()
          .query(criteria)
          .reset_index()
          .drop(columns=['rev', 'index'])          
         )
    return df


def load_df(nrows=None, features=None):
    #load dataframe in chunks if the number of rows requested is high (currently only using 1 million rows for faster training)
    cols = [
        'fare_amount', 'pickup_datetime','pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'passenger_count'
    ]
    df_as_list = []
    for df_chunk in pd.read_csv(TRAIN_PATH, usecols=cols, nrows=nrows, chunksize=5000000):
        df_chunk = clean_df(df_chunk) 
        if features == 'explore':
            df_chunk = exploration_features(df_chunk)
        elif features == 'model':
            df_chunk = modelling_features(df_chunk)
        else:
            df_chunk = df_chunk.drop(columns='pickup_datetime')
        df_as_list.append(df_chunk)
    df = pd.concat(df_as_list)
    return df


def get_split_sets(train):
    x = train.drop(columns=['fare_amount'])
    y = train['fare_amount'].values
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=0)
    return x_train, x_val, y_train, y_val


def lin_model(x_train, x_val, y_train, y_val):
    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return model, rmse, pred


def knn_model(x_train, x_val, y_train, y_val, neighbors):
    min_rmse = 1000
    for n in neighbors:
        knn = KNeighborsRegressor(n_neighbors=n)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        if rmse < min_rmse:
            min_rmse = rmse
            model = knn
            best_pred = pred
        print('Neighbours', n, 'RMSE', rmse)
    return model, min_rmse, best_pred


def lgbm_model(params,x_train, x_val, y_train, y_val):
    lgbm_train = lgbm.Dataset(x_train, y_train, silent=True)
    lgbm_val = lgbm.Dataset(x_val, y_val, silent=True)
    model = lgbm.train(params=params, train_set=lgbm_train, valid_sets=lgbm_val, verbose_eval=100)
    pred = model.predict(x_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return model, rmse, pred




# FINAL MODELING


def modelling_features(df):
    df = shared_features(df)
    # using alternative representation of cyclic features
    df = df.assign(
        sin_time=np.sin(2*np.pi*df['time']/24),
        cos_time=np.cos(2*np.pi*df['time']/24),
        sin_direction=np.sin(2*np.pi*df['direction']/360),
        cos_direction=np.cos(2*np.pi*df['direction']/360),
        sin_dayofyear=np.sin(2*np.pi*df['dayofyear']/365),
        cos_dayofyear=np.cos(2*np.pi*df['dayofyear']/365),
        sin_weekday=np.sin(2*np.pi*df['weekday']/6),
        cos_weekday=np.cos(2*np.pi*df['weekday']/6),
        direction_bucket=pd.cut(df['direction'], bins=37, labels=False)
        ).drop(columns=['pickup_datetime', 'time', 'direction', 'weekday', 'dayofyear'])
    return df

train = load_df(10000000, features='model')

test['pickup_datetime'] = test['pickup_datetime'].str.slice(0, 15)
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
test = modelling_features(test)

train = (train
    .query(f'{test.pickup_longitude.min()-0.1} <= pickup_longitude <= {test.pickup_longitude.max()+0.1}')
    .query(f'{test.pickup_latitude.min()-0.1} <= pickup_latitude <= {test.pickup_latitude.max()+0.1}')
    .query(f'{test.dropoff_longitude.min()-0.1} <= dropoff_longitude <= {test.dropoff_longitude.max()+0.1}')
    .query(f'{test.dropoff_latitude.min()-0.1} <= dropoff_latitude <= {test.dropoff_latitude.max()+0.1}')
)

x_train, x_val, y_train, y_val = get_split_sets(train)

x_train['fare_per_km'] = y_train*1000/(x_train.distance+5)
fares_by_direction = x_train.query('5 < distance').groupby('direction_bucket')['fare_per_km'].mean()

x_train['adj_dist'] = [fares_by_direction[i] for i in x_train.direction_bucket]*x_train.distance/fares_by_direction.max()
x_val['adj_dist'] = [fares_by_direction[i] for i in x_val.direction_bucket]*x_val.distance/fares_by_direction.max()
test['adj_dist'] = [fares_by_direction[i] for i in test.direction_bucket]*test.distance/fares_by_direction.max()

x_train = x_train.drop(columns=['fare_per_km', 'direction_bucket'])
x_val = x_val.drop(columns=['direction_bucket'])
x_test = test.drop(columns=['key', 'direction_bucket'])

lin_final_model, lin_final_rmse, lin_final_pred = lin_model(x_train, x_val, y_train, y_val)


knn_cols = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
k_choices = [18,24,30,40]
knn_final_model, knn_final_rmse, knn_final_pred = knn_model(x_train[knn_cols], x_val[knn_cols], y_train, y_val, k_choices)


lgbm_params = {
    'objective': 'regression',
    'boosting': 'gbdt',
    'reg_sqrt': True,
    'learning_rate': 0.03,
    'num_leaves': 1200,
    'max_depth': -1,
    'max_bin': 5000,
    'num_rounds': 5000,
    'early_stopping_round': 50,
    'metric': 'rmse'
}
lgbm_final_model, lgbm_final_rmse, lgbm_final_pred = lgbm_model(lgbm_params, x_train, x_val, y_train, y_val)

print('Linear Regression RMSE', lin_final_rmse)
print('KNN RMSE', knn_final_rmse)
print('LightGBM RMSE', lgbm_final_rmse)