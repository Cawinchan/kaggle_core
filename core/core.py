import pandas as pd
import time
import numpy as np
import datetime as dt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import minmax_scale, OneHotEncoder, LabelBinarizer
import platform
import os


def csv_to_df(csv):
    if csv == stores:
        df = pd.read_csv(csv,
                         dtype={'store_nbr': np.int64, 'city': str, 'state': str, 'type': str, 'cluster': np.int64})
    elif csv == unit_sales:
        df = pd.read_csv(csv, dtype={'store_nbr': np.int64, 'date': object, 'transactions': np.int64})

    elif csv == holidays_events:
        df = pd.read_csv(csv)

    elif csv == bc:
        df = pd.read_csv(csv, dtype={'date': object, 'PCOCOUSDM': np.float64, 'PBANSOPUSDM': np.float64})

    elif csv == oil:
        df = pd.read_csv(csv, dtype={'date': object, 'dcoilwtico': np.float64})

    elif csv == items:
        df = pd.read_csv(csv,
                         dtype={'item_nbr': np.float32, 'family': str, 'class': np.int16, 'perishable': np.int16})

    elif csv == e:
        df = pd.read_csv(csv, nrows=1000000,
                         dtype={'id': np.int32, 'date': object, 'store_nbr': np.int32, 'item_nbr': np.int32,
                                'unit_sales': np.float32, 'onpromotion': str})
    else:
        df = pd.read_csv(csv)

    return df


def merger(df1, df2, on, how):
    merged = df1.merge(df2, on=on, how=how)
    merged.fillna(0, inplace=True)
    gbcollector(df1, df2)
    return merged


def concatnator(*dfs, axis):
    concatnated = pd.concat([*dfs], axis=axis)
    concatnated.fillna(0, inplace=True)
    gbcollector(*dfs)
    return concatnated


def binarize(df):
    temp = pd.DataFrame()
    removal_list = []
    for column in df:
        if column == 'date':
            pass
        elif column == 'description':
            del df[column]
        elif df[column].dtype == object:
            lb = LabelBinarizer()
            lb.fit(df[column])
            transformed = lb.transform(df[column])
            new_df = pd.DataFrame(transformed)
            new_df.rename(columns=dict(map(lambda x: (x, str(column) + "_" + str(x)), new_df)),
                                 inplace=True)
            temp = concatnator(temp, new_df, axis=1)
            removal_list.append(column)
        else:
            pass
    df = concatnator(df, temp, axis=1)
    for column in removal_list:
        del df[column]
    print("replaced columns", removal_list)
    return df


def get_stores_df(df):
    df = csv_to_df(df)
    df_bi = binarize(df)
    return df_bi


def gbcollector(*dfs):
    lst = [*dfs]
    del lst


def merge_stores_with_unit_sales(df1, df2):
    new_name_df = merger(df1, df_df2, on='store_nbr', how='inner')
    new_name_df['date'] = pd.DatetimeIndex(new_name_df['date'])
    return new_name_df


def get_dates_df():
    beg = pd.Timestamp('2012-01-01')
    end = pd.Timestamp('2017-12-31')
    days = pd.DatetimeIndex(start=beg, end=end, freq='D')
    dates = pd.DataFrame(days)
    dates.rename(columns={0: 'date'}, inplace=True)
    dates['date'] = dates['date'].astype(str)
    dates.set_index('date', inplace=True)
    return dates


def get_bc_df(df):
    df = csv_to_df(df)
    df.rename(index=str, columns={"DATE": "date", "PBANSOPUSDM": "gp_bananas", "PCOCOUSDM": "gp_cocas"}, inplace=True)
    df = df.set_index('date')
    return df


def get_oil_df(df):
    df = csv_to_df(df)
    df = df.set_index('date')
    return df


def concat_dates_oil_bc(*df):
    new_name_df = concatnator(*df, axis=1)
    interpolate(new_name_df)
    return new_name_df


def interpolate(new_name_df):
    new_name_df.reset_index(inplace=True)
    new_name_df.rename(columns={'index': 'date'}, inplace=True)
    new_name_df['date'] = pd.DatetimeIndex(new_name_df['date'])
    new_name_df.set_index('date', inplace=True)
    new_name_df['dcoilwtico'].interpolate(inplace=True, limit_direction='both',
                                          method='time')  # interpolate function because a lot of dates don't have oil values
    new_name_df['gp_bananas'].interpolate(inplace=True, limit_direction='both', method='time')
    new_name_df['gp_cocas'].interpolate(inplace=True, limit_direction='both', method='time')


def get_wages_df():
    beg = pd.Timestamp('2012-01-01')
    end = pd.Timestamp('2017-12-31')
    wages = pd.DatetimeIndex(start=beg, end=end, freq='SM')
    wages = pd.DataFrame(wages)
    wages['wages'] = (1)
    wages.rename(columns={0: 'date'}, inplace=True)
    wages['date'] = pd.DatetimeIndex(wages['date'])
    wages.set_index('date', inplace=True)
    return wages


def get_weekends_df():
    beg = pd.Timestamp('2012-01-01')
    end = pd.Timestamp('2017-12-31')
    saturday = pd.DatetimeIndex(start=beg, end=end, freq='W-SAT')
    sunday = pd.DatetimeIndex(start=beg, end=end, freq='W-SUN')
    sunday = pd.DataFrame(sunday)
    saturday = pd.DataFrame(saturday)
    sunday['nowork_sun'] = (1)
    saturday['nowork_sat'] = (1)
    sunday.rename(columns={0: 'date'}, inplace=True)
    sunday['date'] = pd.DatetimeIndex(sunday['date'])
    sunday.set_index('date', inplace=True)
    saturday.rename(columns={0: 'date'}, inplace=True)
    saturday['date'] = pd.DatetimeIndex(saturday['date'])
    saturday.set_index('date', inplace=True)
    df_weekend = concatnator(saturday, sunday, axis=1)
    return df_weekend


def concat_wages_weekend(*df):
    ww = concatnator(*df, axis=1)
    ww.reset_index(inplace=True)
    ww['date'] = pd.DatetimeIndex(ww['date'])
    ww.set_index('date', inplace=True)
    return ww


def concat_date_oil_bc_wages_weekends(*df):
    dobcww = concatnator(*df, axis=1)
    dobcww.reset_index(inplace=True)
    return dobcww


def get_holiday_events_df(df):
    df = csv_to_df(df)
    df = manage_transferred_dates(df)
    df['transferred'] = np.where(df['transferred'] == 'False', 0, 1)
    df.rename(columns={'transferred': 'holiday_mf'}, inplace=True)
    df = binarize(df)
    df = df[~df['date'].duplicated(keep='first')]
    df['date'] = pd.DatetimeIndex(df['date'])
    return df


def manage_transferred_dates(df):
    df['transferred'] = df['transferred'].astype(str)
    df['type'] = df['type'].astype(str)
    for i, row in df.iterrows():
        if df['transferred'][i] == 'True':
            if df['type'][i] == 'Holiday':

                # change next value where type is transfer

                for j in range(i - 1,
                               len(
                                   df)):  # need to check downwards to find the next instance of transfered
                    if df['type'][j] == 'Transfer':
                        df['type'][j] = 'Holiday'
                        i = j
                        break

            elif df['type'][i] == 'Event':
                for j in range(i,
                               len(
                                   df)):  # need to check downwards to find the next instance of transfered
                    if df['type'][j] == 'Transfer':
                        df['type'][j] = 'Event'
                        i = j
                        break
    return df


def merge_holidays_events_with_dates_oil_bc_wages_weekends(df1, df2):
    new_name_df = merger(df1, df2, on='date', how='inner')
    gbcollector(df1, df2)
    new_name_df['date'] = new_name_df['date'].astype(object)
    new_name_df.fillna(0, inplace=True)
    new_name_df['date'] = pd.DatetimeIndex(new_name_df['date'])
    return new_name_df


def merge_holidays_events_dates_oil_bc_wages_weekends_stores_with_unit_sales(df1, df2):
    new_name_df = merger(df1, df2, on='date', how='inner')
    new_name_df.fillna(0, inplace=True)
    new_name_df['date'] = new_name_df['date'].astype(str)
    return new_name_df


def downcast(df):
    print(df.info())
    converted_int = df.select_dtypes(include=['int']).apply(pd.to_numeric, downcast='unsigned')
    converted_float = df.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
    df_obj = df.select_dtypes(include=['object']).copy()
    df = concatnator(converted_int, converted_float, df_obj, axis=1)
    gbcollector(converted_int, converted_float, df_obj)
    print(df.info())
    return df


def get_items_df(csv):
    df = csv_to_df(csv)
    df = binarize(df)
    return df


def get_training_data_df(csv):
    df = csv_to_df(csv)
    df['onpromotion'] = df['onpromotion'].astype(str)
    df['onpromotion'].replace(to_replace='nan', value=0, inplace=True)
    df['onpromotion'].replace(to_replace='False', value=0, inplace=True)
    df['onpromotion'].replace(to_replace='True', value=1, inplace=True)
    return df


def chunk_processor(chunk):
    chunk_merged1 = merger(chunk, df_hedobcwwsu, on=['date', 'store_nbr'], how='left')
    print(chunk_merged1.shape)
    chunk_merged2 = merger(chunk_merged1, get_items_df, on='item_nbr', how='left')
    print(chunk_merged2.shape)
    return chunk_merged2


def partitioner_array(df):
    chunk_combined = pd.DataFrame()
    end_row = int(len(df) / 12.5)
    first_row = 0
    constant_change = end_row
    temp = pd.DataFrame()
    for i in range(13):
        chunk = df[first_row:end_row]
        first_row = end_row
        end_row = end_row + constant_change
        yield


def get_unit_sales_df():
    return csv_to_df(unit_sales)

def merge_training_data_with_all_other_df(df_e,items):

    for partition in partitioner_array(df_e):
        e_items = partition.merge(items, on='item_nbr', how='inner')

        e_items.fillna(0, inplace=True)

        e_items.info()

        hedobcww.reset_index(inplace=True)

        hedobcww['date'] = hedobcww['date'].astype(str)

        e['date'] = e['date'].astype(str)

        eihedobcww = e_items.merge(hedobcww, on='date', how='left')
        lst = [e_items, hedobcww]
        del lst

        eihedobcww.info()

        # eihedobcww.set_index('store_nbr',inplace=True)
        # su.set_index('store_nbr',inplace=True)
        # eihedobcwwsu = pd.concat([eihedobcww,su],axis=0)

        eihedobcwwsu = eihedobcww.merge(su, on=['store_nbr', 'date'], how='left')

        input_df = eihedobcwwsu  # copy dataset so that we can keep a copy of original

        ground_truth = input_df['unit_sales']
        training_data = input_df.drop(['unit_sales'], axis=1)

        del training_data['id']
        del training_data['index']

        training_data.fillna(0, inplace=True)


TRAINING_DIRECTORY = os.getcwd() + "/data/"
stores = TRAINING_DIRECTORY + "stores.csv"
unit_sales = TRAINING_DIRECTORY + "transactions.csv"
holidays_events = TRAINING_DIRECTORY + "holidays_events.csv"
bc = TRAINING_DIRECTORY + "bc.csv"
oil = TRAINING_DIRECTORY + "oil.csv"
items = TRAINING_DIRECTORY + "items.csv"
e = TRAINING_DIRECTORY + "train.csv"

df_stores = get_stores_df(stores)
df_unit_sales = get_unit_sales_df()
df_su = merge_stores_with_unit_sales(df_stores, df_unit_sales)

df_dates = get_dates_df()
df_oil = get_oil_df(oil)
df_bc = get_bc_df(bc)
df_dobc = concat_dates_oil_bc(df_dates, df_oil, df_bc)

df_wages = get_wages_df()
df_weekends = get_weekends_df()
df_ww = concat_wages_weekend(df_wages, df_weekends)

df_dobcww = concat_date_oil_bc_wages_weekends(df_dobc, df_ww)

df_holidays_events = get_holiday_events_df(holidays_events)
df_hedobcww = merge_holidays_events_with_dates_oil_bc_wages_weekends(df_holidays_events, df_dobcww)

df_hedobcwwsu = merge_holidays_events_dates_oil_bc_wages_weekends_stores_with_unit_sales(df_hedobcww, df_su)
df_dc_hedobcwwsu = downcast(df_hedobcwwsu)

df_items = get_items_df(items)

df_e = get_training_data_df(e)

merge_training_data_with_all_other_df(df_e,df_items,df_dc_hedobcwwsu)

scaled_training_data = minmax_scale(training_data)  # minmax defaults to [0,1]

x_train, x_test, y_train, y_test = train_test_split(scaled_training_data, ground_truth, test_size=0.2)

rf_regressor = RandomForestRegressor()
ln_regressor = LinearRegression(normalize=True)

rf_regressor.fit(x_train, y_train)
rsquared = rf_regressor.score(x_test, y_test)

print("rf score is :", rsquared)

y_pred = rf_regressor.predict(x_test)

mean_squared_error(y_test, y_pred)

ln_regressor.fit(x_train, y_train)
rsquared_2 = ln_regressor.score(x_test, y_test)

print("linear_regressor_score is :", rsquared_2)

y_pred = ln_regressor.predict(x_test)

mean_squared_error(y_test, y_pred)
predicted = rf_regressor.predict(x_test)

print("show some actual and predicted values (actual,predicted)")
print(list(zip(y_test, predicted.round()))[:100])
