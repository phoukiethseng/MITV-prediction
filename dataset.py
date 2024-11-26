import pandas as pd
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo
from datetime import datetime
from datetime import timedelta
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import torch


class MITVDataset(Dataset):
    def __init__(self):
        sample = pd.read_csv('./data.csv', header=0, index_col=False,
                             usecols=['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main',
                                      'weather_description', 'date_time'])
        label = pd.read_csv('./data.csv', header=0, index_col=False,
                            usecols=['traffic_volume'])
        # metro_interstate_traffic_volume = fetch_ucirepo(id=492)
        # index = pd.DatetimeIndex(metro_interstate_traffic_volume.data.features['date_time'].copy())
        index = pd.DatetimeIndex(sample['date_time'].copy()).normalize()
        day_index = pd.Series(index.copy()).map(lambda dt: dt.day)
        month_index = pd.Series(index.copy()).map(lambda dt: dt.month)
        self.date_index_dict = index.copy().unique()
        # X = metro_interstate_traffic_volume.data.features
        X = sample
        X.drop(labels='date_time', axis=1, inplace=True)
        X.set_index(index, inplace=True)
        # y = metro_interstate_traffic_volume.data.targets
        y = label
        s = MinMaxScaler()
        y = pd.Series(s.fit_transform(y).squeeze(), index=index)

        X.insert(loc=1, column='day', value=day_index.to_numpy())
        X.insert(loc=1, column='month', value=month_index.to_numpy())

        # Skip the first day data since it is not full 24 hour and will be a pain to process that
        # X = X.loc['2012-10-03 00:00:00':].copy()
        # y = y.iloc[15:]

        # Drop 'weather_main'
        X.drop(labels='weather_main', axis=1, inplace=True)

        self.weather_intensity_dict = {
            'sky is clear': 0,
            'Sky is Clear': 0,  # Both 'sky is clear' instances have the same intensity value
            'few clouds': 1,
            'scattered clouds': 2,
            'broken clouds': 3,
            'overcast clouds': 4,
            'light rain': 5,
            'light intensity drizzle': 6,
            'drizzle': 7,
            'light intensity shower rain': 8,
            'light rain and snow': 9,
            'light shower snow': 10,
            'light snow': 11,
            'mist': 12,
            'haze': 13,
            'fog': 14,
            'smoke': 15,
            'shower drizzle': 16,
            'moderate rain': 17,
            'proximity shower rain': 18,
            'proximity thunderstorm with drizzle': 19,
            'thunderstorm with light drizzle': 20,
            'thunderstorm with drizzle': 21,
            'thunderstorm with light rain': 22,
            'thunderstorm': 23,
            'proximity thunderstorm': 24,
            'proximity thunderstorm with rain': 25,
            'thunderstorm with rain': 26,
            'heavy intensity drizzle': 27,
            'heavy rain': 28,
            'very heavy rain': 29,
            'sleet': 30,
            'snow': 31,
            'shower snow': 32,
            'freezing rain': 33,
            'heavy snow': 34,
            'thunderstorm with heavy rain': 35,
            'heavy intensity rain': 36,
            'SQUALLS': 37
        }
        X.replace({'weather_description': self.weather_intensity_dict}, inplace=True)
        self.start_datetime = self.date_index_dict[0]
        self.end_datetime = self.date_index_dict[-1]
        self.sample_size = (self.end_datetime - self.start_datetime).days

        self.ohenc = OneHotEncoder(sparse_output=False)
        X = X.fillna('No Holiday')
        onehotencoding_feature = self.ohenc.fit_transform(pd.DataFrame(X[['holiday', 'day', 'month']].values))
        S = X.drop(labels=['holiday', 'day', 'month'], axis=1, inplace=False)
        S = np.hstack((onehotencoding_feature, S))
        self.scaler = MinMaxScaler()
        self.scaler.fit(S)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.date_index_dict)

    def __getitem__(self, idx):
        # time_delta = timedelta(days=idx)
        # t = self.start_datetime + time_delta
        # from_dt, to_dt = t, t + timedelta(hours=23)
        # to_str = lambda t: t.strftime('%Y-%m-%d %H:%M:%S')
        # X, y = self.X[to_str(from_dt):to_str(to_dt)], self.y.loc[to_str(from_dt):to_str(to_dt)]
        i = self.get_time_string(self.date_index_dict[idx])
        # print(f"idx: {idx}, date: {i}")
        X, y = self.X.loc[i].copy(), self.y.loc[i].copy()
        is_dataframe = type(X) is pd.core.frame.DataFrame
        seq_len = X.to_numpy().shape[0] if is_dataframe else 1
        if seq_len == 1: print(f"date {i} has seq_len of 1")
        onehotencoding_feature = self.ohenc.transform(
            X[['holiday', 'day', 'month']].values if seq_len > 1 else np.array([['No Holiday', X['day'], X['month']]]))
        X.drop(labels=['holiday', 'day', 'month'], axis=1 if is_dataframe else 0, inplace=True)
        return torch.tensor(self.scaler.transform(np.hstack(
            (onehotencoding_feature, X.to_numpy() if seq_len > 1 else X.to_numpy().reshape(-1, X.shape[0])))), dtype=torch.float32), torch.tensor(
            y.to_numpy() if seq_len > 1 else y.to_numpy().reshape(-1, y.shape[0]), dtype=torch.float32), torch.tensor(
            [seq_len], dtype=torch.float32)

    def get_time_string(self, dt):
        return dt.strftime('%Y-%m-%d')


if __name__ == '__main__':
    dataset = MITVDataset()
    print(dataset.__getitem__(515))
