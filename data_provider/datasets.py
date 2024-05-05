import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset, DataLoader
import torch

class TSDataset(Dataset):
    def __init__(self, data, window_size, horizon, stride=1):
        self.data = torch.tensor(data, dtype=torch.float32)  
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.window_size - self.horizon) // self.stride + 1

    def __getitem__(self, idx):
        actual_idx = idx * self.stride
        x = self.data[actual_idx:actual_idx + self.window_size]
        y = self.data[actual_idx + self.window_size:actual_idx + self.window_size + self.horizon]
        # y_mean = y.mean(dim=0)
        x = x.permute(1, 0)
        y = y.permute(1, 0)
        return x, y

class SP500Dataset:
    def __init__(self):

        sp500_price_df = pd.read_csv('data_provider/Datasets/sp500/sp500_index.csv', index_col='Date')
        stocks_price_df = pd.read_csv('data_provider/Datasets/sp500/sp500_stocks.csv', index_col='Date')[['Symbol','Close']]

        data_dict = {}
        date_index = next(iter(stocks_price_df.groupby('Symbol')))[1].index
        for symbol, data in stocks_price_df.groupby('Symbol'):
            data_dict[symbol] = data['Close']
        stocks_price_df = pd.DataFrame(data_dict, index=date_index)
        # drop stocks with missing values
        stocks_price_df.dropna(axis=1, inplace=True)

        self.names = stocks_price_df.columns
        self.stock_price_df = stocks_price_df
        self.sp500_price_df = sp500_price_df

        self.sp500_return_df = sp500_price_df.pct_change().dropna()
        self.stocks_return_df = stocks_price_df.pct_change().dropna()

    def get_data_numpy(self, log_return=True):
        if log_return: 
            return self.stocks_return_df.values
        else:
            return self.stock_price_df.values
    
    def get_names(self):
        return self.names

    def get_industry(self):
        stock_info_df = pd.read_csv('data_provider/Datasets/sp500/sp500_companies.csv', usecols=['Symbol','Sector'])
        stock_info_df = stock_info_df[stock_info_df['Symbol'].isin(self.names)]
        return stock_info_df.set_index('Symbol')
    
def train_test_split(data, train_size=0.8):
    # data: [n_timestamps, n_stocks]
    train_size = int(len(data) * train_size)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data