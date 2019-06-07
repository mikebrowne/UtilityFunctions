'''
YahooStockPricing.py

Functionality to download and update the database of daily close stock prices from Yahoo! Finance.

Public Functions:
    update_stock_price_data(ticker, path)
    data_folder_path()

'''
import os
import pandas as pd
import datetime as dt
import fix_yahoo_finance as yf


def data_folder_path():
    '''Path of the data folder for local machine.'''
    return "../AdvancedAlgorithmicTrading/data"


def update_stock_price_data(tickers):
    '''Searches the data folder for the stock(s) and updates the data if needed.'''
    cols = columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]

    if type(tickers) == str:
        tickers = [tickers]

    folder = data_folder_path()
    for ticker in tickers:
        if ticker + "/{}.csv" in os.listdir(folder):
            df = pd.read_csv(folder + "/{}.csv".format(ticker), index_col=0)[cols]
            start = df.index[-1]
        else:
            start = dt.date(2000, 1, 1)
            df = pd.DataFrame(columns=cols)

            df.index.rename("Date", inplace=True)

        end = dt.date.today()

        if start != end:
            print("Downloading data for {} from {} to {}".format(ticker,
                                                                 start,
                                                                 end))

            # Wrap fix_yahoo_finance in try, except because it is possible to reach request limit
            try:
                new_data = yf.download(ticker, start=start, end=end)
                pd.concat([df, new_data], sort=False).to_csv(folder + "/{}.csv".format(ticker))
            except Exception as e:
                print(e)
                print("\tCould not download data for {}".format(ticker))
