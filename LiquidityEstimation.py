import pandas as pd
import glob
import os
import sys

lib_path = os.path.abspath("/home/ubuntu/git/Baus-Research/")
if lib_path not in sys.path:
    sys.path.append(lib_path)

from src.baus_utils.data_utils import get_tardis_files

def get_liquidity_all_coins(universe, start_date, end_date):
    """
    Gathers Liquidity data for all coins in the trading universe

    Parameters
    ----------
    universe: np.ndarray
        Coins to gather data for.
    start_date: pd.TimeDate
        Start Date.
    end_date: pd.TimeDate
        End Date.

    Returns
    -------
    pd.DataFrame
        Liquidiy Data.
    """
    all_liquidity = []

    for coin in universe:
        quotes_data = get_tardis_files(
            start_date, end_date,
            'quotes', 'time-bars_1s', 'binance-futures', f"{coin}USDT",
            normalize=True, allow_missing_dates=True, verbose=False
        )

        # Dollar amount of bids/asks
        try:
            quotes_data["bid_amount"] *= quotes_data["bid_price"]
            quotes_data["ask_amount"] *= quotes_data["ask_price"]

            # Rolling liquidity measures
            quotes_data["bid_amount_fast"] = quotes_data["bid_amount"].rolling(3).mean()
            quotes_data["ask_amount_fast"] = quotes_data["ask_amount"].rolling(3).mean()
            quotes_data["bid_amount_slow"] = quotes_data["bid_amount"].rolling(60 * 15).mean()
            quotes_data["ask_amount_slow"] = quotes_data["ask_amount"].rolling(60 * 15).mean()

            # Spread and midprice
            quotes_data["spread"] = quotes_data["ask_price"] - quotes_data["bid_price"]
            quotes_data["midprice"] = (quotes_data["ask_price"] + quotes_data["bid_price"]) / 2

            # Add Ticker label
            quotes_data["TICKER"] = coin

            # Keep only necessary columns
            quotes_data = quotes_data[[
                "bid_price", "ask_price", "spread", "midprice",
                "bid_amount_fast", "ask_amount_fast",
                "bid_amount_slow", "ask_amount_slow",
                "TICKER"
            ]]
        except:
            quotes_data = None

        all_liquidity.append(quotes_data)

    # Combine and sort
    liquidity = pd.concat(all_liquidity)
    liquidity = liquidity.dropna()
    liquidity = liquidity.reset_index()  # moves index (e.g. timestamp) to a column
    liquidity = liquidity.sort_values(by=["date", "TICKER"])
    liquidity = liquidity.set_index("date")

    return liquidity
