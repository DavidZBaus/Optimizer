import cvxpy as cvx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.linalg import cholesky

import sys
import os

from functools import reduce

# Absolute or relative path to your library
lib_path = os.path.abspath("/home/ubuntu/git/Baus-Research/")

# Add to sys.path
if lib_path not in sys.path:
    sys.path.append(lib_path)

from src.baus_utils.stat_utils import compounded_daily_return_stats

HOURS_PER_YEAR = 8760


class PortfolioOptimizer():
    def __init__(
        self, 
        asset_universe: np.ndarray, 
        max_holding: float, 
        volume_max_turnover: float, 
        lambda_0: float,
        alpha: float,  
        min_portfolio_change: float,
        min_yearly_return: float, 
        taker_fee: float, 
        forecast_periods: int, 
        forecast_length: int
    ) -> None:
        """
        Initializes the Optimizer.

        Parameters
        ----------
        asset_universe: np.ndaray
            Full asset universe.
        max_holding: float
            Maximum holdings in % of GMV per coin.
        volume_max_turnover: float
            Maximum turnover in % of coins volume.
        lambda_0: float
            Base Variance Penalty.
        alpha: float
            Growth of Variance Penalty as model error increases
        min_portfolio_change: float
            Minimum change in portfolio before we update it (reduces turnover significantly)
        min_yearly_return: float
            Minimum yearly return
        taker_fee: float
            Taker fee.
        forecast_periods: int
            Total number of forecasted periods
        forecast_length: int
            Total forecast length in hours
        """
        self.asset_universe = asset_universe # Completely asset universe
        self.max_holding = max_holding # Maximum holdings in % of GMV per coin
        self.volume_max_turnover = volume_max_turnover
        self.lambda_0 = lambda_0 # Base Variance Penalty
        self.alpha = alpha # Variance Penalty Growth
        self.min_portfolio_change = min_portfolio_change # Minimum change in portfolio before we update it (reduces turnover significantly)
        self.min_yearly_return = min_yearly_return # Minimum yearly return
        self.taker_fee = taker_fee # Taker fee
        self.forecast_periods = forecast_periods # Total number of forecasted periods
        self.forecast_length = forecast_length # Total ahead forecast in hours

        # Minimum yearly return to trading period length transformation
        self.min_period_return = self.min_yearly_return * self.forecast_length / HOURS_PER_YEAR

        self.build_optimizer(self.asset_universe)

        
    def build_optimizer(self, cur_tickers: np.ndarray) -> None:
        """
        Builds the optimization problem with cvxpy.

        Parameters
        ----------
        cur_tickers: np.ndarray
            Currently traded tickers
        """

        # Decision Variables
        self.weight = cvx.Variable(len(cur_tickers), name="weight") # Weight that gets optimized
        # Auxiliary Variables
        total_variance_variable = cvx.Variable(name="total_variance_variable")
        abs_diff = cvx.Variable(len(cur_tickers), name="abs_diff")

        # Parameters corresponding to metrics and measurements
        self.predicted_returns = cvx.Parameter((self.forecast_periods, len(cur_tickers)), name="predicted_returns")
        self.current_portfolio = cvx.Parameter(len(cur_tickers), name="current_portfolio")
        self.predicted_funding_rates = cvx.Parameter((self.forecast_periods, len(cur_tickers)), name="predicted_funding_rates")
        self.cur_betas = cvx.Parameter(len(cur_tickers), name="cur_betas")
        self.volumes = cvx.Parameter(len(cur_tickers), nonneg=True, name="volumes")
        self.size = cvx.Parameter(nonneg=True, value=1, name="size")
        
        # Parameters used for DPP compliance
        self.risk_aversion = cvx.Parameter(nonneg=True, name="risk_aversion")
        self.covariance_chol = cvx.Parameter((len(cur_tickers), len(cur_tickers)), name="covariance_chol")
        self.inverse_size_param = cvx.Parameter(nonneg=True, value=1/self.size.value, name="inverse_size_param")
        self.turnover_limit = cvx.Parameter(len(cur_tickers), nonneg=True, name="turnover_limit")
        self.pct_spreads = cvx.Parameter(len(cur_tickers), nonneg=True, name="pct_spreads") 
        
        
        ### Total Return, Taker Fees and Funding Fees ###
        
        # Total Return before costs
        return_before_costs = cvx.sum(
            self.predicted_returns @ self.weight 
        )

        # Total Taker Fee
        total_taker_fees = self.taker_fee * cvx.norm1(self.weight - self.current_portfolio)


        # Total Funding Fee (Funding in our favor set to 0 as to not trade funding rates)
        total_funding_fees = cvx.sum(
            [cvx.sum(cvx.maximum(0, cvx.multiply(self.weight, self.predicted_funding_rates[j]))) for j in range(self.forecast_periods)]
        )

        # Total slippage from bid ask spread
        diff = self.weight - self.current_portfolio

        base_slippage = self.pct_spreads @ abs_diff

        total_slippage = base_slippage

        # Total Return after costs
        return_after_costs = return_before_costs - total_taker_fees - total_funding_fees - total_slippage

        # Total Variance
        z = self.covariance_chol @ self.weight
        total_variance = cvx.sum_squares(z)

        # Optimization Problem Objective
        obj = cvx.Maximize(return_after_costs - self.risk_aversion * total_variance_variable)

        # User Settings Optimization Problem Constraints
        constr = []
        constr += [cvx.abs(cvx.sum(self.weight)) <= 0.01] # Delta neutral (up to 1%)
        constr += [cvx.abs(self.weight @ self.cur_betas) <= 0.01] # Beta neutral (up to 1%)
        constr += [cvx.norm(self.weight, 1) <= 1] # Don't exceed GMV
        constr += [cvx.abs(self.weight) <= self.max_holding] # Max Holdings
        constr += [cvx.abs(self.weight - self.current_portfolio) <= self.turnover_limit] # Max Turnover in terms of Volume
        constr += [return_after_costs >= self.min_period_return] # Minimum Return

        # Optimization constraints for auxiliary variables
        constr += [total_variance_variable >= total_variance]
        constr += [abs_diff >= diff]
        constr += [abs_diff >= -diff]

        self.problem = cvx.Problem(obj, constr)

    def update_parameters(
        self, 
        cur_tickers: np.ndarray,
        max_holding: float | None = None, 
        volume_max_turnover: float | None = None, 
        lambda_0: float | None = None, 
        alpha: float | None = None,
        min_portfolio_change: float | None = None,
        min_yearly_return: float | None = None, 
        taker_fee: float | None = None
    ) -> None: 
        """
        Updates Optimizer Parameters

        Parameters
        ----------
        cur_tickers: np.ndarray
            currently traded instruments
        max_holding: float, optional
            Maximum holdings in % of GMV per coin.
        volume_max_turnover: float, optional
            Maximum turnover in % of coins volume.
        lambda_0: float, optional
            Base Variance Penalty.
        alpha: float, optional
            Growth of Variance Penalty as Dawdown increases
        min_portfolio_change: float
            Minimum change in portfolio before we update it (reduces turnover significantly)
        min_yearly_return: float, optional
            Minimum yearly return
        taker_fee: float, optional
            Taker fee
        """

        if max_holding is not None:
            self.max_holding = max_holding 
        if volume_max_turnover is not None:
            self.volume_max_turnover = volume_max_turnover
        if lambda_0 is not None:
            self.lambda_0 = lambda_0 
        if alpha is not None:
            self.alpha = alpha 
        if min_portfolio_change is not None:
            self.min_portfolio_change = min_portfolio_change
        if min_yearly_return is not None:
            self.min_yearly_return = min_yearly_return 
        if taker_fee is not None:
            self.taker_fee = taker_fee

        # Minimum yearly return to trading period length transformation
        self.min_period_return = self.min_yearly_return * self.forecast_length / HOURS_PER_YEAR

        self.build_optimizer(cur_tickers)
    
    def optimize(
        self, 
        predicted_returns: np.ndarray,
        predicted_funding_rates: np.ndarray, 
        current_portfolio: np.ndarray, 
        mid_prices: np.ndarray, 
        volumes: np.ndarray, 
        cur_size: float,
        cur_tickers: np.ndarray, 
        cur_betas: np.ndarray, 
        cur_error: float,
        covariance: np.ndarray,
        spreads: np.ndarray | None = None, 
    ) -> tuple[pd.Series, str]:
        """
        Single Optimization.

        Parameters
        ----------
        predicted_returns: np.ndarray
            Predicted returns from your model
        predicted_funding_rates: np.ndarray
            Predicted funding_rates
        current_portfolio:  np.ndarray
            Current portfolio holdings
        mid_prices: np.ndarray
            Mid Prices
        volumes: np.ndarray
            Volumes 
        cur_tickers: np.ndarray
            Tickers currently traded
        cur_betas: np.ndarray
            Current betas against BTC
        cur_error: float
            Last days mean absolute error of predicted returns
        spreads: np.ndarray, optional
            Bid-Ask-spreads.

        Returns
        -------
        Dict
            Optimal Portfolio Holdings.
        String
            Optimizer Status.
        """    

        # Update Parameters corresponding to metrics and measurements
        self.predicted_returns.value = predicted_returns
        self.current_portfolio.value = current_portfolio
        self.predicted_funding_rates.value = predicted_funding_rates
        self.cur_betas.value = cur_betas
        self.volumes.value = volumes
        self.size.value = cur_size
        
        # Update Parameters used for DPP compliance
        self.covariance_chol.value = cholesky(covariance, lower=True)
        self.risk_aversion.value = self.lambda_0 + self.alpha * cur_error
        self.turnover_limit.value = (self.volume_max_turnover * volumes) / cur_size
        self.inverse_size_param.value = 1/cur_size
        self.pct_spreads.value = ((mid_prices + 0.5 * spreads) / mid_prices - 1)
            

        # Run Optimizer
        try:
            opt_val = self.problem.solve(solver="ECOS", warm_start=True)
        except:
            opt_val = self.problem.solve(solver="SCS", warm_start=True)


        if self.problem.status == "optimal" or self.problem.status == "optimal_inaccurate":
            solution = np.nan_to_num(self.weight.value)
        else:
            print("Solution not optimal! Keeping portfolio unchanged.")
            print(f"Optimizer Status: {self.problem.status}")
            
            solution = current_portfolio
        
        # Transform Portfolio Weights to Coin Holdings
        solution_dict = pd.Series(0.0, index=self.asset_universe)
        for idx, coin in enumerate(cur_tickers):
            solution_dict.loc[coin] = solution[idx]
        return solution_dict.fillna(0), self.problem.status

    def generate_target_positions(
        self, 
        initial_size: float,
        predicted_returns: pd.DataFrame, 
        betas: pd.DataFrame, 
        volumes: pd.DataFrame, 
        funding_rates: pd.DataFrame,
        resid_returns: pd.DataFrame, 
        factor_returns: pd.DataFrame, 
        factor_exposures: pd.DataFrame,  
        model_error: pd.Series,
        liquidity: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Runs a backtest using the portfolio optimizer

        Parameters
        ----------
        predicted_returns: pd.DataFrame
            Predicted returns from your model

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str] + Column for each forecast period: float  
        betas: pd.DataFrame
            Betas against BTC

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'beta_clip': float] 
        volumes: pd.DataFrame
            USD Volumes

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'volume': float] 
        funding_rates: pd.DataFrame
            Funding Rates

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'funding_rate': float] 
        resid_returns: pd.DataFrame
            Residual returns after taking out the effect of factors

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'index_price': float] 
        factor_returns: pd.DataFrame
            Factor returns

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str] + Column for each factor: float 
        factor_exposures: pd.DataFrame
            Factor exposures

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str] + Column for each factor: float 
        model_error: pd.Series
            Model error of the predicted returns

            - Index: pd.DatetimeIndex
            - Columns: [']
        liquidity: pd.DataFrame, optional
            Liquidity data containing spreads

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'spread': float, 'ask_price': float, 'bid_price': float]

        Returns
        -------
        pd.DataFrame
            Dataframe of Target Positios

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'target': float]
        """
        # Columns corresponding to data
        predicted_returns_columns = [column_name for column_name in predicted_returns if column_name != "TICKER"]
        exposure_columns = [column_name for column_name in factor_exposures.columns if "exposure" in column_name]
        factor_returns_columns = [column_name for column_name in factor_returns.columns if "returns" in column_name]

        # Combine all data into one dataframe
        ticker_dfs_prepped = [predicted_returns.copy(), betas.copy(), volumes.copy(), funding_rates.copy(), resid_returns.copy(), factor_exposures.copy(), liquidity.copy()]
        for df in ticker_dfs_prepped:
            df.index.name = "index"
            df = df.reset_index()

        non_ticker_dfs_prepped = [factor_returns.copy(), model_error.copy()]
        for df in non_ticker_dfs_prepped:
            df.index.name = "index"
            df = df.reset_index()

        full_ticker_df = reduce(lambda left, right: pd.merge(left, right, on=["index", "TICKER"], how="inner"), ticker_dfs_prepped).sort_index()
        full_non_ticker_df = reduce(lambda left, right: pd.merge(left, right, on=["index"], how="inner"), non_ticker_dfs_prepped).sort_index()

        full_ticker_df["mid_price"] = 0.5 * (full_ticker_df["ask_price"].ffill() + full_ticker_df["bid_price"].ffill())

        resid_returns = full_ticker_df[["SRFresret_0_1", "TICKER"]]
        resid_returns = resid_returns.pivot_table(index=resid_returns.index, columns='TICKER', values='SRFresret_0_1').fillna(0)

        indexes = predicted_returns.index.unique()
        
        cur_holdings = pd.Series(0.0, index=self.asset_universe)
        cur_holdings_qty = pd.Series(0.0, index=self.asset_universe)

        # Tracking pnl
        equity = [initial_size]
        equity_no_fees = [initial_size]
        costs_incurred = [0]
        funding_pnl = [0]

        # Tracking Optimizer Results
        statuses = []
        statistics_df = pd.DataFrame()
        coin_data = pd.DataFrame()

        last_index = indexes[-1]
        cur_index = indexes[0]
        next_index = indexes[1]

        # Used to track changes in the asset universe
        last_tickers = []

        results = []

        for i in range(24*30, len(indexes)-1):
            old_holdings = cur_holdings.copy()
            old_holdings_qty = cur_holdings_qty.copy()

            cur_index = indexes[i]
            next_index = indexes[i+1]

            cur_tickers = full_ticker_df.loc[cur_index].TICKER.unique()

            cur_ticker_data = full_ticker_df.loc[cur_index].reset_index().set_index("TICKER").sort_index()
            next_ticker_data = full_ticker_df.loc[next_index].reset_index().set_index("TICKER").sort_index()
            cur_non_ticker_data = full_non_ticker_df.loc[cur_index]

            # Convert Prediction Data into Optimizer Format
            prediction_data = cur_ticker_data[predicted_returns_columns].T

            # Current and next index prices
            cur_mid_prices = cur_ticker_data["mid_price"]
            next_mid_prices = next_ticker_data["mid_price"]

            # Current Volume data
            cur_volumes = cur_ticker_data["volume"]

            # Calculate Covariance based on Factor Model
            cur_factor_exposures_temp = cur_ticker_data[exposure_columns]

            cur_factor_cov = np.array(full_non_ticker_df[factor_returns_columns].loc[cur_index-pd.Timedelta(days=30):cur_index].cov())

            cur_resid_returns = resid_returns.loc[cur_index-pd.Timedelta(days=30):cur_index][cur_tickers]
            
            cur_resid_var = np.diag(cur_resid_returns.var(axis=0))
            cur_factor_exposures_temp = cur_factor_exposures_temp.loc[cur_tickers]
            cur_factor_exposures = np.array(cur_factor_exposures_temp[exposure_columns])

            cov = pd.DataFrame(cur_factor_exposures @ cur_factor_cov @ cur_factor_exposures.T + cur_resid_var, index=cur_tickers, columns=cur_tickers)


            # Current betas to btc
            cur_betas = cur_ticker_data["beta_clip"]

            # Current Liquidity Data
            cur_liquidity = cur_ticker_data["spread"]

    
            # Rebuild Optimizer if dimensions change
            if len(last_tickers) != len(cur_tickers):
                self.build_optimizer(cur_tickers)

            try:
                cur_error = cur_non_ticker_data["error"]
            except:
                cur_error = 0
            

            # Run optimizer
            portfolio_holdings, status = self.optimize(
                prediction_data.values,
                np.zeros((len(prediction_data), len(cur_tickers))),
                cur_holdings[cur_tickers].values,
                cur_mid_prices.values,
                cur_volumes.values,
                initial_size,
                cur_tickers,
                cur_betas.values,
                cur_error,
                cov.values,
                cur_liquidity.values
            ) 

            portfolio_holdings_usd = initial_size * portfolio_holdings
            portfolio_holdings_qty = (portfolio_holdings_usd / cur_mid_prices).fillna(0.0)

            # Change in Portfolio Holdings
            holdings_change = (portfolio_holdings - old_holdings).fillna(0)
            holdings_change_qty = (portfolio_holdings_qty - old_holdings_qty).fillna(0)

            if sum(abs(holdings_change)) >= self.min_portfolio_change:
                new_holdings = portfolio_holdings
                new_holdings_qty = portfolio_holdings_qty

            cur_holdings = (new_holdings_qty * cur_mid_prices).fillna(0) / initial_size

            temp_df = cur_holdings.reset_index()
            temp_df.columns = ["TICKER", "target"]
            temp_df["timestamp"] = cur_index

            results.append(temp_df)

            last_index = cur_index
            last_tickers = cur_tickers

        final_df = pd.concat(results, ignore_index=True)

        return final_df