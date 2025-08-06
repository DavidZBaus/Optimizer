import cvxpy as cvx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.linalg import cholesky

import sys
import os

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
        use_slippage: bool, 
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
        use_slippage: bool
            False to not take into account slippage in the optimizer.
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
        self.use_slippage = use_slippage # True if you want to incorporate slippage in the backtester
        self.forecast_periods = forecast_periods # Total number of forecasted periods
        self.forecast_length = forecast_length # Total ahead forecast in hours

        self.build_optimizer(self.asset_universe)

        
    def build_optimizer(self, cur_tickers: np.ndarray) -> None:
        """
        Builds the optimization problem with cvxpy.

        Parameters
        ----------
        cur_tickers: np.ndarray
            Currently traded tickers
        """

        # Minimum yearly return to trading period length transformation
        self.min_period_return = (1 + self.min_yearly_return)

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
        self.portfolio_holdings_term = cvx.Parameter(len(cur_tickers), name="portfolio_holdings_term")
        self.inverse_size_param = cvx.Parameter(nonneg=True, value=1/self.size.value, name="inverse_size_param")
        self.turnover_limit = cvx.Parameter(len(cur_tickers), nonneg=True, name="turnover_limit")

        # Variables and Parameters used in use_slippage mode
        if self.use_slippage:
            # Parameters corresponding to metrics and measurements
            self.index_prices = cvx.Parameter(len(cur_tickers), nonneg=True, name="index_prices")

            # Parameters used for DPP compliance
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

        # Total slippage (Computed inside optimizer for now!)
        total_slippage = 0
        if self.use_slippage:
            diff = self.weight - self.current_portfolio

            base_slippage = self.pct_spreads @ abs_diff

            total_slippage = base_slippage

        # Total Return after costs
        return_after_costs = return_before_costs - total_taker_fees - total_funding_fees - total_slippage

        # Total Turnover
        total_turnover = cvx.norm1(self.weight - self.current_portfolio)

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
        if self.use_slippage:
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

        self.build_optimizer(cur_tickers)
    
    def optimize(
        self, 
        predicted_returns: np.ndarray,
        predicted_funding_rates: np.ndarray, 
        current_portfolio: np.ndarray, 
        index_prices: np.ndarray, 
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
        index_prices: np.ndarray
            Index Prices
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
        self.portfolio_holdings_term.value = current_portfolio * index_prices

        # Update Variables and Parameters used in use_slippage mode
        if self.use_slippage:
            # Update Parameters corresponding to metrics and measurements
            self.index_prices.value = index_prices

            # Update Parameters used for DPP compliance
            self.pct_spreads.value = ((index_prices + 0.5 * spreads) / index_prices - 1)

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

    def backtest(
        self, 
        initial_size: float,
        predicted_returns: pd.DataFrame, 
        betas: pd.DataFrame, 
        index_prices: pd.DataFrame, 
        volumes: pd.DataFrame, 
        funding_rates: pd.DataFrame,
        resid_returns: pd.DataFrame, 
        factor_returns: pd.DataFrame, 
        factor_exposures: pd.DataFrame,  
        model_error: pd.Series,
        plot_results: bool,
        liquidity: pd.DataFrame | None = None
    ) -> tuple[float, float, float]:
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
            - Columns: ['TICKERS': str, 'beta_clip' float] 
        index_prices: pd.DataFrame
            Index Prices

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'index_price' float] 
        volumes: pd.DataFrame
            USD Volumes

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'volume' float] 
        funding_rates: pd.DataFrame
            Funding Rates

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'funding_rate' float] 
        resid_returns: pd.DataFrame
            Residual returns after taking out the effect of factors

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'index_price' float] 
        factor_returns: pd.DataFrame
            Factor returns

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'SRFresret_0_1' float]
        factor_exposures: pd.DataFrame
            Factor exposures

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str] + Column for each factor: float 
        model_error: pd.Series
            Model error of the predicted returns

            - Index: pd.DatetimeIndex
            - Columns: float
        plot_results: boolean
            If to plot backtest results or not
        liquidity: pd.DataFrame, optional
            Liquidity data containing spreads

            - Index: pd.DatetimeIndex
            - Columns: ['TICKERS': str, 'spread' float]

        Returns
        -------
        float
            Sharpe Ratio.
        float
            Calmar Ratio.
        float
            Final Equity.
        """
        # Columns corresponding to factor exposures
        exposure_columns = [column_name for column_name in factor_exposures.columns if "exposure" in column_name]

        indexes = predicted_returns.index.unique()
        
        cur_holdings = pd.Series(0.0, index=self.asset_universe)

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

        # variance penalties tracker for plotting
        lambdas = []

        min_yearly_return_cached = self.min_yearly_return

        for i in range(len(indexes)-1):
            old_holdings = cur_holdings.copy()
            cur_index = indexes[i]
            next_index = indexes[i+1]

            cur_tickers = predicted_returns.loc[cur_index].TICKER.unique()

            # Convert Prediction Data into Optimizer Format
            cur_predicted_returns = predicted_returns.loc[cur_index].set_index("TICKER").sort_index()
            pivoted = cur_predicted_returns[predicted_returns.columns[1:]] 
            pivoted = pivoted.reindex(index=cur_tickers, fill_value=0)
            prediction_data = pivoted.T

            # Current and next index prices
            cur_index_prices_full = index_prices.loc[cur_index].set_index("TICKER").sort_index()["index_price"]
            cur_index_prices = index_prices.loc[cur_index][
                index_prices.loc[cur_index].TICKER.isin(cur_tickers)
            ].set_index("TICKER").sort_index()["index_price"]

            next_index_prices_full = index_prices.loc[next_index].set_index("TICKER").sort_index()["index_price"]
            next_index_prices = index_prices.loc[next_index][
                index_prices.loc[next_index].TICKER.isin(cur_tickers)
            ].set_index("TICKER").sort_index()["index_price"]

            # Current Volume data
            cur_volumes = volumes.loc[cur_index][volumes.loc[cur_index].TICKER.isin(cur_tickers)].set_index("TICKER").sort_index()["volume"]

            # Calculate Covariance based on Factor Model
            cur_factor_exposures_temp = factor_exposures.loc[cur_index][factor_exposures.loc[cur_index].TICKER.isin(cur_tickers)].set_index("TICKER").sort_index()
            cur_tickers = cur_factor_exposures_temp.index.unique()

            cur_factor_cov = np.array(factor_returns.loc[cur_index-pd.Timedelta(days=30):cur_index].cov())

            cur_resid_returns = resid_returns.loc[cur_index-pd.Timedelta(days=30):cur_index]
            cur_resid_returns = cur_resid_returns[cur_resid_returns.TICKER.isin(cur_tickers)].set_index("TICKER")
            cur_tickers = cur_resid_returns.index.unique()
            cur_resid_returns = cur_resid_returns.pivot_table(
                index=cur_resid_returns.index, columns='TICKER', values='SRFresret_0_1').fillna(0)
            
            cur_resid_var = np.diag(cur_resid_returns.var(axis=0))
            cur_factor_exposures_temp = cur_factor_exposures_temp.loc[cur_tickers]
            cur_factor_exposures = np.array(cur_factor_exposures_temp[exposure_columns])

            cov = pd.DataFrame(cur_factor_exposures @ cur_factor_cov @ cur_factor_exposures.T + cur_resid_var, index=cur_tickers, columns=cur_tickers)

            # Current betas to btc
            cur_betas = betas.loc[cur_index][betas.loc[cur_index].TICKER.isin(cur_tickers)].set_index("TICKER").sort_index()["beta_clip"]

            # Current Liquidity Data
            if self.use_slippage:
                cur_liquidity_full = liquidity.loc[cur_index].set_index("TICKER").sort_index()
                cur_liquidity = liquidity.loc[cur_index][liquidity.loc[cur_index].TICKER.isin(cur_tickers)].set_index("TICKER").sort_index()

            # Eliminate all tickers with missing data anywhere
            set_prediction_data_tickers = set(prediction_data.columns)
            set_index_prices_tickers = set(cur_index_prices.index.unique())
            set_cur_volume_tickers = set(cur_volumes.index.unique())
            set_cov_tickers = set(cov.index.unique())
            set_betas_tickers = set(cur_betas.index.unique())

            if self.use_slippage:
                set_cur_liquidity_tickers = set(cur_liquidity.index.unique())

                cur_tickers = np.array(list(set_prediction_data_tickers
                                & set_index_prices_tickers
                                & set_cur_volume_tickers
                                & set_cov_tickers
                                & set_betas_tickers
                                & set_cur_liquidity_tickers))
            else:
                cur_tickers = np.array(list(set_prediction_data_tickers
                                & set_index_prices_tickers
                                & set_cur_volume_tickers
                                & set_cov_tickers
                                & set_betas_tickers))

            prediction_data = prediction_data[cur_tickers]
            cur_index_prices = cur_index_prices.loc[cur_tickers]
            next_index_prices = next_index_prices.reindex(cur_tickers)
            cur_volumes = cur_volumes.loc[cur_tickers]
            cov = cov.loc[cur_tickers , cur_tickers]
            cur_betas = cur_betas.loc[cur_tickers]
            if self.use_slippage:
                cur_liquidity = cur_liquidity.loc[cur_tickers]

    
            # Rebuild Optimizer if dimensions change
            if len(last_tickers) != len(cur_tickers):
                self.build_optimizer(cur_tickers)


            try:
                cur_error = model_error[cur_index]
            except:
                cur_error = 0

            lambdas.append(self.lambda_0 + self.alpha * cur_error)
            

            # Run optimizer
            if self.use_slippage:
                portfolio_holdings, status = self.optimize(
                    prediction_data.values,
                    np.zeros((len(prediction_data), len(cur_tickers))),
                    (cur_holdings[cur_tickers] * cur_index_prices).values/equity[-1],
                    cur_index_prices.values,
                    cur_volumes.values,
                    equity[-1],
                    cur_tickers,
                    cur_betas.values,
                    cur_error,
                    cov.values,
                    spreads=cur_liquidity["spread"].values
                ) 
            else:
                portfolio_holdings, status = self.optimize(
                    prediction_data.values,
                    np.zeros((len(prediction_data), len(cur_tickers))),
                    (cur_holdings[cur_tickers] * cur_index_prices).values/equity[-1],
                    cur_index_prices.values,
                    cur_volumes.values,
                    equity[-1],
                    cur_tickers,
                    cur_betas.values,
                    cur_error,
                    cov.values
                )
            statuses.append(status)
            portfolio_holdings = ((portfolio_holdings * equity[-1]) / cur_index_prices_full).fillna(0)

            # Change in Portfolio Holdings
            holdings_change = (portfolio_holdings - old_holdings).fillna(0)
            holdings_change_pct = (holdings_change * cur_index_prices_full).fillna(0)/equity[-1]
            if sum(abs(holdings_change_pct)) >= self.min_portfolio_change:
                cur_holdings = portfolio_holdings

            holdings_change = (cur_holdings - old_holdings).fillna(0)
            holdings_change_pct = (holdings_change * cur_index_prices_full).fillna(0)/equity[-1]

            # Price change
            index_price_changes = (next_index_prices_full - cur_index_prices_full).fillna(0)
            cur_return = sum((cur_holdings * index_price_changes).fillna(0))

            # Current funding rates
            cur_funding_rates = funding_rates.loc[cur_index].set_index("TICKER")["funding_rate"]
            
            # Total trading costs
            cur_funding_fee = -sum((cur_funding_rates * cur_holdings*cur_index_prices_full).fillna(0))
            cur_taker_fee = np.linalg.norm((holdings_change*cur_index_prices_full).fillna(0), ord=1) * self.taker_fee
            cur_slippage_fee = 0
            if self.use_slippage:
                pct_spreads = ((cur_index_prices_full + 0.5 * cur_liquidity_full["spread"]) / cur_index_prices_full - 1).fillna(0)
                cur_slippage_fees = (pct_spreads * abs(holdings_change) * cur_index_prices_full).fillna(0)
                cur_slippage_fee = sum(cur_slippage_fees)
            
            return_after_costs = cur_return - cur_taker_fee + cur_funding_fee - cur_slippage_fee

            equity.append(equity[-1] + return_after_costs)
            equity_no_fees.append(equity_no_fees[-1] + cur_return)
            costs_incurred.append(costs_incurred[-1] - cur_taker_fee - cur_slippage_fee)
            funding_pnl.append(funding_pnl[-1] + cur_funding_fee)

            # Aggregated Statistics
            cur_aggregated_data = pd.DataFrame({"Gross Notional": abs(cur_holdings.loc[cur_tickers]) @ cur_index_prices, "EOT Notional": cur_holdings.loc[cur_tickers] @ cur_index_prices, 
                                    "Holding PnL": cur_return, "Taker Fees": -cur_taker_fee, "Slippage Fees": -cur_slippage_fee, "Funding PnL": cur_funding_fee,
                                    "Net Turnover": sum(holdings_change_pct), "Gross Turnover": sum(abs(holdings_change_pct)), "Net PnL": return_after_costs}, 
                                    index=[cur_index])
            statistics_df = pd.concat([statistics_df, cur_aggregated_data])

            # Individual Coin Statistics
            holdings_values = cur_holdings.loc[cur_tickers].values  # All holdings at once (vectorized access)
            prices_values = cur_index_prices.loc[cur_tickers].values  # Precompute index prices
            index_price_changes_values = index_price_changes.loc[cur_tickers].values  # Precompute index price changes
            slippage_fees_values = np.array([cur_slippage_fees.get(coin, 0) for coin in cur_tickers])  # Handle slippage fees
            funding_rates_values = np.array([cur_funding_rates[coin] for coin in cur_tickers])  # Funding rates

            # Preallocate arrays for results
            cur_ticker_gross_notionals = np.abs(holdings_values) * prices_values
            cur_ticker_eot_notionals = holdings_values * prices_values
            cur_ticker_holding_pnls = holdings_values * index_price_changes_values
            cur_ticker_taker_fees = -np.abs(holdings_change.loc[cur_tickers]) * self.taker_fee
            cur_ticker_slippage_fees = -slippage_fees_values if self.use_slippage else np.zeros_like(slippage_fees_values)
            cur_ticker_funding_pnls = -funding_rates_values * holdings_values
            cur_ticker_net_turnovers = holdings_change_pct.loc[cur_tickers].values
            cur_ticker_gross_turnovers = np.abs(holdings_change_pct.loc[cur_tickers].values)

            # Compute net PnLs
            cur_ticker_net_pnls = cur_ticker_holding_pnls + cur_ticker_taker_fees + cur_ticker_funding_pnls + cur_ticker_slippage_fees

            new_data = pd.DataFrame({"Gross Notional": cur_ticker_gross_notionals, "EOT Notional": cur_ticker_eot_notionals, "Holding PnL": cur_ticker_holding_pnls,
                                    "Taker Fees": cur_ticker_taker_fees, "Slippage Fees": cur_ticker_slippage_fees, "Funding PnL": cur_ticker_funding_pnls,
                                    "Net Turnover": cur_ticker_net_turnovers, "Gross Turnover": cur_ticker_gross_turnovers, "Net PnL": cur_ticker_net_pnls, "TICKER": cur_tickers}, 
                                    index=[cur_index]*len(cur_tickers))

            coin_data = pd.concat([coin_data, new_data])

            last_index = cur_index
            last_tickers = cur_tickers

        print(cur_index)
        rets_bps = 10000.0 * pd.Series(equity[::24], index=indexes[::24]).pct_change()
        stats, daily_rets_bps, return_index, drawdown_bps, rolling_1yr_vol_bps = compounded_daily_return_stats(rets_bps)
        if plot_results:
            cur_index_array = list(indexes[:-1])
            figs, ax = plt.subplots(2, figsize=(12, 8), sharex=True)

            # Plot PnL and PnL components
            ax[0].plot([indexes[0] - pd.Timedelta(hours=1)] + cur_index_array, (np.array(equity) - equity[0])/equity[0], label="Net PnL")
            ax[0].plot([indexes[0] - pd.Timedelta(hours=1)] + cur_index_array, (np.array(equity_no_fees) - equity[0])/equity[0], label="Holding PnL")
            ax[0].plot([indexes[0] - pd.Timedelta(hours=1)] + cur_index_array, np.array(costs_incurred)/equity[0], label="Total Cost")
            ax[0].plot([indexes[0] - pd.Timedelta(hours=1)] + cur_index_array, np.array(funding_pnl)/equity[0], label="Funding PnL")

            # Formatting
            ax[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax[0].yaxis.get_major_formatter().set_scientific(False)
            ax[0].yaxis.get_major_formatter().set_useOffset(False)
            ax[0].legend()
            ax[0].set_title("PnL Breakdown")

            # Rotate x-ticks
            plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=20)

            # Plot lambdas
            ax[1].plot(cur_index_array, lambdas, label="Risk Aversion", color='orange')
            ax[1].legend()
            ax[1].set_title("Risk Aversion Over Time")
            plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=20)

            # Final layout
            plt.tight_layout()
            plt.show()

            print(pd.Series(stats))
        print("Backtest Complete!")

        status_df = pd.Series(statuses, index=indexes[:-1])
        status_df.to_csv("SolutionStatus.csv")

        statistics_df.to_csv("Statistics_Aggregated.csv")
        coin_data.to_csv("Statistics.csv")

        return pd.Series(stats)["sharpe                    "], pd.Series(stats)['annual_ret/max_drawdown   '], equity[-1]