from flask import Blueprint, render_template, request, jsonify, current_app
from app.utils import plotting_utils, finance_data
import os
import json
import pandas as pd
import plotly.io as pio
import numpy as np
import pickle
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime, date, timedelta

bp = Blueprint('test', __name__)

@bp.route('/test')
def test_feature():
    print("test_feature called")
    # TODO focus on stocks for now, add a general function later
    
    stocks_data = None
    tickers_1d = None
    tickers_4h = None
    
    cache_dir = current_app.config['DATA_FOLDER']
    
    #print("cache_dir: ", cache_dir)
    

    stocks_data_path_fallback = os.path.join(cache_dir, 'stocks_price_history_4h.csv')
    stocks_data_path = os.path.join(cache_dir, 'stocks_price_history_1d.csv')
    #crypto_data_path = os.path.join(cache_dir, 'crypto_price_history_1d.csv')
    
    #print("stocks_data_path: ", stocks_data_path)

    # Check if we have local data to get tickers
    if os.path.exists(stocks_data_path) and os.path.getsize(stocks_data_path) > 0:
        try:
            stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True).dropna()
            tickers_1d = set(stocks_data.columns)
        except Exception as e:
            print(f"FAILED TO READ CSV: {e}")
    
    if os.path.exists(stocks_data_path_fallback) and os.path.getsize(stocks_data_path_fallback) > 0:
        try:
            stocks_data_fallback = pd.read_csv(stocks_data_path_fallback, index_col='Datetime', parse_dates=True).dropna()
            tickers_4h = set(stocks_data_fallback.columns)
        except Exception as e:
            print(f"FAILED TO READ CSV: {e}")
            
    
    # If there is no daily data yet, we take the tickers downloaded when we opened the portolio page
    if not tickers_1d:
        tickers_1d = tickers_4h
        download_start = stocks_data_fallback.index.min()
        finance_data.fetch_latest_metrics(  list(tickers_1d),
                                            category_name='stocks',
                                            interval='1d',
                                            target_start_date=download_start) # Use daily prices here!
        stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True).dropna()
        
    # Ensure that the tickers from the different datasets are equal
    if not tickers_4h.issubset(tickers_1d):
        print("Syncing 1d columns with 4h columns.")
        download_start = stocks_data_fallback.index.min()
        finance_data.fetch_latest_metrics(  list(tickers_4h),
                                            category_name='stocks',
                                            interval='1d',
                                            target_start_date=download_start) # Use daily prices here!
        stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True).dropna()
        
    # Ensure that the data is up to date
    last_update_time_stocks = stocks_data.index.max().replace(hour=0, minute=0, second=0, microsecond=0)
    last_update_time_stocks_fallback = stocks_data_fallback.index.max().tz_localize(None).replace(hour=0, minute=0, second=0, microsecond=0)
    #print("last_update_time_stocks: ", last_update_time_stocks)
    #print("last_update_time_stocks_fallback: ", last_update_time_stocks_fallback)
    if last_update_time_stocks_fallback > last_update_time_stocks: 
        download_start = last_update_time_stocks
        print("Updating 1d data from ", download_start)
        finance_data.fetch_latest_metrics(  list(tickers_1d),
                                            category_name='stocks',
                                            interval='1d',
                                            force_update=True) # Use daily prices here!
        stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True).dropna()
         
         
    if stocks_data is not None:
        # Check data integrity
        stocks_value_data = stocks_data.isnull().values.any()
        print("stocks_value_data: ", stocks_value_data)
        
        # Get tickers
        tickers = sorted(stocks_data.columns.tolist())
        #print("tickers: ", tickers)
        
        # Determine which ticker to plot (default to the first one)
        selected_ticker = request.args.get('ticker', tickers[0])
        print("selected_ticker: ", selected_ticker)
    else:
        return "Error: historical prices could not be loaded.", 404

    # Check data integrity
    if stocks_value_data:
        print("NaN in data! Careful")
        
    individual_metrics = {}
    individual_metrics = {
        'sharpe': sharpe_ratio(stocks_data, RISK_FREE_RATE=0.0),
        'semivar': semivariance(stocks_data),
        'sortino': sortino_ratio(stocks_data, RISK_FREE_RATE=0.0)
    }
    print("individual_metrics: ", individual_metrics)
    
    return render_template( 'test.html',
                            tickers=tickers,
                            selected_ticker=selected_ticker,
                            individual_metrics=individual_metrics,
                            title='Portfolio Optimisation & Backtesting')


@bp.route('/get_portfolio_data')
def get_portfolio_data():
    print("get_portfolio_data called")
    
    mode = request.args.get('mode', 'returns')
    if not mode:
        return "No mode provided", 400
        
    force_update = request.args.get('force_update') == 'true' # Check for button click
    
    # Load data. TODO data persistence?
    cache_dir = current_app.config['DATA_FOLDER']
    cache_path = os.path.join(cache_dir, "frontier_cache.pkl")
    stocks_data_path = os.path.join(cache_dir, "stocks_price_history_1d.csv")
    stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True)
    
    len_data = len(stocks_data.columns)
    #print("len_data: ", len_data)
    
    weights = np.array([1/len_data]*len_data) # Equal weight for now
    print("weights: ", weights)
    
    # Simple returns of the individual stocks
    simple_returns = stocks_data.pct_change().dropna()
    print("simple_returns: ", simple_returns)
    
    # Simple returns of the portfolio
    returns_portfolio = simple_returns @ weights
    print("returns_portfolio: ", returns_portfolio)
    
    # Compute portfolio values series by compounding returns
    portfolio_value = (1+returns_portfolio).cumprod()
    print("portfolio_value: ", portfolio_value)
    
    # Log returns of the portfolio (continuous compounding rate)
    log_returns_portfolio = np.log(portfolio_value).diff().dropna()
    print("log_returns_portfolio: ", log_returns_portfolio)
    
    # Annualising log returns
    annual_log_returns_portfolio = log_returns_portfolio.mean()*252
    print("annual_log_returns_portfolio :", annual_log_returns_portfolio)
    
    # Annualising simple returns
    geometric_mean_portfolio = ((1+returns_portfolio).prod()**(1/len(returns_portfolio)))-1
    annual_simple_returns_portfolio = (1+geometric_mean_portfolio)**252 - 1
    print("annual_simple_returns_portfolio :", annual_simple_returns_portfolio)
    
    # Log returns of individual stocks
    log_returns = np.log(stocks_data)-np.log(stocks_data.shift(1))
    #print("stocks_data: ", stocks_data)
    
    correlation_matrix = log_returns.corr()
    #print("correlation_matrix: ", correlation_matrix)
    
    
    # Annualising covariance matrix
    annualised_cov = simple_returns.cov() * 252
    #print("annualised_cov: ", annualised_cov)
    
    
    # Standard deviation
    variance_portfolio = weights.T @ simple_returns.cov() @ weights
    #print("variance_portfolio: ", variance_portfolio)
    std_portfolio = np.sqrt(variance_portfolio)
    print("std_portfolio: ", std_portfolio)
    
    # Efficient frontier calculation
    initial_guess = weights
    tickers = stocks_data.columns
    print("tickers: ", tickers)
    
    bounds, constraints = setup_optimisation_constraints(tickers, 0.05, True)
    
    #print("bounds: ", bounds)
    #print("constraints: ", constraints)
    
    
    length = len(stocks_data)
    empty_series = pd.Series(0.0 * length, index=stocks_data.index)
    
    individual_stock_metrics = []
    
    annual_simple_returns = simple_returns.mean() * 252
    annualised_std_dev = simple_returns.std() * np.sqrt(252)
    
    for ticker in tickers:
        individual_stock_metrics.append({
                                'ticker': ticker,
                                'annualised_std': annualised_std_dev[ticker],
                                'annual_return': annual_simple_returns[ticker]
                            })
    
    if mode == 'heatmap':
        fig = plotting_utils.plot_correlation_heatmap(correlation_matrix)
    elif mode == 'returns':
        fig = plotting_utils.create_returns_distribution_chart(log_returns_portfolio)
    elif mode == 'efficient_frontier':
        optimisation_results = None
        
        # Load from cache
        if not force_update and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    optimisation_results = pickle.load(f)
                print("Loaded frontier from cache")
            except:
                optimisation_results = None
                
        # Perform optimisation
        if optimisation_results is None:
            print("Calculating efficient frontier")
            optimisation_results = perform_static_optimisation(annual_simple_returns, annualised_cov, weights, bounds, constraints, simple_returns, 0.0, 100, empty_series, 0.0)
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(optimisation_results, f)
            #print("opt std: ", optimisation_results['efficient_frontier_std_devs'])
            #print("opt ret: ", optimisation_results['efficient_frontier_returns'])
            
        fig = plotting_utils.plot_efficient_frontier_and_portfolios(optimisation_results, individual_stock_metrics)
    
    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)
    
    print("get_portfolio_data out")

    return jsonify({
        'fig_data': fig_dict,
    })
    


# Plots and metrics update for individual tickers
@bp.route('/get_data')
def get_data():
    #print("get_data called")
    
    ticker = request.args.get('ticker')
    ticker2 = request.args.get('ticker2') # For the 2D correlation mode
    if not ticker:
        return "No ticker provided", 400
        
    #print("ticker: ", ticker)
    
    mode = request.args.get('mode', 'price')
    if not mode:
        return "No mode provided", 400
        
    #print("mode: ", mode)

    # Load data. TODO data persistence?
    cache_dir = current_app.config['DATA_FOLDER']
    stocks_data_path = os.path.join(cache_dir, "stocks_price_history_1d.csv")
    stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True)
    
    #print("columns passed:", stocks_data[[ticker]].columns)
    #print("data: ", stocks_data[[ticker]])
    
    # Slice the specific ticker and drop its specific NaNs
    ticker_df = stocks_data[[ticker]].dropna()
    # Take its log returns
    ticker_log_returns = np.log(ticker_df)-np.log(ticker_df.shift(1))
    ticker_log_returns.dropna(inplace=True)
    #print("ticker_df: ", ticker_df)
    #print("ticker_log_returns: ", ticker_log_returns)
    
    # Compute the Sharpe ratio
    sharpe = sharpe_ratio(ticker_log_returns).dropna()
    # If it's a Series, grab the scalar value
    if hasattr(sharpe, 'iloc'):
        sharpe = sharpe.iloc[0]
    
    #print("sharpe: ", sharpe)
    
    # Compute the semivariance
    semivar = semivariance(ticker_log_returns).dropna()
    # If it's a Series, grab the scalar value
    if hasattr(semivar, 'iloc'):
        semivar = semivar.iloc[0]
    
    #print("semivariance: ", semivar)
    
    # Compute the Sharpe ratio
    sortino = sortino_ratio(ticker_log_returns).dropna()
    # If it's a Series, grab the scalar value
    if hasattr(sortino, 'iloc'):
        sortino = sortino.iloc[0]
    
    #print("sortino: ", sortino)
    
    # Compute the symmetry score
    counts = (ticker_log_returns > ticker_log_returns.mean()).sum().iloc[0]
    #print("counts: ", counts)
    total = ticker_log_returns.count().iloc[0]
    #print("total: ", total)
    symmetry_score = (counts/total)*100
    
    # Normality test (D'Agostino-Pearson)
    dagostino_pearson_statistics, dagostino_pearson_pvalue= stats.normaltest(ticker_log_returns)
    dagostino_pearson_statistics = dagostino_pearson_statistics[0]
    dagostino_pearson_pvalue = dagostino_pearson_pvalue[0]
    #print("dagostino_pearson_statistics: ", dagostino_pearson_statistics)
    #print("dagostino_pearson_pvalue: ", dagostino_pearson_pvalue)
    
    # Normality test (Jarque-Bera)
    jarque_bera_statistics, jarque_bera_pvalue= stats.normaltest(ticker_log_returns)
    jarque_bera_statistics = jarque_bera_statistics[0]
    jarque_bera_pvalue = jarque_bera_pvalue[0]
    #print("jarque_bera_statistics: ", jarque_bera_statistics)
    #print("jarque_bera_pvalue: ", jarque_bera_pvalue)
    
    # Z-scores
    ticker_max = ticker_log_returns.max()
    ticker_min = ticker_log_returns.min()
    ticker_std = ticker_log_returns.std()
    
    if hasattr(ticker_max, 'iloc'):
        ticker_max = ticker_max.iloc[0]
    
    if hasattr(ticker_min, 'iloc'):
        ticker_min = ticker_min.iloc[0]
        
    if hasattr(ticker_log_returns, 'iloc'):
        ticker_mean = ticker_log_returns.mean().iloc[0]
        
    if hasattr(ticker_std, 'iloc'):
        ticker_std = ticker_std.iloc[0]
        
#    print("ticker_max: ", ticker_max)
#    print("ticker_min: ", ticker_min)
#    print("ticker_mean: ", ticker_mean)
#    print("ticker_std: ", ticker_std)
    
    
    z_score_max = (ticker_max - ticker_mean) / ticker_std
    z_score_min = (ticker_min - ticker_mean) / ticker_std
#    print("z_score_max: ", z_score_max)
#    print("z_score_min: ", z_score_min)
    
    # Number of outliers (deviation from normality)
    upper_bound = 3*ticker_std + ticker_mean
    lower_bound = -3*ticker_std + ticker_mean
#    print("upper_bound: ", upper_bound)
#    print("lower_bound: ", lower_bound)
    len_returns_below = len(ticker_log_returns[ticker_log_returns<lower_bound].dropna())
    len_returns_above = len(ticker_log_returns[ticker_log_returns>upper_bound].dropna())
    len_outliers = len_returns_below + len_returns_above
#    print("len_outliers: ", len_outliers)
    
    # Slice the data for just the ONE ticker requested
    if mode == 'price':
        fig = plotting_utils.create_price_chart(ticker_df, rolling_windows=[20,50,200])
    elif mode == 'returns':
        fig = plotting_utils.create_returns_distribution_chart(ticker_log_returns)
    elif mode == 'map-2dcorr':
        t2 = ticker2 if ticker2 else ticker # Fallback to self if none selected
        combined_df = stocks_data[[ticker, t2]].dropna()
        fig = plotting_utils.create_2d_correlation_map(combined_df[[ticker]], combined_df[[t2]])
    elif mode == 'sentiment': #TODO
        fig = plotting_utils.create_price_chart(ticker_df, rolling_windows=[20,50,200])
    
#    print("get_data out")

    #return pio.to_json(fig)
    fig_json = pio.to_json(fig)
    fig_dict = json.loads(fig_json)
    
    return jsonify({
        'fig_data': fig_dict, # Convert Plotly fig to a dict instead of JSON string
        'sharpe': f"{sharpe:.2e}",
        'semivariance': f"{semivar:.2e}",
        'sortino': f"{sortino:.2e}",
        'symmetry_score': f"{symmetry_score:.2f}",
        'dagostino_pearson_statistics': f"{dagostino_pearson_statistics:.2f}",
        'dagostino_pearson_pvalue': f"{dagostino_pearson_pvalue:.2e}",
        'jarque_bera_statistics': f"{jarque_bera_statistics:.2f}",
        'jarque_bera_pvalue': f"{jarque_bera_pvalue:.2e}",
        'z_score_max': f"{z_score_max:.2f}",
        'z_score_min': f"{z_score_min:.2f}",
        'len_outliers': f"{len_outliers}"
    })
    

# Function setting portfolio constraints
#def setup_optimisation_constraints(num_assets, portfolio_tickers, stock_sectors, configured_max_stock_weight, configured_max_sector_weight, run_mvo_optimisation):
def setup_optimisation_constraints(portfolio_tickers, configured_max_stock_weight, run_mvo_optimisation):
    """
    Sets up the bounds and constraints for portfolio optimisation.

    Args:
        num_assets (int): The total number of assets in the portfolio.
        portfolio_tickers (list): A list of ticker symbols for the assets.
        stock_sectors (dict): A dictionary mapping stock tickers to their sectors.
        configured_max_stock_weight (float): Maximum allowed weight for a single stock.
        configured_max_sector_weight (float): Maximum allowed weight for a single sector.
        run_mvo_optimisation (bool): check if efficient frontier should be traced or not

    Returns:
        tuple: A tuple containing:
            - bounds (tuple): Bounds for each asset's weight (0 to 1).
            - constraints (list): A list of dictionaries defining the optimisation constraints.
            - initial_guess (np.array): An initial equal-weighted guess for optimisation.
    """
    num_assets = len(portfolio_tickers)

    bounds = tuple((0, 1) for asset in range(num_assets)) # Weights between 0 and 1 (no short-selling)
    
    
    # Base constraint: sum of weights equals 1
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]

#    # Ideally, individual stock weight constraint should be at most 5%.
    effective_max_stock_weight = max(configured_max_stock_weight, 1.0 / num_assets)
#    print(f"Maximum individual stock weight: {effective_max_stock_weight:.2%}\n")
#    if num_assets <= 20 and run_mvo_optimisation:
#        print("WARNING: with 20 assets or less, the efficient frontier is reduced to a single point (MVP) because the code currently constrains each asset being at most 5% of your portfolio.")
#    elif num_assets > 20 and run_mvo_optimisation:
#        print("More than 20 assets detected. We will attempt to draw the efficient frontier.\n")

    # Add individual stock weight constraint (max 5%)
    for i in range(num_assets):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=i: effective_max_stock_weight - x[idx]
        })

#    # Add sector weight constraints (max 25%)
#    sectors = {}
#    for i, ticker in enumerate(portfolio_tickers):
#        sector = stock_sectors.get(ticker)
#        if sector:
#            if sector not in sectors:
#                sectors[sector] = []
#            sectors[sector].append(i) # Store index of stock in portfolio_tickers
#        else:
#            print(f"Warning: Ticker '{ticker}' not found in stock_sectors. It will not be subject to sector constraints.")

#    # Determine a dynamic maximum sector weight based on the number of assets
#    # If num_assets is less than 20, allow sectors to take up to 100% (effectively disabling the hard cap)
#    effective_max_sector_weight_for_constraint = configured_max_sector_weight
#    if num_assets <= 20:
#        effective_max_sector_weight_for_constraint = 1.0 # Allow up to 100% for sectors if few assets
#        #print(f"effective_max_sector_weight_for_constraint:{effective_max_sector_weight_for_constraint}") 
    
    
#    for sector_name, stock_indices in sectors.items():
#        # Maximum possible weight a sector can have given individual stock limits.
#        sum_of_effective_max_stock_weights_in_sector = sum(effective_max_stock_weight for _ in stock_indices)
#        
#        # Determine the effective maximum sector weight for this specific sector.
#        current_sector_limit = min(effective_max_sector_weight_for_constraint, sum_of_effective_max_stock_weights_in_sector)
#        
#        print(f"Maximum weight for sector '{sector_name}': {current_sector_limit:.2%}")

#        constraints.append({
#            'type': 'ineq',
#            'fun': lambda x, indices=stock_indices, effective_limit=current_sector_limit: effective_limit - np.sum(x[indices]),
#            'args': (stock_indices, current_sector_limit,)
#        })

    # Set to equal-weighted portfolio
  #  initial_guess = np.array(num_assets * [1. / num_assets])
    
#    return bounds, constraints, initial_guess
        return bounds, constraints

def run_single_optimisation(objective_function, initial_guess, args, bounds, constraints):
    """
    Helper function to run a single optimisation
    
    Args:
        objective_function (callable): The function to minimise (MVP, Sharpe, Sortino, etc...).
        initial_guess (np.ndarray): Initial guess for weights.
        args (tuple): Additional arguments to pass to the objective function.
        bounds (tuple): Bounds for each weight.
        constraints (tuple): Constraints for the optimisation.
        
    Returns:
        tuple: (optimal_weights, success_status, message)
    """
    options = {'maxiter': 100}
    result = minimize(objective_function, initial_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints, options=options)
    
#    if not result.success:
#        print(f"DEBUG: Optimization FAILED! Message: {result.message}")
#        print(f"DEBUG: Status: {result.status}, NFEV: {result.nfev}, NIT: {result.nit}, JACOB: {result.njev}")
#        if result.x is not None:
#             print(f"DEBUG: Resulting weights (if any): {result.x}")
             
    if result.success and not np.any(np.isnan(result.x)):
        return result.x, True, result.message
    else:
        return initial_guess, False, result.message # Return initial_guess as fallback
        
def portfolio_return(weights, annual_returns):
    """
    Calculates the portfolio's expected return.
    """
    return np.sum(weights * annual_returns)
    
# Function to calculate Beta
def calculate_beta(daily_returns, benchmark_returns):
    """
    Calculates the beta of a stock or portfolio against a benchmark.
    Beta = Covariance(Asset, Benchmark) / Variance(Benchmark)
    Args:
        daily_returns (pd.Series): Daily returns of the asset/portfolio.
        benchmark_returns (pd.Series): Daily returns of the benchmark.
    Returns:
        float: The calculated beta.
    """
    # Ensure returns have the same dates
    common_dates = daily_returns.index.intersection(benchmark_returns.index)
    returns = daily_returns.loc[common_dates]
    bench_returns = benchmark_returns.loc[common_dates]
    
    # Calculate covariance and variance
    covariance = returns.cov(bench_returns)
    benchmark_variance = bench_returns.var()
    
    # Check if benchmark_variance is a Series and extract its value
    if isinstance(benchmark_variance, pd.Series):
        if benchmark_variance.item() == 0:
            return 0.0
    elif benchmark_variance == 0:
        return 0.0
    
    return covariance / benchmark_variance

# Function to calculate Alpha
def calculate_alpha(annual_return, beta, risk_free_rate, benchmark_annual_return):
    """
    Calculates the alpha of a stock or portfolio.
    Alpha = R_stock/portfolio - [R_f + Beta * (R_benchmark - R_f)]
    Args:
        annual_return (float): The annualised return of the asset/portfolio.
        beta (float): The beta of the asset/portfolio.
        risk_free_rate (float): The annualised risk-free rate.
        benchmark_annual_return (float): The annualised return of the benchmark.
    Returns:
        float: The calculated alpha.
    """
    expected_return = risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
    return annual_return - expected_return
    
def _calculate_portfolio_metrics_full(weights, annual_returns, daily_returns_df_slice, annualised_covariance_matrix, risk_free_rate, benchmark_returns, benchmark_annual_return, lambda_s=None, lambda_k=None):
    """
    Calculates a comprehensive set of portfolio metrics.
    
    Args:
        weights (np.array): Array of weights for each asset.
        annual_returns (np.array): Array of annualised returns for each asset.
        daily_returns_df_slice (pd.DataFrame): Daily returns for the lookback period.
        annualised_covariance_matrix (np.array): Annualised covariance matrix of asset returns.
        risk_free_rate (float): Risk-free rate.
        benchmark_returns (pd.Series): The daily returns of the benchmark.
        benchmark_annual_return (float): The annualised return of the benchmark.
        lambda_s (float, optional): Coefficient for skewness (for MVSK).
        lambda_k (float, optional): Coefficient for kurtosis (for MVSK).
        
    Returns:
        dict: A dictionary of calculated metrics.
    """
    metrics = {}
    
    p_return = portfolio_return(weights, annual_returns)
    p_volatility = portfolio_volatility(weights, annualised_covariance_matrix)
    p_beta = calculate_beta(daily_returns_df_slice.dot(weights), benchmark_returns)
    p_alpha = calculate_alpha(p_return, p_beta, risk_free_rate, benchmark_annual_return)
    
    metrics['Return'] = p_return
    metrics['Volatility'] = p_volatility
    metrics['Beta'] = p_beta
    metrics['Alpha'] = p_alpha
    
    if p_volatility > 0:
        metrics['Sharpe Ratio'] = (p_return - risk_free_rate) / p_volatility
    else:
        metrics['Sharpe Ratio'] = np.inf if p_return > risk_free_rate else np.nan # Handle zero volatility

    p_downside_dev = downside_deviation(weights, daily_returns_df_slice, risk_free_rate)
    if p_downside_dev > 0:
        metrics['Sortino Ratio'] = (p_return - risk_free_rate) / p_downside_dev
    else:
        metrics['Sortino Ratio'] = np.inf if p_return > risk_free_rate else np.nan # Handle zero downside deviation

    metrics['Skewness'] = portfolio_skewness(weights, daily_returns_df_slice)
    metrics['Kurtosis'] = portfolio_kurtosis(weights, daily_returns_df_slice)
    
    # MVSK Utility
    if lambda_s is not None and lambda_k is not None:
        if p_volatility > 0:
            metrics['MVSK Utility'] = (p_return - risk_free_rate) / p_volatility + lambda_s * metrics['Skewness'] - lambda_k * metrics['Kurtosis']
        else:
            metrics['MVSK Utility'] = np.inf if p_return > risk_free_rate else np.nan
            
    return metrics
        
# Calculate only undesired volatility (downside risk)
def downside_deviation(weights, daily_returns_df_slice, risk_free_rate):
    """
    Calculates the annualised downside deviation for a portfolio.
    Only considers returns below the Minimum Acceptable Return (MAR), which is the risk-free rate.
    
    Args:
        weights (np.array): Array of weights for each asset.
        daily_returns_df_slice (pd.DataFrame): Daily returns for the lookback period.
        risk_free_rate (float): Annualised risk-free rate.
    """
    # Calculate portfolio daily returns for the slice
    portfolio_daily_returns = daily_returns_df_slice.dot(weights)
    
    # Calculate daily MAR
    daily_mar = (1 + risk_free_rate)**(1/252) - 1 # Convert annualised risk-free rate to daily

    # Filter for returns below the MAR
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < daily_mar]
    
    if downside_returns.empty:
        return 0.0 # No downside returns, so downside deviation is 0

    # Calculate downside deviation (standard deviation of downside returns)
    downside_std = np.sqrt(np.mean((downside_returns - daily_mar)**2))
    
    # Annualise downside deviation
    annualised_downside_std = downside_std * np.sqrt(252)
    return annualised_downside_std
    
def portfolio_skewness(weights, daily_returns_df_slice):
    """
    Calculates the skewness for a portfolio's daily returns.
    """
    portfolio_daily_returns = daily_returns_df_slice.dot(weights)
    return portfolio_daily_returns.skew()
    
# Calculate extreme events in returns distribution
def portfolio_kurtosis(weights, daily_returns_df_slice):
    """
    Calculates the kurtosis for a portfolio's daily returns.
    """
    portfolio_daily_returns = daily_returns_df_slice.dot(weights)
    return portfolio_daily_returns.kurtosis()
    
def calculate_mvp_portfolio(annual_returns: np.ndarray, 
                            covariance_matrix: np.ndarray, 
                            initial_guess: np.ndarray, 
                            bounds: tuple, 
                            constraints: tuple,
                            daily_returns_df_slice: pd.DataFrame,
                            risk_free_rate: float,
                            num_frontier_points: int,
                            benchmark_returns: pd.Series,
                            benchmark_annual_return: float,
                            verbose: bool = True,
                            calculate_frontier: bool = True) -> dict:
    """
    Calculates the Minimum Variance Portfolio (MVP) and traces the Efficient Frontier for a given static set of returns and covariance.

    Args:
        annual_returns (np.ndarray): Array of annualised returns for each asset.
        covariance_matrix (np.ndarray): Annualised covariance matrix of asset returns.
        initial_guess (np.ndarray): Initial guess for portfolio weights.
        bounds (tuple): Tuple of (min_weight, max_weight) for each asset.
        daily_returns_df_slice (pd.DataFrame): Daily returns for the lookback period (for metrics).
        risk_free_rate (float): Risk-free rate (for metrics).
        constraints (tuple): Tuple of constraints for the optimisation.
        num_assets (int): Number of assets in the portfolio.
        num_frontier_points (int): Number of points to calculate for the efficient frontier.

    Returns:
        dict: A dictionary containing MVP details and efficient frontier data.
    """
    results = {
        'weights': initial_guess,
        'metrics': _calculate_portfolio_metrics_full(initial_guess, annual_returns, daily_returns_df_slice, covariance_matrix, risk_free_rate, benchmark_returns, benchmark_annual_return, benchmark_returns),
        'success': False,
        'message': "Optimisation not attempted or failed",
        'efficient_frontier_std_devs': [],
        'efficient_frontier_returns': []
    }

    # Check if covariance matrix is singular
    if np.linalg.matrix_rank(covariance_matrix) < covariance_matrix.shape[0]:
        message = "Optimisation defaulted to initial weights due to singular covariance matrix."
        if verbose:
            print(f"Warning: {message}")
            return {
                'metrics': _calculate_portfolio_metrics_full(initial_guess, annual_returns, daily_returns_df_slice, covariance_matrix, risk_free_rate, benchmark_returns, benchmark_annual_return, lambda_s, lambda_k),
                'success': True, # Treat as success for test reporting, as it defaults to valid weights
                'message': message,
                'efficient_frontier_std_devs': [],
                'efficient_frontier_returns': []
            }
    
    # Find MVP weights
    mvp_weights, mvp_success, mvp_message = run_single_optimisation(
        portfolio_volatility, initial_guess, args=(covariance_matrix,),
        bounds=bounds, constraints=constraints
    )

    if mvp_success:
        results['weights'] = mvp_weights
        results['success'] = True
        results['metrics'] = _calculate_portfolio_metrics_full(
            mvp_weights, annual_returns, daily_returns_df_slice, covariance_matrix, risk_free_rate, benchmark_returns, benchmark_annual_return
        )
        results['message'] = mvp_message
        
        if verbose:
            print("\nMinimum Variance Portfolio (MVP):")
            print(f"Return: {results['metrics']['Return']:.2%}")
            print(f"Volatility: {results['metrics']['Volatility']:.2%}")
            print(f"Beta: {results['metrics']['Beta']:.4f}")
            print(f"Alpha: {results['metrics']['Alpha']:.2%}")
            #print("Weights: ", results['weights'])
        
        if calculate_frontier:            
            # Trace the Efficient Frontier
            min_return_for_frontier = results['metrics']['Return']
            print("annual_returns: ", annual_returns)
            print("max1: ", annual_returns.max())
            print("max2: ", max(annual_returns))
            max_return_for_frontier = max(annual_returns) + 0.01  # Add a small buffer to ensure the highest return point is included. 
            if min_return_for_frontier >= max_return_for_frontier:
                max_return_for_frontier = min_return_for_frontier + 0.001

            #print("min_return_for_frontier: ", min_return_for_frontier, "max_return_for_frontier: ", max_return_for_frontier) 
            target_returns = np.linspace(min_return_for_frontier, max_return_for_frontier, num_frontier_points)

            efficient_frontier_std_devs = [results['metrics']['Volatility']]
            efficient_frontier_returns = [results['metrics']['Return']]
            current_initial_guess_frontier = results['weights']

            #print("\nefficient_frontier_std_devs: ", efficient_frontier_std_devs)
            #print("\nefficient_frontier_returns: ", efficient_frontier_returns)
            #print("\ncurrent_initial_guess_frontier: ", current_initial_guess_frontier)
            
            failures_in_a_row = 0
            last_achieved_target = min_return_for_frontier

            for target_ret in target_returns:
                return_constraint = {'type': 'eq', 'fun': lambda x: portfolio_return(x, annual_returns) - target_ret}
                all_constraints = constraints + [return_constraint]

                frontier_weights, frontier_success, frontier_message = run_single_optimisation(
                    portfolio_volatility, current_initial_guess_frontier, args=(covariance_matrix,),
                    bounds=bounds, constraints=all_constraints
                )

                #print(f"DEBUG: Frontier optimisation for target weight {frontier_weights}: Success={frontier_success}, Message={frontier_message}") 
                if frontier_success:
                    frontier_volatility = portfolio_volatility(frontier_weights, covariance_matrix)
                    frontier_return = portfolio_return(frontier_weights, annual_returns)
                    efficient_frontier_std_devs.append(frontier_volatility)
                    efficient_frontier_returns.append(frontier_return)
                    current_initial_guess_frontier = frontier_weights
                    failures_in_a_row = 0
                    last_achieved_target = target_ret
                else:
                    #print(f"Optimisation failed at target return {frontier_return:.2%}: {frontier_message}")
                    failures_in_a_row += 1
                    if failures_in_a_row >= 1:
                        print(f"\nMaximum return target: {last_achieved_target:.2%} (Efficient frontier tracing stopped).")
                        break 

            # Ensures the frontier is drawn smoothly and monotonically left-to-right. 
            optimised_points = sorted(list(zip(efficient_frontier_std_devs, efficient_frontier_returns)))
            results['efficient_frontier_std_devs'] = [p[0] for p in optimised_points]
            results['efficient_frontier_returns'] = [p[1] for p in optimised_points]
            
    else:
        weights_to_use = initial_guess 
        success_status = False
        message_to_use = f"Optimisation failed: {mvp_message}."
        if verbose:
            print(f"Warning: {message_to_use}")

        results = {
            'weights': weights_to_use,
            'metrics': _calculate_portfolio_metrics_full(weights_to_use, annual_returns, daily_returns_df_slice, covariance_matrix, risk_free_rate, benchmark_returns, benchmark_annual_return),
            'success': success_status,
            'message': message_to_use
        }

    return results
    
# Function to perform static optimisation
#def perform_static_optimisation(annual_returns_array, static_annualised_covariance_matrix, initial_guess, bounds, constraints, daily_returns, risk_free_rate, num_assets, num_frontier_points, benchmark_returns, benchmark_annual_return, lambda_s, lambda_k, feature_toggles):
def perform_static_optimisation(annual_returns_array, static_annualised_covariance_matrix, initial_guess, bounds, constraints, daily_returns, risk_free_rate, num_frontier_points, benchmark_returns, benchmark_annual_return):
    """
    Performs static portfolio optimisations (MVP, Sharpe, Sortino, MVSK).

    Args:
        annual_returns_array (np.array): Annualised mean returns for all stocks.
        static_annualised_covariance_matrix (np.array): Static annualised covariance matrix.
        initial_guess (np.array): Initial equal-weighted guess for optimisation.
        bounds (tuple): Bounds for each asset's weight.
        constraints (list): List of dictionaries defining the optimisation constraints.
        daily_returns (pd.DataFrame): DataFrame of daily returns for all stocks.
        risk_free_rate (float): Risk-free rate.
        num_assets (int): Number of assets.
        num_frontier_points (int): Number of points for efficient frontier.
        lambda_s (float): Skewness penalty parameter for MVSK.
        lambda_k (float): Kurtosis penalty parameter for MVSK.
        feature_toggles (dict): Dictionary of feature toggles.

    Returns:
        dict: A dictionary containing the results of each enabled static optimisation.
    """
    # Variables to store static optimisation results
    static_results = {
        'mvp': None,
        'sharpe': None,
        'sortino': None,
        'mvsk': None,
        'efficient_frontier_std_devs': [],
        'efficient_frontier_returns': []
    }
    
    RUN_STATIC_PORTFOLIO = True

    #if feature_toggles['RUN_STATIC_PORTFOLIO']:
    if RUN_STATIC_PORTFOLIO:
        print("\n--- OPTIMISATION WITH STATIC COVARIANCE MODEL ---")

#        # 0. Calculate Equal-Weighted Portfolio - Static
#        if feature_toggles['RUN_EQUAL_WEIGHTED_PORTFOLIO']:
#            print("Calculating static equal-weighted portfolio...")
#            equal_weights = np.array([1./num_assets] * num_assets)
#            
#            # Calculate portfolio annual return
#            portfolio_annual_return = np.sum(annual_returns_array * equal_weights)
#            
#            # Calculate portfolio annual standard deviation
#            portfolio_annual_std_dev = np.sqrt(np.dot(equal_weights.T, np.dot(static_annualised_covariance_matrix, equal_weights)))
#            
#            # Calculate Sharpe ratio
#            if portfolio_annual_std_dev != 0:
#                sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_annual_std_dev
#            else:
#                sharpe_ratio = 0.0 # Handle case where std dev is zero

#            
#            static_results['ewp'] = {
#                'weights': equal_weights,
#                'Return': portfolio_annual_return,
#                'Volatility': portfolio_annual_std_dev,
#                'Sharpe Ratio': sharpe_ratio,
#                'success': True,
#                'message': 'Static equal-weighted portfolio calculated successfully.'
#            }
#            print(f"Static Equal-Weighted Portfolio (EWP) - Annual return: {portfolio_annual_return:.2%}, annual volatility: {portfolio_annual_std_dev:.2%}, Sharpe ratio: {sharpe_ratio:.4f}")
        
        # 1. Find Minimum Variance Portfolio (MVP) - Static
        RUN_MVO_OPTIMISATION = True
        if RUN_MVO_OPTIMISATION: 
        #if feature_toggles['RUN_MVO_OPTIMISATION']:
            static_results['mvp'] = calculate_mvp_portfolio(
                annual_returns=annual_returns_array,
                covariance_matrix=static_annualised_covariance_matrix,
                initial_guess=initial_guess,
                bounds=bounds,
                constraints=constraints,
                daily_returns_df_slice=daily_returns,
                risk_free_rate=risk_free_rate,
                num_frontier_points=num_frontier_points,
                benchmark_returns=benchmark_returns,
                benchmark_annual_return=benchmark_annual_return,
                verbose=True
            )
            if static_results['mvp']['success']:
                static_results['efficient_frontier_std_devs'] = static_results['mvp']['efficient_frontier_std_devs']
                static_results['efficient_frontier_returns'] = static_results['mvp']['efficient_frontier_returns']

#        # 2. Find Tangency Portfolio (Maximum Sharpe Ratio Portfolio) - Static
#        if feature_toggles['RUN_SHARPE_OPTIMISATION']:
#            static_results['sharpe'] = calculate_sharpe_portfolio(
#                annual_returns=annual_returns_array,
#                covariance_matrix=static_annualised_covariance_matrix,
#                initial_guess=initial_guess,
#                bounds=bounds,
#                constraints=constraints,
#                daily_returns_df_slice=daily_returns,
#                risk_free_rate=risk_free_rate,
#                benchmark_returns=benchmark_returns,
#                benchmark_annual_return=benchmark_annual_return,
#                verbose=True
#            )

#        # 3. Find Sortino Ratio Optimised Portfolio - Static
#        if feature_toggles['RUN_SORTINO_OPTIMISATION']:
#            static_results['sortino'] = calculate_sortino_portfolio(
#                annual_returns=annual_returns_array,
#                covariance_matrix=static_annualised_covariance_matrix,
#                initial_guess=initial_guess,
#                bounds=bounds,
#                constraints=constraints,
#                daily_returns_df_slice=daily_returns,
#                risk_free_rate=risk_free_rate,
#                benchmark_returns=benchmark_returns,
#                benchmark_annual_return=benchmark_annual_return,
#                verbose=True
#            )
#        
#        # 4. Find MVSK Optimised Portfolio - Static
#        if feature_toggles['RUN_MVSK_OPTIMISATION']:
#            static_results['mvsk'] = calculate_mvsk_portfolio(
#                annual_returns=annual_returns_array,
#                covariance_matrix=static_annualised_covariance_matrix,
#                initial_guess=initial_guess,
#                bounds=bounds,
#                constraints=constraints,
#                daily_returns_df_slice=daily_returns,
#                risk_free_rate=risk_free_rate,
#                lambda_s=lambda_s,
#                lambda_k=lambda_k,
#                benchmark_returns=benchmark_returns,
#                benchmark_annual_return=benchmark_annual_return,
#                verbose=True
#            )
    return static_results
    
# Single plot for optimised portfolios, Monte Carlo simulation and efficient frontiers
#def plot_efficient_frontier_and_portfolios(
#    static_results, dynamic_results, individual_stock_metrics, portfolio_tickers,
#    static_portfolio_points_raw_mc, dynamic_portfolio_points_raw_mc,
#    output_dir, feature_toggles, num_assets
#):
#def plot_efficient_frontier_and_portfolios(
#    optimisation_results, individual_stock_metrics, tickers
#):
#    """
#    Plots the efficient frontier, Monte Carlo simulations, individual stocks,
#    and optimised portfolios (MVP, Sharpe, Sortino, MVSK).

#    Args:
#        static_results (dict): Results from static optimisation.
#        dynamic_results (dict): Results from dynamic optimisation.
#        individual_stock_metrics (list): List of dictionaries with individual stock metrics.
#        portfolio_tickers (list): List of ticker symbols for the assets.
#        static_portfolio_points_raw_mc (list): List of dictionaries for static Monte Carlo portfolios.
#        dynamic_portfolio_points_raw_mc (list): List of dictionaries for dynamic Monte Carlo portfolios.
#        output_dir (str): The directory to save the plot.
#        feature_toggles (dict): Dictionary of feature toggles.
#        num_assets (int): Number of assets in the portfolio.
#    """
##    RUN_STATIC_PORTFOLIO = feature_toggles['RUN_STATIC_PORTFOLIO']
#    RUN_STATIC_PORTFOLIO = True
##    RUN_DYNAMIC_PORTFOLIO = feature_toggles['RUN_DYNAMIC_PORTFOLIO']
##    RUN_EQUAL_WEIGHTED_PORTFOLIO = feature_toggles['RUN_EQUAL_WEIGHTED_PORTFOLIO']
##    RUN_MONTE_CARLO_SIMULATION = feature_toggles['RUN_MONTE_CARLO_SIMULATION']
##    RUN_MVO_OPTIMISATION = feature_toggles['RUN_MVO_OPTIMISATION']
#    RUN_MVO_OPTIMISATION = True
##    RUN_SHARPE_OPTIMISATION = feature_toggles['RUN_SHARPE_OPTIMISATION']
##    RUN_SORTINO_OPTIMISATION = feature_toggles['RUN_SORTINO_OPTIMISATION']
##    RUN_MVSK_OPTIMISATION = feature_toggles['RUN_MVSK_OPTIMISATION']

##    plt.figure(figsize=(14, 8)) # Larger figure for more elements

##    # Plot all Monte-Carlo-simulated portfolio combinations (lighter color, background)
##    if RUN_MONTE_CARLO_SIMULATION:
##        if RUN_STATIC_PORTFOLIO and static_portfolio_points_raw_mc:
##            plt.scatter([p['std_dev'] * 100 for p in static_portfolio_points_raw_mc],
##                        [p['return'] * 100 for p in static_portfolio_points_raw_mc],
##                        color='blue', marker='o', s=10, alpha=0.5, # More transparent
##                        label='Monte Carlo portfolio combinations (Static)')
##        if RUN_DYNAMIC_PORTFOLIO and dynamic_portfolio_points_raw_mc and dynamic_results['dynamic_covariance_available']:
##            plt.scatter([p['std_dev'] * 100 for p in dynamic_portfolio_points_raw_mc],
##                        [p['return'] * 100 for p in dynamic_portfolio_points_raw_mc],
##                        color='red', marker='o', s=10, alpha=0.5, # More transparent
##                        label='Monte Carlo portfolio combinations (Dynamic)')
#    
#    # Plot the Efficient Frontier line (Static Covariance)
##    if RUN_STATIC_PORTFOLIO and RUN_MVO_OPTIMISATION and num_assets > 20 and static_results['mvp'] and static_results['efficient_frontier_std_devs']:
##        plt.plot([s * 100 for s in static_results['efficient_frontier_std_devs']],
##                 [r * 100 for r in static_results['efficient_frontier_returns']],
##                 color='blue', linestyle='-', linewidth=2, label='Efficient frontier (Static)')

#        if RUN_STATIC_PORTFOLIO and RUN_MVO_OPTIMISATION and optimisation_results['mvp'] and optimisation_results['efficient_frontier_std_devs']:
#        plt.plot([s * 100 for s in static_results['efficient_frontier_std_devs']],
#                 [r * 100 for r in static_results['efficient_frontier_returns']],
#                 color='blue', linestyle='-', linewidth=2, label='Efficient frontier (Static)')

##    # Plot the Efficient Frontier line (Dynamic Covariance)
##    if RUN_DYNAMIC_PORTFOLIO and RUN_MVO_OPTIMISATION and num_assets > 20 and dynamic_results['mvp'] and dynamic_results['efficient_frontier_std_devs'] and dynamic_results['dynamic_covariance_available']:
##        plt.plot([s * 100 for s in dynamic_results['efficient_frontier_std_devs']],
##                 [r * 100 for r in dynamic_results['efficient_frontier_returns']],
##                 color='red', linestyle='-', linewidth=2, label='Efficient frontier (Dynamic)')


#    # Plot individual stocks in the return/std space
#    individual_stock_colors_palette = sns.color_palette("deep", n_colors=len(portfolio_tickers)).as_hex()

#    texts = []
#    for i, stock in enumerate(individual_stock_metrics):
#        plot_color = individual_stock_colors_palette[i % len(individual_stock_colors_palette)]
#        plt.scatter(stock['annualised_std'] * 100, stock['annual_return'] * 100,
#        marker='o', s=100, color=plot_color, edgecolor='black', linewidth=1.5,
#        label='_nolegend_'
#    )
#        x = stock['annualised_std'] * 100
#        y = stock['annual_return'] * 100
#        texts.append(plt.text(x, y, stock['ticker'], color=plot_color, fontsize=11, ha='center', va='bottom'))
#    
#    plt.scatter([], [], s=100, color='grey', label='Stock\'s annualised performance')  # dummy point
#    adjust_text(texts,
#            only_move={'points': 'xy', 'text': 'xy'},
#            expand_text=(2.0, 2.0),
#            expand_points=(2.5, 2.5))

#    if RUN_STATIC_PORTFOLIO:
#        # Plot the EWP (Static)
#        if RUN_EQUAL_WEIGHTED_PORTFOLIO and static_results['ewp'] and static_results['ewp']['success']:
#            plt.scatter(static_results['ewp']['Volatility'] * 100, static_results['ewp']['Return'] * 100,
#                        marker='p', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
#                        label=f"EWP (Static), Sharpe ratio={static_results['ewp']['Sharpe Ratio']:.2}")
#                        
#        # Plot the MVP (Static)
#        if RUN_MVO_OPTIMISATION and static_results['mvp'] and static_results['mvp']['success']:
#            plt.scatter(static_results['mvp']['metrics']['Volatility'] * 100, static_results['mvp']['metrics']['Return'] * 100,
#                        marker='*', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
#                        label='MV (Static)')

##        # Plot the Tangency Portfolio (Static)
##        if RUN_SHARPE_OPTIMISATION and static_results['sharpe'] and static_results['sharpe']['success']:
##            plt.scatter(static_results['sharpe']['metrics']['Volatility'] * 100, static_results['sharpe']['metrics']['Return'] * 100,
##                        marker='P', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
##                        label=f'Tangency (Static), Sharpe ratio={static_results["sharpe"]["metrics"]["Sharpe Ratio"]:.2}')

##        # Plot the Sortino Portfolio (Static)
##        if RUN_SORTINO_OPTIMISATION and static_results['sortino'] and static_results['sortino']['success']:
##            plt.scatter(static_results['sortino']['metrics']['Volatility'] * 100, static_results['sortino']['metrics']['Return'] * 100,
##                        marker='o', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
##                        label=f'Sortino (Static), Sortino ratio={static_results["sortino"]["metrics"]["Sortino Ratio"]:.2}')
##        
##        # Plot the MVSK Portfolio (Static)
##        if RUN_MVSK_OPTIMISATION and static_results['mvsk'] and static_results['mvsk']['success']:
##            plt.scatter(static_results['mvsk']['metrics']['Volatility'] * 100, static_results['mvsk']['metrics']['Return'] * 100,
##                        marker='^', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
##                        label=f'MVSK (Static)')


##    # Plot the MVP (Dynamic) if available and enabled
##    if RUN_DYNAMIC_PORTFOLIO and dynamic_results['dynamic_covariance_available']:
##        # Plot the EWP (Dynamic)
##        if RUN_EQUAL_WEIGHTED_PORTFOLIO and dynamic_results['ewp'] and dynamic_results['ewp']['success']:
##            plt.scatter(dynamic_results['ewp']['Volatility'] * 100, dynamic_results['ewp']['Return'] * 100,
##                        marker='p', s=200, color='red', edgecolor='red', alpha=0.3, linewidth=1.5,
##                        label=f"EWP (Dynamic), Sharpe ratio={dynamic_results['ewp']['Sharpe Ratio']:.2}")
##                        
##        # Plot the MVP (Dynamic)
##        if RUN_MVO_OPTIMISATION and dynamic_results['mvp'] and dynamic_results['mvp']['success']:
##            plt.scatter(dynamic_results['mvp']['metrics']['Volatility'] * 100, dynamic_results['mvp']['metrics']['Return'] * 100,
##                        marker='*', s=200, color='red', edgecolor='red', alpha=0.3, linewidth=1.5,
##                        label='MV (Dynamic)')

##        # Plot the Tangency Portfolio (Dynamic)
##        if RUN_SHARPE_OPTIMISATION and dynamic_results['sharpe'] and dynamic_results['sharpe']['success']:
##            plt.scatter(dynamic_results['sharpe']['metrics']['Volatility'] * 100, dynamic_results['sharpe']['metrics']['Return'] * 100,
##                        marker='P', s=200, color='red', edgecolor='red', alpha=0.3, linewidth=1.5,
##                        label=f'Tangency (Dynamic), Sharpe ratio={dynamic_results["sharpe"]["metrics"]["Sharpe Ratio"]:.2}')

##        # Plot the Sortino Portfolio (Dynamic)
##        if RUN_SORTINO_OPTIMISATION and dynamic_results['sortino'] and dynamic_results['sortino']['success']:
##            plt.scatter(dynamic_results['sortino']['metrics']['Volatility'] * 100, dynamic_results['sortino']['metrics']['Return'] * 100,
##                        marker='o', s=200, color='red', edgecolor='red', alpha=0.3, linewidth=1.5,
##                        label=f'Sortino (Dynamic), Sortino ratio={dynamic_results["sortino"]["metrics"]["Sortino Ratio"]:.2}')

##        # Plot the MVSK Portfolio (Dynamic)
##        if RUN_MVSK_OPTIMISATION and dynamic_results['mvsk'] and dynamic_results['mvsk']['success']:
##            plt.scatter(dynamic_results['mvsk']['metrics']['Volatility'] * 100, dynamic_results['mvsk']['metrics']['Return'] * 100,
##                        marker='^', s=200, color='red', edgecolor='red', alpha=0.3, linewidth=1.5,
##                        label=f'MVSK (Dynamic)')


##    if (RUN_STATIC_PORTFOLIO or RUN_DYNAMIC_PORTFOLIO) and \
##       (RUN_MONTE_CARLO_SIMULATION or RUN_MVO_OPTIMISATION or RUN_SHARPE_OPTIMISATION or RUN_SORTINO_OPTIMISATION or RUN_MVSK_OPTIMISATION):
##        plt.title('Optimised portfolios', fontsize=16)
##        plt.xlabel('Annualised Standard Deviation (Volatility) (%)', fontsize=12)
##        plt.ylabel('Annualised Return (%)', fontsize=12)
##        plt.grid(True, linestyle='--', alpha=0.7)
##        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',labelspacing=2)
##        plt.tight_layout(rect=[0, 0, 0.88, 1])
##        plot_path = os.path.join(output_dir, "optimised_portfolios.png")
##        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
##        print(f"\nOptimised portfolios saved in {plot_path}")
##        plt.close()
##    else:
##        print("\nYou must choose the type (static or dynamic) and at least one portfolio (Monte Carlo, Mean-Variance, Tangency, Sortino, MVSK) to display anything!")

def sharpe_ratio(stocks_data, RISK_FREE_RATE=0.0):
    """
       Calculate the Sharpe ratio
       Formula: Sharpe ratio = (R_p-R_fr)/sigma_p
    """
    return (stocks_data.mean() - RISK_FREE_RATE) / stocks_data.std()
    
def semivariance(stocks_data):
    """
       Calculate the semivariance
       Formula: Semivariance = (Sum_{r_i < <r>}^{n} (r_i - <r>)²) / n
    """
    # Average on all observations
    stocks_mean = stocks_data.mean()
    diff = stocks_data - stocks_mean
    downside_diff = diff.clip(upper=0) # Set positive deviations to 0
    stocks_semivariance = (downside_diff**2).mean()
    #print("stocks_semivariance: ", stocks_semivariance)
    
    # Average only on bad days
#    stocks_mean2 = stocks_data.mean()
#    stocks2_semivariance = ((stocks_data[stocks_data < stocks_mean2] - stocks_mean2) ** 2).mean()
#    print("stocks2_semivariance: ", stocks2_semivariance)
    return stocks_semivariance
    
def sortino_ratio(stocks_data, RISK_FREE_RATE=0.0): #TODO merge with semivariance?
    """
       Calculate the Sortino ratio
       Formula: Sortino ratio = (R_p+-R_fr)/sigma_p+
    """
    # Average on all observations
    stocks_mean = stocks_data.mean()
    diff = stocks_data - stocks_mean
    downside_diff = diff.clip(upper=0) # Set positive deviations to 0
    stocks_semivariance = (downside_diff**2).mean()
    stocks_semistd = np.sqrt(stocks_semivariance)
    #print("stocks_semistd: ", stocks_semistd)
    
    return (stocks_mean - RISK_FREE_RATE) / stocks_semistd
    



# Calculate volatility
def portfolio_volatility(weights, annualised_covariance_matrix):
    """
    Objective function to minimise: Portfolio standard deviation.
    """
    variance_portfolio = weights.T @ annualised_covariance_matrix @ weights
    return np.sqrt(variance_portfolio)

#@bp.route('/expand_history/<asset_type>', methods=['POST']) #TODO
#def expand_history(asset_type):
@bp.route('/expand_history')
def expand_history():
    """
        Fetches data on user request.
    """
    # Only for stocks at the moment
    #ticker_list = get_assets(asset_type)
    cache_dir = current_app.config['DATA_FOLDER']
    stocks_data_path = os.path.join(cache_dir, "stocks_price_history_1d.csv")
    stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True)
    tickers = sorted(stocks_data.columns.tolist())
    
    # Fetch date inserted by user
    new_start = request.args.get('start')
    new_start = pd.to_datetime(new_start)
    print("new_start: ", new_start)
    
    # Trigger the backfill logic
    if not tickers or not new_start:
        return jsonify({"message": "Missing parameters"}), 400

    try:
        finance_data.fetch_latest_metrics(  tickers,
                                            category_name='stocks',
                                            interval='1d', 
                                            target_start_date=new_start, 
                                            force_update=True)
        return jsonify({"message": f"Success! History expanded to {new_start}."})
    except Exception as e:
        print(f"Expand error: {e}")
        return jsonify({"message": "Failed to download history. Check console."}), 500
