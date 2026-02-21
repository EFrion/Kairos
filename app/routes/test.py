from flask import Blueprint, render_template, request, jsonify, current_app
from app.utils import plotting_utils, finance_data
import os
import json
import pandas as pd
import plotly.io as pio
import numpy as np
from scipy import stats
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
    
    sharpe = sharpe_ratio(stocks_data, RISK_FREE_RATE=0.0)
    semivar = semivariance(stocks_data)
    sortino = sortino_ratio(stocks_data, RISK_FREE_RATE=0.0)
    
#    # Check data integrity
#    if ~stocks_value_data:
#        print("No NaN in stocks data")
#        
#        # Gives some information. To be improved!
#        investments, stocks_simple_returns, stocks_log_returns  = investCompare(
#            stocks_data,
#            date(2025, 1, 16),
#            date(2026, 1, 16),
#            {"MAN", "SLF"}, # Test sample
#        )
#        
#        print("investments: ", investments)
    
    
    return render_template( 'test.html',
                            tickers=tickers,
                            selected_ticker=selected_ticker,
                            sharpe=sharpe,
                            semivariance=semivar,
                            sortino=sortino,
                            title='Portfolio Optimisation & Backtesting')

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
    
#def investCompare(stocks_data, startTime, endTime, tickers, cache_dir):
#    startTime = pd.Timestamp(startTime).tz_localize('UTC')
#    endTime = pd.Timestamp(endTime).tz_localize('UTC')
#    #startTime = pd.Timestamp(startTime)
#    #endTime = pd.Timestamp(endTime)
#    
#    print("startTime: ", startTime)
#    print("endTime: ", endTime)


#    stocks_returns = stocks_data.pct_change().dropna()
#    print(stocks_returns.head(10))
#    
#        
#    stocks_value_returns = stocks_returns.isnull().values.any()
#    if ~stocks_value_returns:
#        print("No NaN in stocks_value_returns")
#    
##    returns = crypto_returns.join(stocks_returns, how='outer')
##    print("returns: ", returns)
#    
#    # Compute returns
#    log_return_rate = (np.log(stocks_data) - np.log(stocks_data.shift(1))).dropna()
#    print("log_return_rate (%): ", log_return_rate*100)

#    # pull data into separate DataFrame to just look at the last 365 days of
#    # data for calculating our high/low metric
#    currYear = stocks_data.loc[(endTime - timedelta(365)).tz_convert(stocks_data.index.tz) : endTime.tz_convert(stocks_data.index.tz)]

#    # High-Low
#    highLow = (currYear.max() - currYear.min()) / stocks_data.iloc[-1]
#    highLow = pd.DataFrame(highLow, columns=["HighMinusLow"])

#    # Moving average volatility
#    MA = pd.DataFrame(
#        ((abs(stocks_data - stocks_data.rolling(2).mean())) / stocks_data).mean(),
#        columns=["MovingAverageVolatility"],
#    )
#    investments = pd.merge(highLow, MA, left_index=True, right_index=True)

#    # Standard deviation
#    investments = pd.merge(
#        investments,
#        pd.DataFrame(stocks_returns.std(), columns=["StandardDeviation"]),
#        left_index=True,
#        right_index=True,
#    )

#    # Daily return
#    investments = pd.merge(
#        investments,
#        pd.DataFrame(stocks_returns.mean(), columns=["Daily Return Percentage"]),
#        left_index=True,
#        right_index=True,
#    )

#    # Format columns:
#    # Hogh-Low column rounded to 5 decimal
#    investments["HighMinusLow"] = investments["HighMinusLow"].round(4).astype(str)

#    # MA, std and returns columns as percentage
#    columns = ["MovingAverageVolatility", "StandardDeviation", "Daily Return Percentage"]
#    for col in columns:
#      investments[col] = (investments[col] * 100).round(3).astype(str) + '%'
#      
#    return investments, stocks_value_returns, log_return_rate

@bp.route('/get_portfolio_data')
def get_portfolio_data():
    print("get_portfolio_data called")
    
    mode = request.args.get('mode', 'returns')
    if not mode:
        return "No mode provided", 400
    
    # Load data. TODO data persistence?
    cache_dir = current_app.config['DATA_FOLDER']
    stocks_data_path = os.path.join(cache_dir, "stocks_price_history_1d.csv")
    stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True)
    
    len_data = len(stocks_data.columns)
    #print("len_data: ", len_data)
    
    weights = [1/len_data]*len_data # Equal weight for now
    #print("weights: ", weights)
    
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
    
    if mode == 'heatmap':
        fig = plotting_utils.plot_correlation_heatmap(correlation_matrix)
    elif mode == 'returns':
        fig = plotting_utils.create_returns_distribution_chart(log_returns_portfolio)
    
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
