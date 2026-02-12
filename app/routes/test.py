from flask import Blueprint, render_template, request, jsonify, current_app
from app.utils import plotting_utils, finance_data
import os
import pandas as pd
import plotly.io as pio
import numpy as np
from datetime import datetime, date, timedelta

bp = Blueprint('test', __name__)

@bp.route('/test')
def test_feature():
    print("test_feature called")
    # TODO focus on stocks for now, add a general function later
    
    stocks_data = None
    
    cache_dir = current_app.config['DATA_FOLDER']
    
    print("cache_dir: ", cache_dir)

    stocks_data_path_fallback = os.path.join(cache_dir, 'stocks_price_history_4h.csv')
    stocks_data_path = os.path.join(cache_dir, 'stocks_price_history_1d.csv')
    crypto_data_path = os.path.join(cache_dir, 'crypto_price_history_1d.csv')
    
    print("stocks_data_path: ", stocks_data_path)

    # Check if we have local data
    if os.path.exists(stocks_data_path) and os.path.getsize(stocks_data_path) > 0:
        try:
            stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True).dropna()
            tickers = sorted(stocks_data.columns.tolist())
        except Exception as e:
            print(f"FAILED TO READ CSV: {e}")
    elif os.path.exists(stocks_data_path_fallback) and os.path.getsize(stocks_data_path_fallback) > 0: # Fallback if we don't
        print(f"No price history in {stocks_data_path}. Trying fallback")
        try:
            stocks_data_fallback = pd.read_csv(stocks_data_path_fallback, index_col='Datetime', parse_dates=True).dropna()
        except:
            print("No fallback historical data.")
            
        tickers = sorted(stocks_data_fallback.columns.tolist())
        print("tickers: ", tickers)
        finance_data.fetch_latest_metrics(  tickers,
                                            category_name='stocks',
                                            interval='1d', target_start_date=datetime.now()-timedelta(days=365)) # Use daily prices here!
        stocks_data = pd.read_csv(stocks_data_path, index_col='Datetime', parse_dates=True).dropna()
    else:
        print("No fallback. Abort")
        return
         
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
                            title='Portfolio Optimisation & Backtesting')

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

# Plots update
@bp.route('/get_plot')
def get_plot():
    print("get_plot called")
    
    ticker = request.args.get('ticker')
    ticker2 = request.args.get('ticker2')
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
    #print("ticker_df: ", ticker_df)
    
    # Slice the data for just the ONE ticker requested
    if mode == 'price':
        fig = plotting_utils.create_price_chart(stocks_data[[ticker]], rolling_windows=[20,50,200])
    elif mode == 'returns':
        fig = plotting_utils.create_returns_distribution_chart(ticker_df)
    elif mode == 'map-2dcorr':
        t2 = ticker2 if ticker2 else ticker # Fallback to self if none selected
        fig = plotting_utils.create_2d_correlation_map(stocks_data[[ticker]], stocks_data[[t2]])
    elif mode == 'sentiment': #TODO
        fig = plotting_utils.create_price_chart(stocks_data[[ticker]], rolling_windows=[20,50,200])
    
    print("get_plot out")

    return pio.to_json(fig)

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
