import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
from pandas.tseries.offsets import Day, BDay
import numpy as np
from scipy.stats import linregress
import json
import matplotlib.pyplot as plt
from flask import current_app

# TODO automate this to fetch any currency pair
def fetch_exchange_rates(cache_dir):
    """Fetches EUR/USD and EUR/CHF rates using yfinance."""
    #print("fetch_exchange_rates called")
    
    os.makedirs(cache_dir, exist_ok=True)
    date_cache_path = os.path.join(cache_dir, f"last_fetch_date.json")
    
    # Default fallback values
    exchange_data = {
        "last_call_fx": "1900-01-01",
        "usd_eur_rate": 1.0,
        "chf_eur_rate": 1.0
    }
    
    # Save new values
    if os.path.exists(date_cache_path) and os.path.getsize(date_cache_path) > 0:
        try:
            with open(date_cache_path, 'r') as f:
                exchange_data.update(json.load(f))
            print("\nLoaded last exchange rate: ", exchange_data["last_call_fx"])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Date cache file corrupted or empty, ignoring: {e}")
    else:
        print("\nDate cache file is 0 bytes, ignoring.")
        
    # Get last time function was called
    last_call_fx = datetime.strptime(exchange_data["last_call_fx"], "%Y-%m-%d").date()
    #print("last_call_fx: ", last_call_fx)
    
    # Lazy loading logic (once per day)
    if date.today()>last_call_fx:
        print("Fetching exhange rates.")        

        try:
            # Fetch EUR/USD rate
            eur_usd_ticker = yf.Ticker("EURUSD=X")
            eur_usd_hist = eur_usd_ticker.history(period="1d")
            if not eur_usd_hist.empty and eur_usd_hist["Close"].dropna().iloc[-1] > 0:
                exchange_data["usd_eur_rate"] = 1 / eur_usd_hist["Close"].dropna().iloc[-1] # Reciprocal (want EUR)
            
            # Fetch EUR/CHF rate
            eur_chf_ticker = yf.Ticker("EURCHF=X")
            eur_chf_hist = eur_chf_ticker.history(period="1d")
            if not eur_chf_hist.empty and eur_chf_hist["Close"].dropna().iloc[-1] > 0:
                exchange_data["chf_eur_rate"] = 1 / eur_chf_hist["Close"].dropna().iloc[-1] # Reciprocal (want EUR)
                
            exchange_data["last_call_fx"] = date.today().isoformat()
            with open(date_cache_path, 'w') as f:
                json.dump(exchange_data, f, indent=4)
            
            #print(f"Rates fetched: USD/EUR = {usd_eur_rate:.4f}, CHF/EUR = {chf_eur_rate:.4f}")

        except Exception as e:
            print(f"Error fetching exchange rates: {e}. Using last known/default rates.")
            
    else:
        print("No need to update exchange rates for today.")
        
    #print("fetch_exchange_rates out")
    return exchange_data["usd_eur_rate"], exchange_data["chf_eur_rate"]


def calculate_growth_rate(divs_filtered):
    """Calculates dividend growth rate (CAGR) using log-linear regression."""
    #print("calculate_growth_rate called")
    
    yearly_divs = divs_filtered.groupby(divs_filtered.index.year).sum()
    yearly_divs = yearly_divs[yearly_divs > 0] # Filter for log calculation
            
    growth_rate = "N/A"
    if len(yearly_divs) >= 2:
        #print("yearly_divs.index: ", yearly_divs.index)
        # x-values (years from the start)
        x = yearly_divs.index - yearly_divs.index[0]
        #print("x: ", x)
        
        # y-values (natural log of dividends)
        #print("yearly_divs.values: ", yearly_divs.values)
        y = np.log(yearly_divs.values)
        #print("y: ", y)
                
        # Log-linear regression: slope is compounded growth rate
        #slope, intercept, rvalue, _, _ = linregress(x, y) # Uncomment to visualise regression
        slope, _, _, _, _ = linregress(x, y)
        #print("slope: ", slope)
        #print(f"R-squared: {rvalue**2:.6f}")
        
        # Uncomment to visualise regression
#        plt.plot(x, y, 'o', label='original data')
#        plt.plot(x, intercept + slope*x, 'r', label='fitted line')
#        plt.legend()
#        plt.show()
        
        growth_rate = np.exp(slope) - 1
        growth_rate = round(growth_rate, 4)
        #print("growth_rate: ", growth_rate)
            
    #print("calculate_growth_rate out")
    
    return growth_rate


def check_price_cache_status(hist_prices, tickers_list, required_days): # TODO Maybe not needed any more?
    """Determines if a download is needed based on date and ticker coverage."""
    #print("check_price_cache_status called")
    
    today = datetime.today()
    target_start_date = today - timedelta(days=required_days)
    
    # Check for new tickers or empty cache
    if hist_prices.empty or not all(t in hist_prices.columns for t in tickers_list):
        print("\nNew tickers detected or cache empty. Fetching full history.")
        #print("today: ", today)
        #print("target_start_date: ", target_start_date)
        #print("today - target_start_date: ", today - target_start_date)
        return True, target_start_date # Full update
    
    # Check if we need to go further back in time. #TODO unnecessary now?
    earliest_date = hist_prices.index.min()
    if earliest_date > target_start_date:
        print(f"History too short (Earliest: {earliest_date.date()}). Fetching from {target_start_date.date()}.")
        return True, target_start_date
    
    # Check if the cache is outdated
    last_date = hist_prices.index.max()
    #print("last_date: ", last_date)
    
    if last_date < (today - timedelta(days=1)):
        print(f"Cache is behind. Fetching data.")
        #print("last_date + timedelta(days=1): ", last_date + timedelta(days=1))
        return True, last_date + timedelta(days=1) # Increment update
    
    #print("check_price_cache_status out")
    return False, None
    
def calculate_ticker_metrics(ticker, ticker_handle, hist_prices, usd_eur_rate, chf_eur_rate):
    """Get metrics for a single ticker."""
    #print("calculate_ticker_metrics called")

    one_year_ago = datetime.today() - timedelta(days=365)
    n_years = 10 # Number of years used in CAGR regression
    n_years_ago = datetime.today() - timedelta(days=n_years * 365)
    
    # Initialisation
    data = {"Ticker": ticker, "Currency": "N/A", "Quote":0.0, "Quote_EUR": 0.0, "P/E": 0.0, "Fwd_P/E": 0.0, 
            "P/B": 0.0, "PEG": 0.0, "Earnings_Growth": 0.0, "Div_Yield": 0.0, "Div_CAGR": 0.0, 
            "Latest_Div_EUR": 0.0, "Months_Paid": [0]*12, "Sector": "N/A", "PayoutRatio": 0.0}
    
    try:
        print(f"\nProcessing data for {ticker}")
        # Get info
        info = ticker_handle.info
        #print("info:", info)
        if not info: raise ValueError("No info returned")
        
        # Get currency
        currency = info.get('currency', 'EUR')
        rate = usd_eur_rate if currency == 'USD' else chf_eur_rate if currency == 'CHF' else 1.0
        
        # Update dictionary with the latest fetch
        data.update({
            "Currency": currency,
            "Sector": info.get('sector'),
            "PayoutRatio": info.get('payoutRatio'),
            "P/E": round(info['trailingPE'], 2) if isinstance(info.get('trailingPE'), (int, float)) else 0.0,
            "Fwd_P/E": round(info['forwardPE'], 2) if isinstance(info.get('forwardPE'), (int, float)) else 0.0,
            "P/B": round(info['priceToBook'], 2) if isinstance(info.get('priceToBook'), (int, float)) else 0.0
        })
                
        #print("Am I here?")
        
        # Price calculation from cached hist_prices
        if ticker in hist_prices.columns:
            #print("Am I here now?")
            #print("hist_prices[ticker]: ", hist_prices[ticker])
            #print("len(hist_prices): ", len(hist_prices))
            data["Quote"] = hist_prices[ticker].dropna().iloc[-1] # Get original quote
            data["Quote_EUR"] = round(float(data["Quote"] * rate),2) # Get quote in euro (because I'm French :) )
            #print(f"Last close for {ticker}: {data["Quote"]}")
            #print(f"rate for {ticker}: {rate}")
            
        # Fallback P/E calculation
        #print("pe: ", data["P/E"])
        
        if data["P/E"] == 0.0:
            eps_ttm = info.get('trailingEps') # If trailingPE is 0.0 (missing), try to calculate it manually using trailingEps
            #print("eps: ", eps_ttm)
            # Use currentPrice from info or fallback to cached quote
            price_native = info.get('currentPrice') or (data["Quote"] if data["Quote"] != 0 else None)
            #print("price_native: ", price_native)
            if eps_ttm and price_native:
                data["P/E"] = round(price_native / eps_ttm, 2)
                print(f"Calculated fallback P/E for {ticker}: {data['P/E']}")
        
        # Get income statement for earnings growth 
        try:
            income = ticker_handle.income_stmt
            if not income.empty:
                
                net_income = income.loc["Net Income"].T.to_frame(name="Earnings").sort_index()
                #print("Here")
                # TODO Include operating income later
                #op_income = income.loc["Operating Income"].T.to_frame(name="Operating Income").sort_index()
                
                #print("data 1: ", net_income.index[-1].strftime('%Y-%m-%d'))
                last_net_income = net_income["Earnings"].iloc[-1]
                #print("Last net_income: ", last_net_income)
                #print("op_income: ", op_income["Operating Income"].iloc[-1])
                
                #print("data 2: ", net_income.index[-2].strftime('%Y-%m-%d'))
                penultimate_net_income = net_income["Earnings"].iloc[-2]
                #print("Penultimate net_income: ", penultimate_net_income)
                #print("op_income: ", op_income["Operating Income"].iloc[-2])
                 
            
                # Annual earnings growth (YoY)
                if not np.isnan(last_net_income):
                    net_growth = last_net_income / abs(penultimate_net_income) - 1
                else:
                    # Fallback necessary because Yahoo Finance may take time to update financial data.
                    print(f"Last net income for {ticker} absent. Using previous two incomes.")
                    penpenultimate_net_income = net_income["Earnings"].iloc[-3]
                    #print("Penpenultimate net_income: ", penpenultimate_net_income)
                    net_growth = penultimate_net_income / abs(penpenultimate_net_income) - 1
                    
                #print("net_growth: ", net_growth)
                data["Earnings_Growth"] = round(net_growth, 4) if isinstance(net_growth, (int, float)) else "N/A"
                
                #op_growth = op_income["Operating Income"].iloc[-1] / abs(op_income["Operating Income"].iloc[-2]) - 1
                #data["Op_Inc_Growth"] = round(op_growth, 4) if isinstance(op_growth, (int, float)) else "N/A"
                
                #print("Quality check: ", abs(net_growth - op_growth))
                
                if isinstance(data["P/E"], float):
                    data["PEG"] = round(data["P/E"] / (net_growth * 100), 2)
        except Exception:
            print("Couldn't get Income Statement!")
        
        # Get dividend data
        actions = ticker_handle.actions
        #print("actions: ", actions)
        
        if not actions.empty and 'Dividends' in actions.columns and actions['Dividends'].sum() > 0:
            #print("Action in here")
            
            divs = actions[actions["Dividends"] > 0]["Dividends"].tz_localize(None) # Forget timezone
            if not divs.empty:
                #print("divs in the place")
                # TTM Yield
                #print("divs.index: ", divs.index)
                #print("one_year_ago: ", one_year_ago)
                ttm_divs = divs[divs.index >= one_year_ago]
                
                # I focus on euro from here
                if not ttm_divs.empty and data["Quote_EUR"] != 0.0:
                    #print("ttm not empty")
                    ttm_sum_eur = ttm_divs.sum() * (usd_eur_rate if currency == 'USD' else chf_eur_rate if currency == 'CHF' else 1)
                    data["Div_Yield"] = round(ttm_sum_eur / data["Quote_EUR"], 4)

                # Months paid
                for m_date in ttm_divs.index:
                    data["Months_Paid"][m_date.month - 1] = 1
                    
                #print("being paid")

                # Latest dividend amount
                latest_div = divs.iloc[-1]
                if isinstance(latest_div, (int, float)):
                    if currency == 'USD':
                        latest_div = latest_div * usd_eur_rate
                    elif currency == 'CHF':
                        latest_div = latest_div * chf_eur_rate
                    data["Latest_Div_EUR"] = round(latest_div, 4)
                    
                #print("latest div amount")

                # Growth rate (CAGR via log-linear regression)
                #print("divs.index: ", divs.index )
                divs_filtered = divs[divs.index >= n_years_ago]
                data["Div_CAGR"] = calculate_growth_rate(divs_filtered)
                #print("data[Div_CAGR]: ", data["Div_CAGR"])
                
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
    
    #print("calculate_ticker_metrics out")
    return data

# To compare the P/B ratio of a stock to the five biggest companies by market cap, I use the benchmark below
# TODO automate the choice of companies, and add the remaining sectors
SECTOR_BENCHMARK_MAP = {
    'Technology': ['AAPL', 'MSFT','NVDA','TSM','AVGO'],
    'Financial Services': ['JPM', 'BRK-A','MA','BAC','V'],
    'Industrials': ['GE', 'CAT', 'RTX', 'SIE.DE', 'AIR.PA'],
    'Utilities': ['NEE', 'IBDSF', 'DOGEF', 'ENLAY', 'CEG'],
    'Healthcare': ['LLY', 'JNJ', 'AZN', 'UNH', 'ROG.SW'],
    'Real Estate': ['WELL', 'PLD', 'AMT', 'EQIX', 'SPG'],
    'Communication Services': ['GOOGL', 'META', 'TCEHY', 'NFLX', 'SFTBY'],
    'Consumer Cyclical': ['AMZN', 'TSLA', 'LVMUY', 'BABAF', 'HD'],
}

def get_sector_pb_benchmark(sector, cache_dir, required_days: int =7):
    """
    Calculates avg P/B for a sector. 
    Uses existing static_metrics if available to avoid API calls.
    """
    #print("get_sector_pb_benchmark called")
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"last_fetch_date.json")
    
    # Initialisation
    bench_data = {
        "pb_bench_dates": {},
        "benchmarks": {}
    }
    
    # Try loading data
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        try:
            with open(cache_path, 'r') as f:
                bench_data.update(json.load(f))
            #print(f"Loaded last P/B benchmark")
        except (json.JSONDecodeError, IOError) as e:
            print(f"\nP/B cache file corrupted or empty, ignoring: {e}")
    else:
        print("\nP/B cache file is 0 bytes, ignoring.")
        
    # Check last update per sector
    last_call_pb_bench_str = bench_data.get("pb_bench_dates", {}).get(sector, "1900-01-01")
    last_call_pb_bench = datetime.strptime(last_call_pb_bench_str, "%Y-%m-%d").date()
    #print("last_call_pb_bench_str: ", last_call_pb_bench_str)
    #print("last_call_pb_bench: ", last_call_pb_bench)
    
    is_stale    = date.today() - last_call_pb_bench > timedelta(required_days)
    is_missing  = sector not in bench_data["benchmarks"]
    
    #print("date.today() - last_call_pb_bench: ", date.today() - last_call_pb_bench)
    #print("timedelta(required_days): ",  timedelta(required_days))
    #print("is_missing: ", is_missing)
    
    if is_stale or is_missing:
        if is_stale:
            print(f"\n{required_days} days passed. Updating P/B benchmark for {sector}...")
            
        if is_missing:
            print(f"\nSector {sector} is missing. Updating its P/B benchmark")
        
        proxies = SECTOR_BENCHMARK_MAP.get(sector, [])
        print("proxies: ", proxies)
        if not proxies:
            return 0.0

        # Fetch only the proxies we need, and only the 'info' part
        proxy_objs = yf.Tickers(" ".join(proxies))
        #print("proxy_objs: ", proxy_objs)
        pb_values = []
        
        for p in proxies:
            try:
                p_info = proxy_objs.tickers[p].info
                pb = p_info.get("priceToBook")
                #print("pb: ", pb)
                if isinstance(pb, (int, float)):
                    pb_values.append(pb)
            except: continue
            
        result = round(sum(pb_values) / len(pb_values), 2) if pb_values else 0.0
        print("result: ", result)
        
        # Update the dict if new value
        if "pb_bench_dates" not in bench_data:
            data["pb_bench_dates"] = {}
        if "benchmarks" not in bench_data:
            bench_data["benchmarks"] = {}
            
        if bench_data["benchmarks"].get(sector) != result or is_stale:
            bench_data["benchmarks"][sector] = result
            bench_data["pb_bench_dates"][sector] = date.today().isoformat()
        
            with open(cache_path, 'w') as f:
                json.dump(bench_data, f, indent=4)
                 
        #print("bench_data[pb_bench_dates][sector]: ", bench_data["pb_bench_dates"][sector])
        
        return result
            
    #print("get_sector_pb_benchmark out")
    return bench_data["benchmarks"].get(sector, 0.0)
    
def fetch_latest_metrics(tickers_list, category_name='assets', test=False, required_days=1, interval="1d", force_update=False, target_start_date=None):
    """
    Fetches the latest data and valuation metrics for a list of tickers using yfinance.
    """
    print("fetch_latest_metrics called")
    if not tickers_list:
        return []
        
    # Handle price caching
    if test==False:
        cache_dir = current_app.config['DATA_FOLDER']
    else:
        cache_dir = current_app.config['TEST_FOLDER']
    print("cache_dir: ", cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    price_cache_path    = os.path.join(cache_dir, f"{category_name}_price_history_{interval}.csv") # Interval is 4h for intraday monitoring portfolio, 1d for backtesting
    info_cache_path     = os.path.join(cache_dir, f"{category_name}_metrics_static.json")
    
    if target_start_date:
        force_update=True
        
    # Load metrics
    static_metrics = {}
    if os.path.exists(info_cache_path) and os.path.getsize(info_cache_path) > 0:
        try:
            with open(info_cache_path, 'r') as f:
                static_metrics.update(json.load(f))
            print(f"\nLoaded static metrics from cache: {info_cache_path}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"\nCache file corrupted or empty, ignoring: {e}")
            static_metrics = {} # Reset to empty if file is bad
    else:
        print("\nMetrics cache file is 0 bytes, ignoring.")
        
    # If we have a cache and there's no forced update, return now.
    if not force_update and static_metrics:
        # Ensure all requested tickers are in the cache before returning
        if all(t in static_metrics for t in tickers_list):
            print(f"Returning cached {category_name} data immediately.")
            return [static_metrics[t] for t in tickers_list]
    
    # Fetch exchange rates
    usd_eur_rate, chf_eur_rate = fetch_exchange_rates(cache_dir)
        
    # Load price data
    hist_prices = pd.DataFrame()
    first_datetime = None
    last_datetime = None
    
    if os.path.exists(price_cache_path):
        hist_prices = pd.read_csv(price_cache_path, index_col=0, parse_dates=True)
        hist_prices.index.name = 'Datetime'
        #print("hist_prices: ", hist_prices.tail(5))
        if not hist_prices.empty:
            # Determine the existing range of cached data
            first_datetime = hist_prices.index[0].tz_localize(None)
            last_datetime = hist_prices.index[-1]
            print("Last update time from price history: ", last_datetime)
    else:
        print("No historical prices. Fetching new prices.")
        last_datetime = datetime.now()-timedelta(days=3) # Set to 3 arbitrarily. Creates a file
        #print("last_datetime: ", last_datetime)
        
    # Determine the timezone of the existing data (if any)
    current_tz = None
    if isinstance(hist_prices.index, pd.DatetimeIndex):
        current_tz = hist_prices.index.tz
        print("current_tz: ", current_tz)
    
    # Check if historical prices are needed
    if target_start_date:
        target_ts = pd.to_datetime(target_start_date).tz_localize(None)
        print("target_ts: ", target_ts)
   
        # Ensure timezones match
        if current_tz:
            print("Yes current tz")
            target_ts = target_ts.tz_localize(current_tz)
            print("target_ts2: ", target_ts)
            if first_datetime:
                first_datetime = first_datetime.tz_localize(current_tz)
        else:
            print("No current tz")
            target_ts = target_ts.tz_localize('UTC')
            print("target_ts2: ", target_ts)
            if first_datetime:
                first_datetime = first_datetime.tz_localize('UTC')
            
        #print("target_ts: ", target_ts)
        #print("first_datetime: ", first_datetime)
        
        # If the target date is earlier than the cached date, fill the gap
        if first_datetime is None or target_ts < first_datetime:
            print(f"Backfilling history from {target_ts} to {first_datetime or 'now'}")
            
            # Download the older data
            # end_date is the start of the current data to avoid overlapping the whole set
            backfill_end = first_datetime if first_datetime else datetime.now()
            #print("backfill_end: ", backfill_end)
            
            historical_gap = yf.download(tickers_list, start=target_ts, end=backfill_end, 
                                         interval=interval, group_by='ticker', auto_adjust=True, progress=False)
            
            #print("historical_gap: ", historical_gap.head(5))
            if not historical_gap.empty:
                # Process Close prices
                if len(tickers_list) == 1:
                    new_old_prices = historical_gap[['Close']].rename(columns={'Close': tickers_list[0]})
                else:
                    new_old_prices = historical_gap.xs('Close', axis=1, level=1)
                
                #print("new_old_prices: ", new_old_prices.head(5))
                # Combine old history and existing history
                hist_prices = pd.concat([new_old_prices, hist_prices]).sort_index()
                hist_prices = hist_prices.ffill().groupby(level=0).last()
                hist_prices = hist_prices[~hist_prices.index.duplicated(keep='first')]
                # Remove overlaps and keep latest
                hist_prices = hist_prices.groupby(level=0).last()
                hist_prices = hist_prices.reindex(sorted(hist_prices.columns), axis=1)
                hist_prices = hist_prices.dropna()
                hist_prices.index.name = 'Datetime'
                #print("hist_prices: ", hist_prices.head(5))
                # Save to CSV
                hist_prices.to_csv(price_cache_path, index=True)
                print("Historical gap closed.")
        
    # Determine last local update time
    last_modified = 0
    if os.path.exists(price_cache_path):
        last_modified = os.path.getmtime(price_cache_path)
        last_update_time = datetime.fromtimestamp(last_modified)
        #print("Last local update time: ", last_update_time)
    else:
        last_update_time = datetime(1900, 1, 1)
        
    #last_update_time = datetime(1900, 1, 1) # Use that to debug
        
    # Determine if we need an update
    if interval=="1d":
        needs_price_download = datetime.now() - last_update_time > timedelta(days=1)
    else:
        needs_price_download = datetime.now() - last_update_time > timedelta(hours=4)
    #needs_price_download, download_start = check_price_cache_status(hist_prices, tickers_list, required_days) #TODO remove?
           
    # Check if a ticker is new and force download
    new_ticker = [t for t in tickers_list if t not in hist_prices.columns]
    if new_ticker:
        ticker_to_add = new_ticker[0]
        print(f"New ticker: {ticker_to_add}, start download.")
        download_start = hist_prices.index.min() # Use existing history as boundary
        new_ticker_data = yf.download(  ticker_to_add, 
                                        start= download_start,
                                        interval=interval, # TODO replace by correct interval
                                        auto_adjust=True)
                                        
        if not new_ticker_data.empty:
            new_price = new_ticker_data.xs('Close', axis=1, level=0)
            hist_prices = hist_prices.join(new_price, how='outer')
            hist_prices = hist_prices.reindex(sorted(hist_prices.columns), axis=1)
            hist_prices.index.name = 'Datetime'
            hist_prices.to_csv(price_cache_path, index=True)
            print(f"Added {ticker_to_add} to history.")

    # Don't download stocks prices if not a business day
    is_business_day = BDay().is_on_offset(datetime.now())
    print("is_business_day: ", is_business_day)
    
    if (not is_business_day and category_name=='stocks' and not new_ticker):
        needs_price_download = False
    
    print("needs_price_download: ", needs_price_download)
    #print("download_start: ", download_start)
    
    if needs_price_download: 
        print(f"Cache older than {interval}. Updating market data...")
        
        new_data = yf.download(tickers_list, start=last_datetime, end=datetime.now(), interval=interval, group_by='ticker', auto_adjust=True, progress=False)
        print("new_data: ", new_data.tail(5))
        
        
        if not new_data.empty:
            if len(tickers_list) == 1:
                new_prices = new_data[['Close']].rename(columns={'Close': tickers_list[0]})
            else:
                # Extract only 'Close' columns and simplify headers
                new_prices = new_data.xs('Close', axis=1, level=1)
                
            # Combine old and new, remove duplicates, and sort
            hist_prices = pd.concat([hist_prices, new_prices]).sort_index()
            #print(hist_prices.head(5))
            #print(hist_prices.tail(5))
            hist_prices = hist_prices.ffill().groupby(level=0).last()
            #print(hist_prices.head(5))
            #print(hist_prices.tail(5))
            #hist_prices = hist_prices[~hist_prices.index.duplicated(keep='last')]
            hist_prices = hist_prices.reindex(sorted(hist_prices.columns), axis=1)
            #print(hist_prices.head(5))
            #print(hist_prices.tail(5))
            hist_prices.index.name = 'Datetime'
            hist_prices.to_csv(price_cache_path, index=True)
            print(f"Price cache updated. hist_prices now has columns: {hist_prices.columns.tolist()}")
                                    
    # Process ticker metrics
    tickers_obj = yf.Tickers(" ".join(tickers_list))
    needs_info_update = False
    
    #print("I'm here")
    
    for ticker in tickers_list:
        # Check if we already processed this ticker today in the JSON cache
        #print("Got here")   
        if ticker not in static_metrics or needs_price_download: 
            needs_info_update = True # If not cached, fetch it
            static_metrics[ticker] = calculate_ticker_metrics(ticker, tickers_obj.tickers[ticker], hist_prices, usd_eur_rate, chf_eur_rate)
            #print(f"static_metrics[{ticker}]: ", static_metrics[ticker])
           
        # Sector benchmark
        sector = static_metrics[ticker].get("Sector")
        #print("ticker: ", ticker)
        #print("sector: ", sector)
        if sector:
            benchmark_pb = get_sector_pb_benchmark(sector, cache_dir)
            static_metrics[ticker]["Sector_PB_Benchmark"] = benchmark_pb
            #print(f"static_metrics[{ticker}][Sector_PB_Benchmark]: ", static_metrics[ticker]["Sector_PB_Benchmark"])
                
        
    # Save if we added new data
    if needs_info_update:
        try:
            with open(info_cache_path, 'w') as f:
                json.dump(static_metrics, f, indent=4)
            print(f"Static metrics updated in cache: {info_cache_path}")
        except Exception as e:
            print(f"Could not save cache: {e}")
            
    #print("fetch_latest_metrics out")
    return [static_metrics[t] for t in tickers_list]
        

def remove_from_price_history(ticker, interval, category_name='assets'):
    #print("remove_from_price_history called")
    
    cache_dir = current_app.config['DATA_FOLDER']
    os.makedirs(cache_dir, exist_ok=True)
    price_cache_path = os.path.join(cache_dir, f"{category_name}_price_history_{interval}.csv") #TODO replace 4h by interval
    try:
        df = pd.read_csv(price_cache_path, index_col=0)
        if ticker in df.columns:
            df.drop(columns=[ticker], inplace=True)
            df.to_csv(price_cache_path)
            print(f"\n{ticker} removed from price history")
    except FileNotFoundError:
        print(f"\nPrice history file not found.")
        
    #print("remove_from_price_history out")
        
def remove_from_metrics(ticker, category_name='assets'):
    #print("remove_from_metrics called")
    
    cache_dir = current_app.config['DATA_FOLDER']
    os.makedirs(cache_dir, exist_ok=True)
    info_cache_path = os.path.join(cache_dir, f"{category_name}_metrics_static.json")
    try:
        with open(info_cache_path, 'r') as f:
            metrics = json.load(f)
        
        if ticker in metrics:
            del metrics[ticker]
            with open(info_cache_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"\n{ticker} removed from metrics")
    except FileNotFoundError:
        print("\nMetrics file not found.")
        
    #print("remove_from_metrics out")
        
if __name__ == '__main__':
    # This block is for testing the data fetching function independently
#    TICKERS = [
#        "AMAT", "AMT", "AMUN.PA", "ASML.AS", "BMO", "BMW.DE", "BNP.PA",
#        "COFB.BR", "CS.PA", "DSY.PA", "ES", "ISP.MI", "LRCX", "MAN", "MBG.DE",
#        "MSI", "NN.AS", "NOVN.SW", "NVDA", "OMC", "PIA.MI", "PLD", "PUB.PA", "QBTS", "S4VC.F", "SAN.PA", "SLF", "TRN.MI", "TT"
#    ]
    TICKERS = [
            "SAN.PA", "TT"
        ]
    #benchmark_tickers = []
    #for ticker_list in SECTOR_BENCHMARK_MAP.values():
    #    benchmark_tickers.extend(ticker_list)
        
    #all_tickers = sorted(list(set(TICKERS + benchmark_tickers)))
    #print("all_tickers: ", all_tickers)
    
    metrics = fetch_latest_metrics(TICKERS, category_name='stocks', cache_dir='test', required_days=1, interval="4h", force_update=True)
    for i in range(len(TICKERS)):
        print(metrics[i])
