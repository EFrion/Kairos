from flask import Blueprint, render_template, request, jsonify, current_app
from app.utils import plotting_utils, storage_utils, finance_data
from app.models import PortfolioManager
from app.analytics.optimiser import PortfolioOptimiser
from app.services.data_fetching import PortfolioDataManager
from app.analytics.analyser import AssetAnalyser, PortfolioAnalyser
import os
import pickle
import pandas as pd
from flask.views import MethodView

bp = Blueprint('test', __name__)

# TODO focus on stocks for now, add a general function later
class PortfolioView(MethodView):
    def __init__(self):
        # Initialise data manager once per view instance
        self.dm = PortfolioDataManager(current_app, finance_data, storage_utils)
        self.stocks_data = None

    def get(self):
        """
        Handle GET requests to /test: render the main portfolio page with initial data.
        """
        # Load stocks data once
        self.stocks_data = self.dm.get_data(asset_type='stocks')
        if self.stocks_data is None:
            return "Error: historical prices could not be loaded.", 404

        # Check for NaNs
        has_nan = self.stocks_data.isnull().values.any()

        # Get sorted tickers
        tickers = sorted(self.stocks_data.columns.tolist())

        # Determine selected ticker (default first)
        selected_ticker = request.args.get('ticker', tickers[0])

        if has_nan:
            current_app.logger.warning("NaN values detected in stocks data")

        # Create portfolio and analyser instances
        portfolio = PortfolioManager.from_storage(
            asset_classes=['stocks'],
            storage_utils=storage_utils,
            finance_data=finance_data,
            interval='1d',
            force_update=False
        )
        analyser = PortfolioAnalyser(portfolio.stocks, self.stocks_data)
        current_app.extensions["portfolio_analyser"] = analyser
        
        analysis_results = analyser.to_dict()

        return render_template(
            'test.html',
            tickers=analyser.tickers,
            analysis=analysis_results,
            selected_ticker=selected_ticker,
            title='Portfolio Optimisation & Backtesting'
        )

class PortfolioDataAPI(MethodView):
    def get(self):
        """
        Handle GET requests to /get_data: return JSON plot data for a ticker.
        """
        ticker = request.args.get('ticker')
        ticker2 = request.args.get('ticker2')
        mode = request.args.get('mode', 'price')

        analyser = current_app.extensions.get("portfolio_analyser")
        
        if analyser is None:
            return jsonify({"error": "Portfolio analyser not initialised"}), 500

        asset_analyser = analyser.asset_analysers.get(ticker)

        if asset_analyser is None:
            return jsonify({"error": "Ticker not found"}), 404
        if not ticker:
            return jsonify({'error': 'No ticker provided'}), 400
        if not mode:
            return jsonify({'error': 'No mode provided'}), 400

        ticker_df = asset_analyser.data.to_frame(name=ticker)

        # Prepare plot figure JSON depending on mode
        if mode == 'price':
            fig = plotting_utils.create_price_chart(ticker_df, rolling_windows=[20, 50, 200])
        elif mode == 'returns':
            returns_df = asset_analyser.percent_returns.to_frame(name=ticker)
            fig = plotting_utils.create_returns_distribution_chart(returns_df, asset_analyser.student_t_params)
        elif mode == 'map-2dcorr':
            asset_analyser2 = analyser.asset_analysers.get(ticker2)
            if asset_analyser2 is None:
                return jsonify({"error": "Second ticker not found"}), 404
            
            df1 = asset_analyser.data.to_frame(name=ticker)
            df2 = asset_analyser2.data.to_frame(name=ticker2)

            fig = plotting_utils.create_2d_correlation_map(df1, df2)
        elif mode == 'trend':
            trend_df = asset_analyser.trend.drop(columns=['isPartial'], errors='ignore')
            fig = plotting_utils.create_trends_chart(trend_df, rolling_windows=[7])
        elif mode == 'sentiment':
            # Placeholder for future sentiment plot
            fig = plotting_utils.create_price_chart(ticker_df, rolling_windows=[20, 50, 200])
        else:
            return jsonify({'error': 'Invalid mode'}), 400

        # Convert figure to JSON for frontend rendering
        fig_json = fig.to_json()

        return jsonify({'fig_data': fig_json})

# Register the views with URL rules
bp.add_url_rule('/test', view_func=PortfolioView.as_view('portfolio_test'))
bp.add_url_rule('/get_data', view_func=PortfolioDataAPI.as_view('portfolio_data_api'))


@bp.route('/get_portfolio_data')
def get_portfolio_data():
    print("get_portfolio_data called")
    
    mode = request.args.get('mode', 'returns')
    if not mode:
        return "No mode provided", 400
        
    force_update = request.args.get('force_update') == 'true' # Check for button click
 
    analyser = current_app.extensions.get("portfolio_analyser")
    if analyser is None:
            return jsonify({"error": "Portfolio analyser not initialised"}), 500
    
    cache_path = os.path.join(current_app.config['DATA_FOLDER'], "frontier_cache.pkl")

    if mode == 'heatmap':
        fig = plotting_utils.plot_correlation_heatmap(analyser.percent_correlation_matrix)
    elif mode == 'returns':
        fig = plotting_utils.create_returns_distribution_chart(analyser.percent_returns)
    elif mode == 'efficient_frontier':
        optimisation_results = None
        
        # Load from cache
        if not force_update and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    optimisation_results = pickle.load(f)
                print("Loaded frontier from cache")
            except:
                current_app.logger.error(f"Cache load failed")
                
        # Perform optimisation
        if optimisation_results is None:
            print("Calculating efficient frontier")

            opt = PortfolioOptimiser(analyser)

            # Prepare inputs from Analyser
            inputs = analyser.get_optimisation_inputs()

            bounds, constraints = opt.setup_optimisation_constraints(inputs['tickers'], 0.2, True)

            optimisation_results = opt.perform_full_analysis(bounds, constraints)

            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(optimisation_results, f)
            
        fig = plotting_utils.plot_efficient_frontier_and_portfolios(optimisation_results, analyser.asset_analysers)
    
    fig_json = fig.to_json()

    #print("get_portfolio_data out")

    return jsonify({
        'fig_data': fig_json,
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
