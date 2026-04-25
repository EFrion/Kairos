from flask import Blueprint, render_template, request, jsonify, current_app
from app.utils import plotting_utils, storage_utils, finance_data
from app.models import PortfolioManager
from app.analytics.optimiser import PortfolioOptimiser
from app.services.data_fetching import ResearchDataManager
from app.analytics.analyser import AssetAnalyser, PortfolioAnalyser
import os
import pickle
import pandas as pd
from flask.views import MethodView

bp = Blueprint('research', __name__)

# TODO focus on stocks for now, add a general function later
class PortfolioView(MethodView):
    # def __init__(self):
    #     finance_managers = current_app.config['FINANCE_MANAGERS']

    #     # Initialise data manager once per view instance
    #     self.dm = ResearchDataManager(finance_managers)
    #     self.stocks_data = None

    def get(self):
        """
        Handle GET requests to /research: render the main portfolio page with initial data.
        """
        dm = current_app.extensions["research_dm"]

        # Load stocks data
        stocks_data = dm.get_data(asset_type='stocks')
        if stocks_data is None or stocks_data.empty:
            return "Error: historical prices could not be loaded.", 404

        # Check for NaNs
        if stocks_data.isnull().values.any():
            current_app.logger.warning("NaN values detected in stocks data")

        # Get sorted tickers
        tickers = sorted(stocks_data.columns.tolist())

        # Determine selected ticker (default first)
        selected_ticker = request.args.get('ticker', tickers[0])

        self._rebuild_analyser(stocks_data)  # side effect isolated here
        analyser = current_app.extensions["portfolio_analyser"]

        return render_template(
            'research.html',
            tickers=analyser.tickers,
            analysis=analyser.to_dict(),
            selected_ticker=selected_ticker,
            title='Research'
        )
    
    def _rebuild_analyser(self, stocks_data: pd.DataFrame) -> None:
        finance_managers = current_app.config['FINANCE_MANAGERS']
        portfolio = PortfolioManager.from_cache(
            asset_classes=['stocks'],
            storage_utils=storage_utils,
            finance_managers=finance_managers,
        )
        analyser = PortfolioAnalyser(portfolio.stocks, stocks_data)
        current_app.extensions["portfolio_analyser"] = analyser

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
bp.add_url_rule('/research', view_func=PortfolioView.as_view('portfolio_research'))
bp.add_url_rule('/get_data', view_func=PortfolioDataAPI.as_view('portfolio_data_api'))


@bp.route('/get_portfolio_data')
def get_portfolio_data():
    print("get_portfolio_data called")

    mode = request.args.get('mode', 'returns')
    if not mode:
        return "No mode provided", 400
        
    force_update = request.args.get('force_update') == 'true' # Check for button click
    print(f"[DEBUG] get_portfolio_data called: mode={mode}, force_update={force_update}")


    analyser = current_app.extensions.get("portfolio_analyser")
    if analyser is None:
            return jsonify({"error": "Portfolio analyser not initialised"}), 500
    
    print(f"[DEBUG] calling _build_figure")
    fig = _build_figure(mode, analyser, force_update)

    return jsonify({
            'fig_data': fig.to_json(),
        })

def _build_figure(mode, analyser, force_update):
    if mode == 'heatmap':
        return plotting_utils.plot_correlation_heatmap(analyser.percent_correlation_matrix)
    elif mode == 'returns':
        return plotting_utils.create_returns_distribution_chart(analyser.percent_returns)
    elif mode == 'efficient_frontier':
        results = _get_frontier(analyser, force_update)
        return plotting_utils.plot_efficient_frontier_and_portfolios(results, analyser.asset_analysers)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _get_frontier(analyser, force_update):
    print(f"[DEBUG] _get_frontier called, force_update={force_update}")
    cache_path = os.path.join(current_app.config['DATA_FOLDER'], "frontier_cache.pkl")
    print(f"[DEBUG] cache exists: {os.path.exists(cache_path)}")
    # Load from cache
    if not force_update and os.path.exists(cache_path):
        print("[DEBUG] loading from cache")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
            print("Loaded frontier from cache")
        except Exception:
            current_app.logger.error("Frontier cache load failed, recalculating")
    
    print("[DEBUG] computing frontier")
    # Perform optimisation
    try:
        opt = PortfolioOptimiser(analyser)
        inputs = analyser.get_optimisation_inputs()
        bounds, constraints = opt.setup_optimisation_constraints(inputs['tickers'], 0.2, True)
        results = opt.perform_full_analysis(bounds, constraints)
        print(f"[DEBUG] frontier computed successfully")
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"[DEBUG] frontier saved to cache")
        return results
    except Exception as e:
        print(f"[DEBUG] frontier computation failed: {e}")
        raise
            

#@bp.route('/expand_history/<asset_type>', methods=['POST']) #TODO
#def expand_history(asset_type):
@bp.route('/expand_history')
def expand_history():
    """
        Fetches data on user request.
    """
    # TODO Only for stocks at the moment
    
    # Fetch date inserted by user
    new_start = request.args.get('start')
    if not new_start:
        return jsonify({"message": "Missing start date"}), 400
    print("new_start: ", new_start)

    target_start = pd.to_datetime(new_start)
    finance = current_app.config['FINANCE_MANAGERS']['stocks']
    interval = current_app.config['APP_CONFIG'].get("research_interval")
    print(interval)
    tickers = finance._hist_prices.get(interval, pd.DataFrame()).columns.tolist()
    print(tickers)
    # Trigger the backfill logic
    if not tickers:
        # Fall back to metrics JSON for ticker list
        metrics = finance._load_json(finance._metrics_path, default={})
        tickers = list(metrics.keys())

    try:
        finance._ensure_prices(tickers, interval, force=True, target_start=target_start)
        return jsonify({"message": f"History expanded to {new_start}."})
    except Exception as e:
        current_app.logger.error(f"Expand error: {e}")
        return jsonify({"message": "Failed to expand history."}), 500
