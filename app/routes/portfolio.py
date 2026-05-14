from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify, current_app
import json
import os
from app.utils import plotting_utils
from app.utils.storage_utils import AssetDataManager, PortfolioDataManager
from app.models import PortfolioManager
from app.utils.time_debug import timed
import logging
logger = logging.getLogger(__name__)

bp = Blueprint('portfolio', __name__)

@bp.route('/portfolio', methods=['GET', 'POST'])
@bp.route('/', methods=['GET', 'POST'])
@timed
def portfolio_feature():
    app_config = current_app.config['APP_CONFIG']
    data = get_portfolio_data_from_cache()
    
    return render_template(
        'portfolio.html',
        title='Portfolio Dashboard',
        portfolio=data['portfolio'],
        income_plot=data['income_plot'].to_html(full_html=False, include_plotlyjs='cdn'),
        live_interval_ms=_interval_to_ms(app_config.get("live_interval"))
    )

@timed
def get_portfolio_data(force_update=False):
    interval = current_app.config['APP_CONFIG'].get("live_interval")
    finance_managers = current_app.config['FINANCE_MANAGERS']

    # TODO automatic asset classes handling
    portfolio = PortfolioManager.from_storage(
        asset_classes=['stocks', 'crypto'],
        finance_managers=finance_managers,
        interval=interval,
        force_update=force_update
    )

    income_plot = plotting_utils.create_income_plot(portfolio.total_income_data)

    return {
        'portfolio': portfolio,
        'income_plot': income_plot
    }


def _interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds for JS polling."""
    units = {"m": 60, "h": 3600, "d": 86400}
    try:
        value = int(interval[:-1])
        unit = interval[-1]
        return value * units[unit] * 1000
    except (ValueError, KeyError):
        return 15 * 60 * 1000  # fallback: 15 minutes

# TODO remove this function?
@bp.route('/update_portfolio_cache', methods=['POST'])
@timed
def update_portfolio_cache():
    """ Loads cached data when app opens. """                       
    data = get_portfolio_data()
        
    return jsonify({
        'portfolio': data['portfolio'].to_dict(),
        'income_plot': data['income_plot'].to_json()
    })
     
@bp.route('/update_portfolio_data', methods=['POST'])
def update_portfolio_data():
    """Called on UI changes (shares, price, ESG). No market data fetch."""
    data = get_portfolio_data_from_cache()

    return jsonify({
        'portfolio': data['portfolio'].to_dict(),
        'income_plot': data['income_plot'].to_json()
    })

@timed
def get_portfolio_data_from_cache():
    """Rebuild portfolio math from cached metrics only."""
    finance_managers = current_app.config['FINANCE_MANAGERS']        
    portfolio = PortfolioManager.from_cache(  
        asset_classes=['stocks', 'crypto'],
        finance_managers=finance_managers,
    )
    income_plot = plotting_utils.create_income_plot(portfolio.total_income_data)
    return {'portfolio': portfolio, 'income_plot': income_plot} 

@bp.route('/save_single_value/<asset_type>', methods=['POST'])
def save_single_value(asset_type):
    """
    Handles saving feature update from an AJAX request.
    """
    logger.debug(f"save_single_value called for asset type {asset_type}")
    data = request.get_json()
    ticker = data.get('ticker')
    field = data.get('field')
    value = data.get('value')
    
    if not all([ticker, field, isinstance(value, (int, float))]):
        return jsonify({'status': 'error', 'message': 'Invalid data received.'}), 400
        
    manager = AssetDataManager(asset_type)
    manager.update_ticker_metric(ticker, field, max(0.0, value))
    portfolio_data = get_portfolio_data_from_cache()
    logger.debug("save_single_value out")
    return jsonify({
        'status': 'success',
        'portfolio': portfolio_data['portfolio'].to_dict(),
        'income_plot': portfolio_data['income_plot'].to_json()
    })
  
@bp.route('/save_cash', methods=['POST'])
def save_cash():
    data = request.get_json()
    cash_value = data.get('cash', 0)
    
    #storage_utils.save_cash(cash_value)
    PortfolioDataManager().save_cash(cash_value)

    # Store in session so it persists for the user
    session['free_cash'] = float(cash_value)
    
    return jsonify({'status': 'success', 'saved_value': cash_value})
      

@bp.route('/add/<asset_class>', methods=['POST'])
def add_asset(asset_class):
    ticker = request.form.get('ticker', '').upper().strip()
    logger.debug(f"[ADD_ASSET] ticker={ticker}, asset_class={asset_class}")
    if ticker:
        #assets = storage_utils.get_assets(asset_class)
        manager = AssetDataManager(asset_class)
        current_tickers = manager.tickers
        #logger.debug(f"[ADD_ASSET] current assets before append: {assets}")
        if ticker not in current_tickers:
            current_tickers.append(ticker)
            #save_assets(assets, asset_class)
            # This assignment triggers the @setter in the class, 
            # which automatically sorts, de-duplicates, and saves the file.
            manager.tickers = current_tickers
            logger.info(f"Added {ticker} to {asset_class} portfolio.")

            # Download data for the new ticker before responding
            finance_manager = current_app.config['FINANCE_MANAGERS'][asset_class]
            live_interval = current_app.config['APP_CONFIG'].get("live_interval")
            research_interval = current_app.config['APP_CONFIG'].get("research_interval")
            
            finance_manager._ensure_prices([ticker], live_interval, force=True)
            if research_interval != live_interval:
                finance_manager._ensure_prices([ticker], research_interval, force=True)
            logger.debug(f"[ADD_ASSET] calling get_metrics with interval={live_interval}, tickers={manager.tickers}")
            
            finance_manager.get_metrics(manager.tickers, interval=live_interval, force=False)
            logger.debug(f"[ADD_ASSET] get_metrics returned")
    return '', 200
  
@bp.route('/delete/<asset_class>/<ticker>', methods=['POST'])
def delete_asset(asset_class, ticker):
    manager = AssetDataManager(asset_class)
    manager.delete_ticker_globally(ticker)
    
    # Clean up the external finance manager cache
    fm = current_app.config['FINANCE_MANAGERS'][asset_class]
    live_interval = current_app.config['APP_CONFIG'].get("live_interval")
    research_interval = current_app.config['APP_CONFIG'].get("research_interval")
    fm.remove_ticker(ticker, live_interval)
    fm.remove_ticker(ticker, research_interval)
    
    # Return fresh data
    portfolio_data = get_portfolio_data_from_cache()
    return jsonify({
        'status': 'success',
        'portfolio': portfolio_data['portfolio'].to_dict(),
        'income_plot': portfolio_data['income_plot'].to_json()
    })

# Needed for returning correctly formatted numbers at initialisation
@bp.app_template_filter('format_finance')
def format_finance(val):
    try:
        val = float(val)
        if val == 0: return "0.00"
        if 0 < abs(val) < 0.01: return f"{val:.2e}"
        return f"{val:,.2f}".replace(",", " ")
    except:
        return "N/A"