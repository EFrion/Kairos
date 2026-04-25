from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify, current_app
import json
import os
from app.utils import plotting_utils, storage_utils
from app.models import PortfolioManager
from app.utils.time_debug import timed

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
        storage_utils=storage_utils,
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
     
@bp.route('/update_portfolio_data/<asset_type>', methods=['POST'])
@timed
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
        storage_utils=storage_utils,
        finance_managers=finance_managers,
    )
    income_plot = plotting_utils.create_income_plot(portfolio.total_income_data)
    return {'portfolio': portfolio, 'income_plot': income_plot} 

@bp.route('/save_single_value/<asset_type>', methods=['POST'])
def save_single_value(asset_type):
    """
    Handles saving feature update from an AJAX request.
    """
    print(f"save_single_value called for asset type {asset_type}")
    data = request.get_json()
    ticker = data.get('ticker')
    field = data.get('field')
    value = data.get('value')
    
    if not all([ticker, field, isinstance(value, (int, float))]):
        return jsonify({'status': 'error', 'message': 'Invalid data received.'}), 400
        
    loaders = {
        'shares': storage_utils.load_shares, 'price': storage_utils.load_prices,
        'env': storage_utils.load_env, 'soc': storage_utils.load_soc,
        'gov': storage_utils.load_gov, 'cont': storage_utils.load_cont
    }
    savers = {
        'shares': storage_utils.save_shares, 'price': storage_utils.save_prices,
        'env': storage_utils.save_env, 'soc': storage_utils.save_soc,
        'gov': storage_utils.save_gov, 'cont': storage_utils.save_cont
    }
    
    if field not in loaders:
        return jsonify({'status': 'error', 'message': f'Unknown field: {field}'}), 400

    current_data = loaders[field](asset_type)
    current_data[ticker] = max(0.0, value)
    savers[field](current_data, asset_type)
    #session[field] = current_data
    
    # Recalculate and return updated portfolio in the same request
    portfolio_data = get_portfolio_data_from_cache()
    print("save_single_value out")
    return jsonify({
        'status': 'success',
        'portfolio': portfolio_data['portfolio'].to_dict(),
        'income_plot': portfolio_data['income_plot'].to_json()
    })
  
@bp.route('/save_cash', methods=['POST'])
def save_cash():
    data = request.get_json()
    cash_value = data.get('cash', 0)
    
    storage_utils.save_cash(cash_value)
    
    # Store in session so it persists for the user
    session['free_cash'] = float(cash_value)
    
    return jsonify({'status': 'success', 'saved_value': cash_value})
      

@bp.route('/add/<asset_class>', methods=['POST'])
def add_asset(asset_class):
    ticker = request.form.get('ticker', '').upper().strip()
    if ticker:
        assets = storage_utils.get_assets(asset_class)
        if ticker not in assets:
            assets.append(ticker)
            save_assets(assets, asset_class)
            print(f"Added {ticker} to portfolio.")
            # Download data for the new ticker before responding
            finance_manager = current_app.config['FINANCE_MANAGERS'][asset_class]
            interval = current_app.config['APP_CONFIG'].get("live_interval")
            finance_manager.get_metrics(assets, interval=interval, force=False)
    return '', 200
    
def save_assets(asset_list, asset_class='stocks'):
    print("save_assets called")
    data_dir = current_app.config['DATA_FOLDER']
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir,f"{asset_class}_list.json")
    unique_assets = sorted(list(set(asset_list))) # Use set to avoid duplicates
    with open(path, 'w') as f:
        json.dump(unique_assets, f)
        
@bp.route('/delete/<asset_class>/<ticker>', methods=['POST'])
def delete_asset(asset_class, ticker):
    print(f"delete_asset called for: {asset_class}/{ticker}")
    assets = storage_utils.get_assets(asset_class)
    if ticker not in assets:
        return jsonify({'status': 'error', 'message': 'Ticker not found.'}), 404

    # Remove from main asset list
    assets.remove(ticker)
    save_assets(assets, asset_class)
    
    # Clean up associated share & average price data
    # TODO reduce the amount of files used?
    for loader, saver in [
        (storage_utils.load_shares, storage_utils.save_shares),
        (storage_utils.load_prices, storage_utils.save_prices),
        (storage_utils.load_env,    storage_utils.save_env),
        (storage_utils.load_soc,    storage_utils.save_soc),
        (storage_utils.load_gov,    storage_utils.save_gov),
        (storage_utils.load_cont,   storage_utils.save_cont),
    ]:
        d = loader(asset_class)
        d.pop(ticker, None)
        saver(d, asset_class)
    
    # Clean up metrics and price history
    live_interval = current_app.config['APP_CONFIG'].get("live_interval")
    research_interval = current_app.config['APP_CONFIG'].get("research_interval")
    finance_manager = current_app.config['FINANCE_MANAGERS'][asset_class]
    finance_manager.remove_ticker(ticker,live_interval)
    finance_manager.remove_ticker(ticker,research_interval)
    
    # Return fresh portfolio data, no reload needed
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
