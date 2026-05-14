import json
import os
from flask import current_app
import tempfile
import logging
logger = logging.getLogger(__name__)

class JSONPersistenceManager:
    """Base class to handle atomic JSON operations."""
    
    def __init__(self, data_dir=None):
        self._data_dir = data_dir or current_app.config['DATA_FOLDER']
        os.makedirs(self._data_dir, exist_ok=True)

    def _get_path(self, filename):
        return os.path.join(self._data_dir, filename)

    def load(self, filename, default=None):
        path = self._get_path(filename)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return default if default is not None else {}
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading {path}: {e}")
            return default if default is not None else {}

    def save(self, filename, data):
        path = self._get_path(filename)
        dir_name = os.path.dirname(path)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, suffix='.tmp') as tmp:
                json.dump(data, tmp, indent=4)
                tmp_path = tmp.name
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"Error saving to {path}: {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

class AssetDataManager(JSONPersistenceManager):
    """Manages data for a specific asset type (e.g., 'crypto', 'stocks')."""
    
    def __init__(self, asset_type):
        super().__init__()
        self.asset_type = asset_type
        self.filename = f"{asset_type}_data.json"
        self.data = self.load(self.filename, default={
            'tickers': [],
            'shares': {}, 'price': {}, 'env': {}, 
            'soc': {}, 'gov': {}, 'cont': {}, 'syield': {}
        })

    @property
    def tickers(self):
        return self.data.get('tickers', [])
    
    @tickers.setter
    def tickers(self, ticker_list):
        self.data['tickers'] = sorted(list(set(ticker_list)))
        self.save(self.filename, self.data)
    
    def get_data(self, kind):
        """Returns the specific dictionary (e.g., 'shares') from memory."""
        return self.data.get(kind, {})

    def save_data(self, kind, data):
        """Updates the in-memory dictionary and persists the whole file."""
        self.data[kind] = data
        self.save(self.filename, self.data)

    def update_ticker_metric(self, ticker, field, value):
        """Updates a specific metric for a ticker and saves."""
        if field not in self.data:
            self.data[field] = {}
        self.data[field][ticker] = value
        self.save(self.filename, self.data)

    def delete_ticker_globally(self, ticker):
        """Removes ticker from the list and all metric dictionaries."""
        # Remove from ticker list
        if ticker in self.data['tickers']:
            self.data['tickers'].remove(ticker)
        
        # Remove from all metric categories
        for key, value in self.data.items():
            if isinstance(value, dict):
                value.pop(ticker, None)
        
        self.save(self.filename, self.data)

    def update_ticker_metric_batch(self, kind, data_dict):
        self.data[kind] = data_dict
        self.save(self.filename, self.data)

class PortfolioDataManager(JSONPersistenceManager):
    """Handles global/cross-asset data like cash and forex."""
    
    def get_cash(self):
        return self.load('global_cash.json').get('free_cash', 0.0)

    def save_cash(self, amount):
        self.save('global_cash.json', {'free_cash': float(amount)})

    def get_forex_rates(self):
        data = self.load('last_fetch_date.json')
        return {k: v for k, v in data.items() if k.endswith('_rate')}