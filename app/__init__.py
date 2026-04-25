from flask import Flask
from app.utils import database
from app.utils.config import AppConfig
from app.utils.finance_data import FinanceDataManager
from app.services.data_fetching import ResearchDataManager
from config import Config
    
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config) 

    # User config (intervals, CAGR years, etc.)
    app_config = AppConfig(app.config['USER_CONFIG_PATH'])
    app.config['APP_CONFIG'] = app_config   # attach to app for access in routes
    app.config['FINANCE_MANAGERS'] = {
        'stocks': FinanceDataManager(app.config['DATA_FOLDER'], 'stocks', app_config),
        'crypto': FinanceDataManager(app.config['DATA_FOLDER'], 'crypto', app_config),
    }
    app.extensions["research_dm"] = ResearchDataManager(
        app.config['FINANCE_MANAGERS'],
        app_config
    )

    from .routes import cashflow, portfolio, research
    app.register_blueprint(cashflow.bp)
    app.register_blueprint(portfolio.bp)
    app.register_blueprint(research.bp)
    
    # Initialise a database    
    with app.app_context():
        database.init_db()

    app.config['TEMPLATES_AUTO_RELOAD'] = True  #TODO CHANGE THIS IN PRODUCTION!
    
    # Register filters with Jinja2
    from .routes.cashflow import datetime_format
    app.jinja_env.filters['strftime'] = datetime_format
    
    return app
