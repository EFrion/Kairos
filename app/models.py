from types import SimpleNamespace
from app.utils import storage_utils
    
# Define a single asset (one per ticker)
class Asset:
    def __init__(self, ticker, metrics, asset_type='stocks', shares=0.0, avg_price=0.0, env=0, soc=0, gov=0, cont=0):
        self.ticker = ticker
        self.asset_type = asset_type
        self.metrics = metrics
        self.shares = float(shares)
        self.avg_price = float(avg_price)
        self.sector = metrics.get("Sector", "Null")
        self.quote = self._safe_float(metrics.get("Quote"))
        self.quote_eur = self._safe_float(metrics.get("Quote_EUR"))
        self.latest_div = self._safe_float(metrics.get("Latest_Div_EUR"))
        self.months_paid = metrics.get("Months_Paid", [0]*12)
        self.div_yield = self._safe_float(metrics.get("Div_Yield")*100)
        self.div_growth = self._safe_float(metrics.get("Div_CAGR")*100)
        self.pe_ratio = self._safe_float(metrics.get("P/E", 0))
        self.fwd_pe = self._safe_float(metrics.get("Fwd_P/E", 0))
        self.peg = self._safe_float(metrics.get("PEG", 0))
        self.pb_ratio = self._safe_float(metrics.get('P/B',0))
        self.bench_pb = self._safe_float(metrics.get('Sector_PB_Benchmark',0))
        self.earnings_gr = self._safe_float(metrics.get('Earnings_Growth',0)*100)
        self.payout_ratio = self._safe_float(metrics.get('PayoutRatio',0)*100)
        self.env = int(env)
        self.soc = int(soc)
        self.gov = int(gov)
        self.cont = int(cont)
        self.weight = 0.0
        
    # Ensure floats are retrieved from data
    def _safe_float(self, val):
        try:
            return float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
            
    @property
    def market_value(self):
        return self.shares * self.quote_eur
        
    @property
    def cost_basis(self):
        return self.shares * self.avg_price
        
    @property
    def asset_income(self):
        return self.market_value * self.div_yield
        
    def get_monthly_income(self):
        if not hasattr(self, 'months_paid'): return [0.0] * 12
            
        return [self.latest_div * self.shares if m == 1 else 0.0 for m in self.months_paid]
    
    @property
    def annual_dividend(self):
        payment_frequency = sum(self.months_paid)
        total = self.latest_div * payment_frequency * self.shares
        return total
    
    # Determine background colour. TODO: give user choice instead of hard coding
    def get_metric_config(self, metric_name):
        # Conditional rule
        if self.asset_type == 'crypto':
            weight_rules = {'low': 40, 'high': 50, 'dir': 'low_is_better'}
        else:
            weight_rules = {'low': 4, 'high': 5, 'dir': 'low_is_better'}

        # Fixed rules
        rules = {
            'pe_ratio': {'val': self.pe_ratio, 'low': 10,  'high': 20, 'dir': 'low_is_better'},
            'fwd_pe': {'val': self.fwd_pe,   'low': 10,  'high': 20, 'dir': 'low_is_better'},
            'earnings_gr': {'val': self.earnings_gr,   'low': 0,  'high': 0, 'dir': 'high_is_better'},
            'peg': {'val': self.peg,   'low': 0.9,  'high': 1, 'dir': 'low_is_better'},
            'div_yield': {'val': self.div_yield, 'low': 2, 'high': 4, 'dir': 'high_is_better'},
            'div_growth': {'val': self.div_growth, 'low': 4, 'high': 8, 'dir': 'high_is_better'},
            'payout_ratio': {'val': self.payout_ratio, 'low': 90, 'high': 100, 'dir': 'low_is_better'},
            'env': {'val': self.env,      'low': 4,   'high': 6,  'dir': 'high_is_better'},
            'soc': {'val': self.soc,      'low': 4,   'high': 6,  'dir': 'high_is_better'},
            'gov': {'val': self.gov,      'low': 4,   'high': 6,  'dir': 'high_is_better'},
            'cont': {'val': self.cont,     'low': 4,   'high': 6,  'dir': 'high_is_better'},
            'pb_ratio': {'val': self.pb_ratio, 'low': self.bench_pb, 'high': self.bench_pb, 'dir': 'low_is_better'},
            'weight': {'val': self.weight, **weight_rules}
        }

        config = rules.get(metric_name.lower())
        if not config:
            return {'val': 0, 'class': ''}

        return {
            'val': config['val'],
            'class': calculate_color(config['val'], config['low'], config['high'], config['dir'])
        }
    
    # Define schema for the HTML renderer
    def get_schema(self):
        # Common columns
        schema = [
            {'id': 'ticker',    'label': 'Ticker',      'type': 'ticker'},
            {'id': 'shares',    'label': 'Shares',      'type': 'input'},
            {'id': 'avg_price', 'label': 'Avg Price',   'type': 'input'},
            {'id': 'market_value', 'label': 'Value (€)', 'type': 'finance', 'suffix': ' €'},
            {'id': 'quote',     'label': 'Quote',       'type': 'finance'},
            {'id': 'quote_eur', 'label': 'Quote (€)',   'type': 'finance', 'suffix': ' €'},
            {'id': 'weight',    'label': 'Weight',      'type': 'monitor', 'suffix': '%'},
        ]

        # Category-specific columns
        if self.asset_type == 'stocks':
            schema += [
                {'id': 'pe_ratio',    'label': 'P/E',         'type': 'monitor'},
                {'id': 'fwd_pe',      'label': 'Fwd P/E',     'type': 'monitor'},
                {'id': 'pb_ratio',    'label': 'P/B',         'type': 'monitor'},
                {'id': 'peg',         'label': 'PEG',         'type': 'monitor'},
                {'id': 'earnings_gr', 'label': 'Earnings gr', 'type': 'monitor', 'suffix': '%'},
                {'id': 'div_yield',   'label': 'Div. yield',  'type': 'monitor', 'suffix': '%'},
                {'id': 'div_growth',  'label': 'Div. CAGR',   'type': 'monitor', 'suffix': '%'},
                {'id': 'payout_ratio','label': 'Payout ratio','type': 'monitor', 'suffix': '%'},
                {'id': 'env',         'label': 'E',           'type': 'monitor_input'},
                {'id': 'soc',         'label': 'S',           'type': 'monitor_input'},
                {'id': 'gov',         'label': 'G',           'type': 'monitor_input'},
                {'id': 'cont',        'label': 'Cont',        'type': 'monitor_input'},
                {'id': 'latest_div',  'label': 'Latest div.', 'type': 'finance', 'suffix': ' €'},
                {'id': 'months_paid', 'label': 'Months paid', 'type': 'visualizer'},
                {'id': 'annual_dividend','label': 'Annual div','type': 'finance', 'suffix': ' €'},
                {'id': 'currency',    'label': 'Curr',        'type': 'text'},
                {'id': 'sector',      'label': 'Sector',      'type': 'text'},
            ]
        else:
            schema += [
                {'id': 'staking_yield', 'label': 'Staking yield', 'type': 'input', 'placeholder': '0.05'}
            ]
        return schema
    
    # Convert to a dictionary for the frontend
    def to_dict(self):
        # Basic attributes
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        # Properties
        data.update({
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'annual_dividend': self.annual_dividend,
            'asset_income': self.asset_income,
            'status_colors': {
                'weight': self.get_metric_config('weight')['class'],
                'pe_ratio': self.get_metric_config('pe_ratio')['class'],
                'fwd_pe': self.get_metric_config('fwd_pe')['class'],
                'earnings_gr': self.get_metric_config('earnings_gr')['class'],
                'peg': self.get_metric_config('peg')['class'],
                'pb_ratio': self.get_metric_config('pb_ratio')['class'],
                'div_yield': self.get_metric_config('div_yield')['class'],
                'div_growth': self.get_metric_config('div_growth')['class'],
                'payout_ratio': self.get_metric_config('payout_ratio')['class'],
                'weight': self.get_metric_config('weight')['class'],
                'env': self.get_metric_config('env')['class'],
                'soc': self.get_metric_config('soc')['class'],
                'gov': self.get_metric_config('gov')['class'],
                'cont': self.get_metric_config('cont')['class']
            }
        })
        return data
        
    def __repr__(self):
        return f"Asset({self.ticker}, MarketValue={self.market_value:.2f})"
        
# Define a portfolio of a given asset type (stocks, crypto, etc...)        
class Portfolio:
    def __init__(self, assets):
        self.assets = assets
        self.update_weights()

    @property
    def total_market_value(self):
        return sum(asset.market_value for asset in self.assets)
        
    @property
    def total_cost_basis(self):
        return sum(asset.cost_basis for asset in self.assets)
        
    @property
    def monthly_income_data(self):
        payouts = [0.0]*12
        counts = [0] * 12
        details = [[] for _ in range(12)] # List of lists to store stock details per month
        
        for asset in self.assets:
            asset_payouts = asset.get_monthly_income()
            for i in range(12):
                if asset_payouts[i] > 0:
                    payouts[i] += asset_payouts[i]
                    counts[i] += 1
                    details[i].append(f"{asset.ticker}: €{asset_payouts[i]:.2f}") # Details for hovering in plot

        # Process details to include the total income in plot
        final_details = []
        for i in range(12):
            if payouts[i] > 0:
                # Combine the ticker list and add the total header
                ticker_list = "<br>".join(details[i])
                final_details.append(f"<b>Total: €{payouts[i]:.2f}</b><br><br>{ticker_list}")
            else:
                final_details.append("No Income")
                
        return SimpleNamespace(
            payouts = payouts,
            counts = counts,
            details = final_details
        )
        
    @property
    def annual_dividends(self):
        return sum(self.monthly_income_data.payouts)
    
    @property
    def portfolio_yield_data(self):
        total_mv = self.total_market_value
        total_income = sum(asset.market_value * asset.div_yield for asset in self.assets)
        total_income_growth = sum(asset.market_value * asset.div_yield * asset.div_growth for asset in self.assets)
        
        if total_mv < 0 or total_income <= 0:
            return SimpleNamespace(
            div_yield = 0,
            div_growth= 0
        )

        return SimpleNamespace(
            div_yield = (total_income / total_mv),
            div_growth = (total_income_growth / total_income)
        )
        
    @property
    def sectors(self):
        sectors = {}
        for asset in self.assets:
            sector = asset.sector
            sectors[sector] = sectors.get(sector,0) + asset.market_value
            
        return SimpleNamespace(
            labels = list(sectors.keys()),
            values = list(sectors.values())
        )

    def update_weights(self):
        total_mv = self.total_market_value
        if self.total_market_value > 0:
            for asset in self.assets:
                asset.weight = (asset.market_value / total_mv) * 100
        else:
            for asset in self.assets:
                asset.weight = 0

    # Define schema for the HTML renderer
    def get_footer(self, asset_type):
        if not self.assets:
            return []

        # Get the column structure from the first asset
        asset_schema = self.assets[0].get_schema()
        
        # Define which IDs in the schema should have footer values
        if asset_type == 'stocks':
            footer_map = {
                'ticker':       {'label': 'Total Stocks', 'class': 'font-bold'},
                'avg_price':    {'val': self.total_cost_basis, 'id': 'total-cost-basis-stocks', 'type': 'finance'},
                'market_value': {'val': self.total_market_value, 'id': 'total-market-value-stocks', 'type': 'finance'},
                'div_yield':    {'val': self.portfolio_yield_data.div_yield * 100, 'id': 'portfolio-yield-display', 'type': 'finance', 'suffix': '%'},
                'div_growth':   {'val': self.portfolio_yield_data.div_growth * 100, 'id': 'portfolio-div-growth-display', 'type': 'finance', 'suffix': '%'},
                'months_paid':  {'type': 'visualizer'},
                'annual_dividend': {'val': self.annual_dividends, 'id': 'annual-dividends', 'type': 'finance', 'suffix': ' €'},
            }
        else:
            footer_map = {
                'ticker':       {'label': 'Total Crypto', 'class': 'font-bold'},
                'avg_price':    {'val': self.total_cost_basis, 'id': 'total-cost-basis-crypto', 'type': 'finance'},
                'market_value': {'val': self.total_market_value, 'id': 'total-market-value-crypto', 'type': 'finance'},
                'staking_yield':{'label': "Avg: N/A%", 'type': 'text'},
            }

        resolved_footer = []
        current_gap = 0

        # Iterate through the Asset schema and find matches
        for col in asset_schema:
            if col['id'] in footer_map:
                # If there's a gap, add a spacer cell
                if current_gap > 0:
                    resolved_footer.append({'type': 'spacer', 'colspan': current_gap})
                
                # Add the data
                cell_data = footer_map[col['id']]
                cell_data['colspan'] = 1
                resolved_footer.append(cell_data)
                
                # Reset gap
                current_gap = 0
            else:
                # No footer value for this column, increment gap
                current_gap += 1

        # Add trailing spacer if needed
        if current_gap > 0:
            resolved_footer.append({'type': 'spacer', 'colspan': current_gap})

        return resolved_footer
        
    def to_dict(self):
        return {
            'assets': [a.to_dict() for a in self.assets],
            'total_market_value': self.total_market_value,
            'total_cost_basis': self.total_cost_basis,
            'monthly_income_data': vars(self.monthly_income_data), # vars is used to extract dicts from object instance
            'annual_dividends': self.annual_dividends,
            'portfolio_yield_data': vars(self.portfolio_yield_data),
            'sectors': vars(self.sectors)
        }

    def __repr__(self):
        return f"Portfolio(Assets={len(self.assets)}, TotalValue=€{self.total_market_value:.2f})"

# Load external data into a portfolio
class PortfolioLoader:
    @staticmethod
    def load_asset_data(asset_type):
        return {
            'shares': storage_utils.load_shares(asset_type),
            'avg_price': storage_utils.load_prices(asset_type),
            'env': storage_utils.load_env(asset_type),
            'soc': storage_utils.load_soc(asset_type),
            'gov': storage_utils.load_gov(asset_type),
            'cont': storage_utils.load_cont(asset_type)
        }


# Define the complete portfolio
class PortfolioManager:
    def __init__(self, portfolios_dict, free_cash=0.0, silent=False):
        self._portfolios = portfolios_dict
        self.free_cash = free_cash
        # Set attributes based on asset classes (stocks, crypto, etc.)
        for name, portfolio_obj in portfolios_dict.items():
            setattr(self, name, portfolio_obj)
            
        # Show summary
        if not silent:
            print(self)
            
    @property
    def total_market_value(self):
        return sum(p.total_market_value for p in self._portfolios.values())
            
    @property
    def grand_total_cost_basis(self):
        return sum(p.total_cost_basis for p in self._portfolios.values())
        
    @property
    def grand_total_with_cash(self):
        return self.total_market_value + self.free_cash
        
    @property
    def total_income_data(self):
        grand_total = [0.0] * 12
        grand_details = [[] for _ in range(12)]
        
        for portfolio in self.values():
            report = portfolio.monthly_income_data
            for i in range(12):
                grand_total[i] += report.payouts[i]
                if report.payouts[i] > 0:
                    # Add a header for the asset class in the hover text
                    grand_details[i].append(report.details[i])
                    
        return {
            "payouts": grand_total,
            "details": ["<br>".join(d) for d in grand_details]
        }
    
    def __iter__(self):
        return iter(self._portfolios)
        
    def keys(self):
        return self._portfolios.keys()
        
    def values(self):
        return self._portfolios.values()
        
    def items(self):
        return self._portfolios.items()
        
    def to_dict(self):
        data = {name: p.to_dict() for name, p in self.items()}
        data['summary'] = {
            'total_market_value': self.total_market_value,
            'grand_total_cost_basis': self.grand_total_cost_basis,
            'grand_total_with_cash': self.grand_total_with_cash,
            'free_cash': self.free_cash,
            'total_income_data': self.total_income_data
        }
        return data
            
    def __repr__(self):
        count = 60
        # Header
        lines = [
            "\n" + "="*count,
            "PORTFOLIO SUMMARY",
            "="*count
        ]
        
        # Add each sub-portfolio
        for name, p in self._portfolios.items():
            lines.append(f" • {name.upper():<8}: {p.__repr__()}")
            
        # Add Global Totals
        lines.append("-" * count)
        lines.append(f" CASH       : €{self.free_cash:,.2f}")
        lines.append(f" TOTAL MV   : €{self.total_market_value:,.2f}")
        lines.append(f" GRAND TOTAL: €{self.grand_total_with_cash:,.2f}")
        lines.append("="*count + "\n")
        
        return "\n".join(lines)


# Generic background colour function
def calculate_color(value, low, high, direction='high_is_better'):
    """Python implementation of your Jinja color logic."""
    if not isinstance(value, (int, float)) or value <= 0:
        return "bg-red"
    
    if direction == 'high_is_better':
        if value < low: return "bg-red"
        if value <= high: return "bg-orange"
        return "bg-green"
    else: # low_is_better
        if value > high: return "bg-red"
        if value >= low: return "bg-orange"
        return "bg-green"























        
    
