import numpy as np
import pandas as pd
from scipy import stats
from functools import cached_property

class Metric:
    registry = []

    def __init__(self, label, type="number", suffix=""):
        self.label = label
        self.type = type
        self.suffix = suffix

    def __call__(self, func):
        metric_info = {
            "id": func.__name__,
            "label": self.label,
            "type": self.type,
            "suffix": self.suffix
        }

        Metric.registry.append(metric_info)

        return property(func)
    
class AssetAnalyser:
    def __init__(self, asset, price_history):
        self.asset = asset

        self.data = price_history[self.asset.ticker].dropna()
        self.percent_returns = self.data.pct_change().dropna()
        
        # Annualisation factor: 252 for daily data
        self.ann_factor = 252
        self.risk_free_rate = 0.0

    @Metric(label="Sharpe ratio")
    def percent_sharpe_ratio(self):
        return sharpe_ratio(self.percent_returns, self.risk_free_rate)
    
    @Metric(label="Semivariance") # TODO returns?
    def percent_semivariance(self):
        return semivariance(self.percent_returns)
    
    @Metric(label="Sortino ratio")
    def percent_sortino_ratio(self):
        return sortino_ratio(self.percent_returns, self.risk_free_rate)
        
    @Metric(label="Symmetry score", suffix="%")
    def percent_symmetry_score(self):
        return symmetry_score(self.percent_returns)
        
    @cached_property
    def percent_dp_normality_test(self):
        return dp_normality_test(self.percent_returns)

    @Metric(label="D Agostino-Pearson stats")    
    def percent_normal_dp_stat(self):
        stat, _ = self.percent_dp_normality_test
        return stat
        
    @Metric(label="D Agostino-Pearson p-value")
    def percent_normal_dp_pvalue(self):
        _, pvalue = self.percent_dp_normality_test
        return pvalue
    
    @cached_property
    def percent_jb_normality_test(self):
        return jb_normality_test(self.percent_returns)

    @Metric(label="Jarque-Bera stats")    
    def percent_normal_jb_stat(self):
        stat, _ = self.percent_jb_normality_test
        return stat
        
    @Metric(label="Jarque-Bera p-value")
    def percent_normal_jb_pvalue(self):
        _, pvalue = self.percent_jb_normality_test
        return pvalue

    @cached_property
    def percent_z_score(self):
        return z_score(self.percent_returns)
    
    @Metric(label="Z-score max")
    def percent_zmax(self):
        max, _ = self.percent_z_score
        return max
    
    @Metric(label="Z-score min")
    def percent_zmin(self):
        _, min = self.percent_z_score
        return min
    
    @Metric(label="Number outliers")
    def percent_num_outliers(self):
        return num_outliers(self.percent_returns)
    
    @property
    def annual_return(self):
        return self.percent_returns.mean() * self.ann_factor

    @property
    def annualised_volatility(self):
        return self.percent_returns.std() * np.sqrt(self.ann_factor)
    
    @Metric(label="Hist. VaR 95", suffix='%')
    def percent_historical_var(self):
        return historical_var(self.percent_returns, 0.95)*100
    
    @Metric(label="Hist. CVaR 95", suffix='%')
    def percent_historical_cvar(self):
        return historical_cvar(self.percent_returns, 0.95)*100
    
    @cached_property
    def student_t_params(self):
        # Fit Student's t distribution to returns once and cache
        return stats.t.fit(self.percent_returns)

    @Metric(label="Student-t VaR 95", suffix='%')
    def percent_student_t_var(self):
        return student_t_var(self.student_t_params, 0.95)*100


class PortfolioAnalyser:
    def __init__(self, portfolio, price_history, risk_free_rate=0.0):
        self.portfolio = portfolio
        self.tickers = [a.ticker for a in portfolio.assets]
        
        self.data = price_history[self.tickers].dropna()
        self.returns = self.data.pct_change().dropna()
        self.ann_factor = 252
        self.risk_free_rate = risk_free_rate

        self.asset_analysers = self._build_asset_analysers()

    def _build_asset_analysers(self):
        analysers = {}

        for asset in self.portfolio.assets:
            analysers[asset.ticker] = AssetAnalyser(
                asset,
                self.data
            )

        return analysers

    @property
    def individual_annual_returns(self):
        """Returns a numpy array of annual returns for all assets in the portfolio."""
        return np.array([a.annual_return for a in self.asset_analysers.values()])

    def get_optimisation_inputs(self):
        """Package everything needed for the PortfolioOptimiser."""
        return {
            "annual_returns": self.individual_annual_returns,
            "covariance_matrix": self.ann_covariance_matrix.values,
            "initial_weights": self.current_weights,
            "tickers": self.tickers,
            "daily_returns": self.returns,
            "risk_free_rate": self.risk_free_rate
        }

    @property
    def current_weights(self):
        """Extracts weights directly from the Asset object."""
        return np.array([a.weight / 100 for a in self.portfolio.assets])

    @property
    def percent_returns(self):
        return self.returns @ self.current_weights
    
    @property
    def log_returns(self):
        return get_log_returns(self.percent_returns)
    
    @property
    def ann_log_returns(self):
        return self.log_returns.mean() * self.ann_factor

    @property
    def ann_percent_returns(self):
        geometric_mean = ((1+self.percent_returns).prod()**(1/len(self.percent_returns)))-1  
        return (1+geometric_mean)**self.ann_factor - 1

    @property
    def percent_correlation_matrix(self):
        return self.returns.corr()

    @property
    def correlation_matrix(self):
        return self.log_returns.corr()
    
    @property
    def ann_covariance_matrix(self):
        return self.returns.cov() * self.ann_factor
    
    @property
    def variance(self):
        return self.weights.T @ self.returns.cov() @ self.weights
        
    @property
    def std(self):
        return np.sqrt(self.variance)
    
    @property
    def sharpe_ratio(self):
        return sharpe_ratio(self.percent_returns, self.risk_free_rate)

    @property
    def sortino_ratio(self):
        return sortino_ratio(self.percent_returns, self.risk_free_rate)
    
    def get_individual_metrics_data(self):
        data = {}
        for ticker, analyser in self.asset_analysers.items():
            metrics = {}
            for metric in Metric.registry:
                metric_id = metric["id"]
                # Read attribute from AssetAnalyser
                value = getattr(analyser, metric_id)
                if isinstance(value, float):
                    value = value

                metrics[metric_id] = value

            data[ticker] = metrics
        
        return data
    
    def to_dict(self):
        return {
            'schema': Metric.registry,
            'assets': self.get_individual_metrics_data(),
            'portfolio_stats': {
                'sharpe': f"{self.sharpe_ratio:.2f}",
                'sortino': f"{self.sortino_ratio:.2f}"
            }
        }

    # Function to calculate Beta
    def calculate_beta(self, returns, benchmark_returns):
        """
        Beta = Covariance(Asset, Benchmark) / Variance(Benchmark)
        """
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        
        return covariance / benchmark_variance

    # Function to calculate Alpha
    def calculate_alpha(self, annual_return, beta, risk_free_rate, benchmark_annual_return):
        """
        Alpha = R_stock/portfolio - [R_f + Beta * (R_benchmark - R_f)
        """
        expected_return = risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
        return annual_return - expected_return
        

    def portfolio_return(self, weights, annual_returns):
        return np.sum(weights * annual_returns)

    def portfolio_volatility(self, weights, covariance_matrix):
        return np.sqrt(weights.T @ covariance_matrix @ weights)
    
    def _calculate_portfolio_metrics_full(self, weights, annual_returns, daily_returns_df_slice, annualised_covariance_matrix, risk_free_rate, benchmark_returns, benchmark_annual_return, lambda_s=None, lambda_k=None):
        """
        Calculates a comprehensive set of portfolio metrics.
        
        Args:
            weights (np.array): Array of weights for each asset.
            annual_returns (np.array): Array of annualised returns for each asset.
            daily_returns_df_slice (pd.DataFrame): Daily returns for the lookback period.
            annualised_covariance_matrix (np.array): Annualised covariance matrix of asset returns.
            risk_free_rate (float): Risk-free rate.
            benchmark_returns (pd.Series): The daily returns of the benchmark.
            benchmark_annual_return (float): The annualised return of the benchmark.
            lambda_s (float, optional): Coefficient for skewness (for MVSK).
            lambda_k (float, optional): Coefficient for kurtosis (for MVSK).
            
        Returns:
            dict: A dictionary of calculated metrics.
        """
        metrics = {}
        
        p_return = self.portfolio_return(weights, self.individual_annual_returns)
        p_volatility = self.portfolio_volatility(weights, self.ann_covariance_matrix.values)
        #p_beta = self.calculate_beta(daily_returns_df_slice.dot(weights), benchmark_returns)
        #p_alpha = self.calculate_alpha(p_return, p_beta, risk_free_rate, benchmark_annual_return)
        
        metrics.update({
            'Return': p_return,
            'Volatility': p_volatility,
            'Sharpe Ratio': (p_return - self.risk_free_rate) / p_volatility if p_volatility > 0 else 0
        })
        #metrics['Beta'] = p_beta
        #metrics['Alpha'] = p_alpha
        
        if p_volatility > 0:
            metrics['Sharpe Ratio'] = (p_return - risk_free_rate) / p_volatility
        else:
            metrics['Sharpe Ratio'] = np.inf if p_return > risk_free_rate else np.nan # Handle zero volatility

        p_downside_dev = self.downside_deviation(weights, daily_returns_df_slice, risk_free_rate)
        if p_downside_dev > 0:
            metrics['Sortino Ratio'] = (p_return - risk_free_rate) / p_downside_dev
        else:
            metrics['Sortino Ratio'] = np.inf if p_return > risk_free_rate else np.nan # Handle zero downside deviation

        metrics['Skewness'] = self.portfolio_skewness(weights, daily_returns_df_slice)
        metrics['Kurtosis'] = self.portfolio_kurtosis(weights, daily_returns_df_slice)
        
        # MVSK Utility
        if lambda_s is not None and lambda_k is not None:
            if p_volatility > 0:
                metrics['MVSK Utility'] = (p_return - risk_free_rate) / p_volatility + lambda_s * metrics['Skewness'] - lambda_k * metrics['Kurtosis']
            else:
                metrics['MVSK Utility'] = np.inf if p_return > risk_free_rate else np.nan
                
        return metrics
            
    # Calculate only undesired volatility (downside risk)
    def downside_deviation(self, weights, daily_returns_df_slice, risk_free_rate):
        """
        Calculates the annualised downside deviation for a portfolio.
        Only considers returns below the Minimum Acceptable Return (MAR), which is the risk-free rate.
        
        Args:
            weights (np.array): Array of weights for each asset.
            daily_returns_df_slice (pd.DataFrame): Daily returns for the lookback period.
            risk_free_rate (float): Annualised risk-free rate.
        """
        # Calculate portfolio daily returns for the slice
        portfolio_daily_returns = daily_returns_df_slice.dot(weights)
        
        # Calculate daily MAR
        daily_mar = (1 + risk_free_rate)**(1/252) - 1 # Convert annualised risk-free rate to daily

        # Filter for returns below the MAR
        downside_returns = portfolio_daily_returns[portfolio_daily_returns < daily_mar]
        
        if downside_returns.empty:
            return 0.0 # No downside returns, so downside deviation is 0

        # Calculate downside deviation (standard deviation of downside returns)
        downside_std = np.sqrt(np.mean((downside_returns - daily_mar)**2))
        
        # Annualise downside deviation
        annualised_downside_std = downside_std * np.sqrt(252)
        return annualised_downside_std
        
    def portfolio_skewness(self, weights, daily_returns_df_slice):
        """
        Calculates the skewness for a portfolio's daily returns.
        """
        portfolio_daily_returns = daily_returns_df_slice.dot(weights)
        return portfolio_daily_returns.skew()
        
    # Calculate extreme events in returns distribution
    def portfolio_kurtosis(self, weights, daily_returns_df_slice):
        """
        Calculates the kurtosis for a portfolio's daily returns.
        """
        portfolio_daily_returns = daily_returns_df_slice.dot(weights)
        return portfolio_daily_returns.kurtosis()
        

# Standalone functions

def historical_var(returns, confidenceLevel):
    return np.quantile(returns, 1 - confidenceLevel)

def historical_cvar(returns, confidenceLevel):
    return returns[returns <= historical_var(returns,confidenceLevel)].mean()

def student_t_var(params, confidenceLevel):
    dof, loc, scale = params
    # Calculate the quantile for the left tail (VaR)
    quantile = stats.t.ppf(1 - confidenceLevel, dof, loc=loc, scale=scale)
    # Return quantile as a number (e.g., decimal return)
    return quantile

def sharpe_ratio(returns, risk_free_rate=0.0, ann_factor=1):
    """
    Calculate the Sharpe ratio
    Formula: Sharpe ratio = (R_p-R_fr)/sigma_p
    """
    excess_return = returns.mean() - risk_free_rate
    return (excess_return / returns.std()) * np.sqrt(ann_factor)

def semivariance(returns, ann_factor=1):
    """
    Calculate the semivariance
    Formula: Semivariance = (Sum_{r_i < <r>}^{n} (r_i - <r>)²) / n
    """
    # Average on all observations
    mean_return = returns.mean()
    downside_diff = (returns - mean_return).clip(upper=0) # Set positive deviations to 0
    semivar = (downside_diff ** 2).mean()
    #print("stocks_semivariance: ", stocks_semivariance)

    # Average only on bad days
    #    stocks_mean2 = price_history.mean()
    #    stocks2_semivariance = ((price_history[price_history < stocks_mean2] - stocks_mean2) ** 2).mean()
    #    print("stocks2_semivariance: ", stocks2_semivariance)
    return semivar

def sortino_ratio(returns, risk_free_rate=0.0, ann_factor=1): #TODO merge with semivariance?
    """
    Calculate the Sortino ratio
    Formula: Sortino ratio = (R_p+-R_fr)/sigma_p+
    """
    # Average on all observations
    mean_return = returns.mean()
    downside_diff = (returns - mean_return).clip(upper=0)
    semivar = (downside_diff ** 2).mean()
    semistd = np.sqrt(semivar)
    excess_return = mean_return - risk_free_rate
    #print("stocks_semistd: ", stocks_semistd)

    return (excess_return / semistd) * np.sqrt(ann_factor)
    
def get_log_returns(self,percent_returns):
    value = (1+self.percent_returns).cumprod()
    return np.log(value).diff().dropna()

def symmetry_score(returns):
    counts = (returns > returns.mean()).sum()
    #print("counts: ", counts)
    total = returns.count()
    #print("total: ", total)
    return (counts/total)*100

def dp_normality_test(returns):
    # Normality test (D'Agostino-Pearson)
    stat, pvalue = stats.normaltest(returns)
    return stat, pvalue

def jb_normality_test(returns):
    # Normality test (Jarque-Bera)
    stat, pvalue = stats.jarque_bera(returns)
    return stat, pvalue

def z_score(returns):
    max = returns.max()
    min = returns.min()
    mean = returns.mean()
    std = returns.std()
    z_score_max = (max - mean) / std
    z_score_min = (min - mean) / std
    return z_score_max, z_score_min

def num_outliers(returns):
    mean = returns.mean()
    std = returns.std()
    # Number of outliers (deviation from normality)
    upper_bound = 3*std + mean
    lower_bound = -3*std + mean
    len_returns_below = len(returns[returns<lower_bound].dropna())
    len_returns_above = len(returns[returns>upper_bound].dropna())
    return len_returns_below + len_returns_above
