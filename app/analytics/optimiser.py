import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimiser:
    def __init__(self, analyser):
        self.analyser = analyser
        self.returns = analyser.individual_annual_returns
        self.cov = analyser.ann_covariance_matrix.values
        self.rf = analyser.risk_free_rate

    # Function setting portfolio constraints
    def setup_optimisation_constraints(self, tickers, max_weight=0.2, long_only=True):
        n = len(tickers)

        bounds = []
        for _ in range(n):
            if long_only:
                bounds.append((0, max_weight))
            else:
                bounds.append((-max_weight, max_weight))

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        return bounds, constraints

    def _min_variance_objective(self, weights, covariance):
        return self.analyser.portfolio_volatility(weights, covariance)

    def optimise_min_variance(self,
                              covariance_matrix,
                              initial_guess,
                              bounds,
                              constraints
    ):

        result = minimize(self._min_variance_objective,
                          initial_guess,
                          args=(covariance_matrix,),
                          method="SLSQP",
                          bounds=bounds,
                          constraints=constraints
                          )

        return result
    
    def build_efficient_frontier(self, annual_returns, covariance_matrix, bounds, constraints, num_points=25):
        # Find the minimum variance portfolio
        mvp_res = self.optimise_min_variance(covariance_matrix, np.repeat(1/len(annual_returns), len(annual_returns)), bounds, constraints)
        min_return = self.analyser.portfolio_return(mvp_res.x, annual_returns)

        # Find the max return given constraints
        max_ret_res = minimize(
            lambda w: -self.analyser.portfolio_return(w, annual_returns),
            np.repeat(1/len(annual_returns), len(annual_returns)),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        max_return = self.analyser.portfolio_return(max_ret_res.x, annual_returns)

        # Create targets between the min and max
        target_returns = np.linspace(min_return, max_return, num_points)
        
        frontier_returns = []
        frontier_risks = []
        frontier_weights = []

        for target in target_returns:
            cons = constraints + [{'type': 'eq', 'fun': lambda w: self.analyser.portfolio_return(w, annual_returns) - target}]
            
            result = minimize(
                self._min_variance_objective,
                mvp_res.x,
                args=(covariance_matrix,),
                method="SLSQP",
                bounds=bounds,
                constraints=cons
            )

            if result.success:
                frontier_risks.append(self.analyser.portfolio_volatility(result.x, covariance_matrix))
                frontier_returns.append(target)
                frontier_weights.append(result.x)

        return frontier_risks, frontier_returns, frontier_weights
    
    def perform_static_optimisation(
        self,
        annual_returns,
        covariance_matrix,
        initial_guess,
        bounds,
        constraints,
        daily_returns,
        risk_free_rate,
        num_frontier_points=25
    ):

        results = {
            "mvp": None,
            "efficient_frontier_std_devs": [],
            "efficient_frontier_returns": []
        }

        # -------------------
        # Minimum variance
        # -------------------

        mvp_result = self.optimise_min_variance(
            covariance_matrix,
            initial_guess,
            bounds,
            constraints
        )

        weights = mvp_result.x

        metrics = self.analyser._calculate_portfolio_metrics_full(
            weights,
            annual_returns,
            daily_returns,
            covariance_matrix,
            risk_free_rate,
            None,
            None
        )

        results["mvp"] = {
            "weights": weights,
            "metrics": metrics,
            "success": mvp_result.success,
            "message": mvp_result.message
        }

        # -------------------
        # Efficient frontier
        # -------------------

        stds, rets, wghts = self.build_efficient_frontier(
            annual_returns,
            covariance_matrix,
            bounds,
            constraints,
            num_frontier_points
        )

        results["efficient_frontier_std_devs"] = stds
        results["efficient_frontier_returns"] = rets
        results["efficient_frontier_weights"] = wghts

        return results
    
    def optimise_max_sharpe_ratio(self, annual_returns, covariance_matrix, bounds, constraints, risk_free_rate):
        n = len(annual_returns)
        initial_guess = np.repeat(1/n, n)

        def neg_sharpe_ratio(weights):
            port_return = self.analyser.portfolio_return(weights, annual_returns)
            port_vol = self.analyser.portfolio_volatility(weights, covariance_matrix)
            return -(port_return - risk_free_rate) / port_vol

        result = minimize(
            neg_sharpe_ratio,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        return result
    
    def build_capital_market_line(self, bounds, constraints):
        # Find tangency portfolio
        res = self.optimise_max_sharpe_ratio(self.returns, self.cov, bounds, constraints, self.rf)
        if not res.success:
            raise ValueError("Tangency portfolio optimisation failed")

        weights_tangency = res.x
        ret_tangency = self.analyser.portfolio_return(weights_tangency, self.returns)
        vol_tangency = self.analyser.portfolio_volatility(weights_tangency, self.cov)

        # Generate multiples of the tangency portfolio risk
        max_multiplier = 2.0
        cml_vols = np.array([0, vol_tangency * max_multiplier])
        cml_rets = np.array([self.rf, self.rf + max_multiplier * (ret_tangency - self.rf)])

        return {
            "vols": cml_vols,
            "returns": cml_rets,
            "tangency_vol": vol_tangency,
            "tangency_ret": ret_tangency,
            "tangency_weights": weights_tangency
        }
    
    def perform_full_analysis(self, bounds, constraints):
        # Get basics
        static = self.perform_static_optimisation(
            self.returns, self.cov, 
            self.analyser.current_weights, 
            bounds, constraints, 
            self.analyser.returns, self.rf
        )
        
        # Get CML
        cml_data = self.build_capital_market_line(bounds, constraints)
        
        return {
            "optimisation": static,
            "cml": cml_data
        }
