import matplotlib
matplotlib.use('Agg') 
# 'Agg' is a non-GUI backend designed for file output (like PNG, JPEG, etc.)

import numpy as np
import os
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy import stats

def create_trends_chart(interest_over_time_df, terms=None, start_date=None, end_date=None, rolling_windows=None):
    """
    Line chart showing Google Trends interest over time for multiple terms.

    Args:
        interest_over_time_df (pd.DataFrame): DataFrame with 'Date' index and terms as columns.
        terms (list): List of terms to plot. If None, plot all columns.
        start_date (str/datetime): Start date for filtering (e.g., '2024-01-01').
        end_date (str/datetime): End date for filtering.
        rolling_windows (list): List of integers for rolling means, e.g., [7, 30].

    Returns:
        go.Figure: The Plotly figure object.
    """

    df = interest_over_time_df.copy()

    # Filter terms if specified
    if terms:
        df = df[terms]

    # Filter data based on dates
    if start_date:
        start_ts = pd.to_datetime(start_date)
        if df.index.tz:
            start_ts = start_ts.tz_localize(df.index.tz)
        df = df[df.index >= start_ts]

    if end_date:
        end_ts = pd.to_datetime(end_date)
        if df.index.tz:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts]

    # Create the figure
    fig = go.Figure()

    # Add a line for each term
    for term in df.columns:
        term_series = df[term].dropna()

        if not term_series.empty:
            fig.add_trace(go.Scatter(
                x=term_series.index.tolist(),
                y=term_series.values.tolist(),
                mode='lines',
                name=term,
                hovertemplate=f"<b>{term}</b><br>Trends Index: %{{y:.2f}}<extra></extra>"
            ))

        # Add rolling averages if specified
        if rolling_windows:
            for window in rolling_windows:
                rolling_mean = term_series.rolling(window=window).mean()
                fig.add_trace(go.Scatter(
                    x=rolling_mean.index.tolist(),
                    y=rolling_mean.values.tolist(),
                    mode='lines',
                    name=f"{term} ({window}d SMA)",
                    line=dict(dash='dash', width=1.5),
                    hovertemplate=f"{term} {window}d SMA: %{{y:.2f}}<extra></extra>"
                ))

    # Update layout
    fig.update_layout(
        title='Interest over time',
        xaxis_title='Date',
        yaxis_title='Google Trends index',
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        ),
        yaxis=dict(
            autorange=True,
            title='Trends Index'
        )
    )

    return fig

def plot_efficient_frontier_and_portfolios(
    results,
    asset_analysers
):
    """
    Plots the efficient frontier, Monte Carlo simulations, individual stocks,
    and optimised portfolios (MVP, Sharpe, Sortino, MVSK).

    Args:
        static_results (dict): Results from static optimisation.
        dynamic_results (dict): Results from dynamic optimisation.
        individual_stock_metrics (list): List of dictionaries with individual stock metrics.
        portfolio_tickers (list): List of ticker symbols for the assets.
        static_portfolio_points_raw_mc (list): List of dictionaries for static Monte Carlo portfolios.
        dynamic_portfolio_points_raw_mc (list): List of dictionaries for dynamic Monte Carlo portfolios.
        output_dir (str): The directory to save the plot.
        feature_toggles (dict): Dictionary of feature toggles.
        num_assets (int): Number of assets in the portfolio.
    """
    fig = go.Figure()
    tickers = list(asset_analysers.keys())
    
#    RUN_STATIC_PORTFOLIO = feature_toggles['RUN_STATIC_PORTFOLIO']
    RUN_STATIC_PORTFOLIO = True
#    RUN_DYNAMIC_PORTFOLIO = feature_toggles['RUN_DYNAMIC_PORTFOLIO']
#    RUN_EQUAL_WEIGHTED_PORTFOLIO = feature_toggles['RUN_EQUAL_WEIGHTED_PORTFOLIO']
#    RUN_MONTE_CARLO_SIMULATION = feature_toggles['RUN_MONTE_CARLO_SIMULATION']
#    RUN_MVO_OPTIMISATION = feature_toggles['RUN_MVO_OPTIMISATION']
    RUN_MVO_OPTIMISATION = True
#    RUN_SHARPE_OPTIMISATION = feature_toggles['RUN_SHARPE_OPTIMISATION']
#    RUN_SORTINO_OPTIMISATION = feature_toggles['RUN_SORTINO_OPTIMISATION']
#    RUN_MVSK_OPTIMISATION = feature_toggles['RUN_MVSK_OPTIMISATION']

#    plt.figure(figsize=(14, 8)) # Larger figure for more elements

#    # Plot all Monte-Carlo-simulated portfolio combinations (lighter color, background)
#    if RUN_MONTE_CARLO_SIMULATION:
#        if RUN_STATIC_PORTFOLIO and static_portfolio_points_raw_mc:
#            plt.scatter([p['std_dev'] * 100 for p in static_portfolio_points_raw_mc],
#                        [p['return'] * 100 for p in static_portfolio_points_raw_mc],
#                        color='blue', marker='o', s=10, alpha=0.5, # More transparent
#                        label='Monte Carlo portfolio combinations (Static)')
#        if RUN_DYNAMIC_PORTFOLIO and dynamic_portfolio_points_raw_mc and dynamic_results['dynamic_covariance_available']:
#            plt.scatter([p['std_dev'] * 100 for p in dynamic_portfolio_points_raw_mc],
#                        [p['return'] * 100 for p in dynamic_portfolio_points_raw_mc],
#                        color='red', marker='o', s=10, alpha=0.5, # More transparent
#                        label='Monte Carlo portfolio combinations (Dynamic)')
    
    # Plot the Efficient Frontier line (Static Covariance)
#    if RUN_STATIC_PORTFOLIO and RUN_MVO_OPTIMISATION and num_assets > 20 and static_results['mvp'] and static_results['efficient_frontier_std_devs']:
#        plt.plot([s * 100 for s in static_results['efficient_frontier_std_devs']],
#                 [r * 100 for r in static_results['efficient_frontier_returns']],
#                 color='blue', linestyle='-', linewidth=2, label='Efficient frontier (Static)')

    # Extract sub-dictionaries for readability
    opt_res = results.get("optimisation")
    cml_res = results.get("cml")

    if RUN_STATIC_PORTFOLIO and RUN_MVO_OPTIMISATION and opt_res and opt_res['efficient_frontier_std_devs']:
        # Create a list of hover strings for each point
        hover_texts = []
        for w_set in opt_res.get('efficient_frontier_weights', []):
            # Only show assets with a weight > 0.5% to keep it clean
            weight_str = "<br>".join([
                f"{tickers[i]}: {w*100:.1f}%" 
                for i, w in enumerate(w_set) if w > 0.005
            ])
            hover_texts.append(weight_str)
        
        fig.add_trace(go.Scatter(
            x=[s * 100 for s in opt_res['efficient_frontier_std_devs']],
            y=[r * 100 for r in opt_res['efficient_frontier_returns']],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Efficient frontier (Static)',
            customdata=hover_texts,
            hovertemplate=(
                "<b>Efficient Portfolio</b><br>" +
                "Volatility: %{x:.2f}%<br>" +
                "Return: %{y:.2f}%<br>" +
                "------------------<br>" +
                "%{customdata}%<extra></extra>"
            )
        ))

#    # Plot the Efficient Frontier line (Dynamic Covariance)
#    if RUN_DYNAMIC_PORTFOLIO and RUN_MVO_OPTIMISATION and num_assets > 20 and dynamic_results['mvp'] and dynamic_results['efficient_frontier_std_devs'] and dynamic_results['dynamic_covariance_available']:
#        plt.plot([s * 100 for s in dynamic_results['efficient_frontier_std_devs']],
#                 [r * 100 for r in dynamic_results['efficient_frontier_returns']],
#                 color='red', linestyle='-', linewidth=2, label='Efficient frontier (Dynamic)')

    # Plot CML
        fig.add_trace(go.Scatter(
            x=[v * 100 for v in cml_res['vols']],
            y=[r * 100 for r in cml_res['returns']],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='CML'
        ))

        # Mark the tangency portfolio point
        t_vol = cml_res['tangency_vol']
        t_ret = cml_res['tangency_ret']
        t_weights = cml_res['tangency_weights']

        t_hover = "<br>".join([
            f"{tickers[i]}: {w*100:.1f}%" 
            for i, w in enumerate(t_weights) if w > 0.005
        ])

        fig.add_trace(go.Scatter(
            x=[t_vol * 100],
            y=[t_ret * 100],
            mode='markers',
            marker=dict(color='red', size=10, symbol='star'),
            name='Tangency portfolio',
            customdata=[t_hover],
            hovertemplate=(
                "<b>Tangency portfolio</b><br>" +
                "Volatility: %{x:.2f}%<br>" +
                "Return: %{y:.2f}%<extra></extra>"+
                "------------------<br>" +
                "%{customdata}<extra></extra>"
            )
        ))

    # Plot individual assets in the return/std space
    colors = px.colors.qualitative.Dark24  # or any palette you prefer
    all_vols = []
    all_rets = []

    for i, (ticker, analyser) in enumerate(asset_analysers.items()):
        ret = analyser.annual_return
        vol = analyser.annualised_volatility
        all_vols.append(vol)
        all_rets.append(ret)

        fig.add_trace(go.Scatter(
            x=[vol * 100],
            y=[ret * 100],
            mode='markers+text',
            marker=dict(size=12, color=colors[i % len(colors)], line=dict(width=1.5, color='black')),
            text=[ticker],
            textposition='top center',
            name='',
            showlegend=False
        ))


    if RUN_STATIC_PORTFOLIO:
#        # Plot the EWP (Static)
#        if RUN_EQUAL_WEIGHTED_PORTFOLIO and static_results['ewp'] and static_results['ewp']['success']:
#            plt.scatter(static_results['ewp']['Volatility'] * 100, static_results['ewp']['Return'] * 100,
#                        marker='p', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
#                        label=f"EWP (Static), Sharpe ratio={static_results['ewp']['Sharpe Ratio']:.2}")
                        
        # Plot the MVP (Static)
        if RUN_MVO_OPTIMISATION and opt_res['mvp'] and opt_res['mvp']['success']:
            mvp_metrics = opt_res['mvp']['metrics']
            fig.add_trace(go.Scatter(
                x=[mvp_metrics['Volatility'] * 100],
                y=[mvp_metrics['Return'] * 100],
                mode='markers',
                marker=dict(size=16, symbol='star', color='darkblue', opacity=0.7, line=dict(width=1.5, color='darkblue')),
                name='MV portfolio'
            ))
            
    max_v = max(all_vols) * 110 if all_vols else 30
    min_r = min(all_rets) * 90 if all_rets else 0
    max_r = max(all_rets) * 110 if all_rets else 20

    fig.update_layout(
        xaxis=dict(title="Annualised Volatility (%)", range=[0, max_v]),
        yaxis=dict(title="Annualised Return (%)", range=[min_r, max_r]),
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def create_income_plot(income_data, title="Expected monthly income"):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create Plotly figure
    fig = go.Figure(data=[
        go.Bar(
            x=months,
            y=income_data['payouts'],
            marker_color=['#28a745' if amount > 0 else '#cccccc' for amount in income_data['payouts']],
            hovertext=income_data['details'],
            hovertemplate='%{hovertext}<extra></extra>', # <extra></extra> removes the secondary 'trace' box
            name=title
        )
    ])

    # Update layout for a non-static look
    fig.update_layout(
        title=title + ' (€)',
        xaxis_title='Month',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# Plot correlation matrix between assets
def plot_correlation_heatmap(correlation_matrix):
    """
    Plot a heatmap of the correlation matrix.

    Args:
        correlation_matrix (pd.DataFrame): The correlation matrix of stock returns.
    """
    print("plot_correlation_heatmap called")
    
#    print("correlation_matrix.values.tolist: " , correlation_matrix.values.tolist())
#    print("correlation_matrix.columns.tolist: " , correlation_matrix.columns.tolist())
#    print("correlation_matrix.index.tolist: " , correlation_matrix.index.tolist())

    matrix_values = correlation_matrix.round(3).values.tolist()
    tickers = correlation_matrix.columns.tolist()
    #num_assets = len(tickers)
    
    #dynamic_size = min(500, (num_assets * 35) + 150)
    
    heatmap = go.Heatmap(
        z=matrix_values,
        x=tickers,
        y=tickers,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in matrix_values],
        hovertemplate='x: %{x}<br>y: %{y}<br>Correlation: %{z}<extra></extra>',
        colorbar=dict(title='Correlation')
    )
    
    fig = go.Figure(data=[heatmap])
    
#    fig.update_layout(
#        autosize=True,
#        template="plotly_white",
#         margin=dict(l=40, r=40, t=10, b=40), 
#        yaxis=dict(
#            autorange='reversed',
#            #scaleanchor="x",
#            #scaleratio=1,
#            domain=[0, 1],  # full vertical space
#            side='left',
#            tickangle=0,
#            ticks='outside',
#            showline=True,
#            linewidth=1,
#            linecolor='black',
#            mirror=True,
#            automargin=True
#        ),
#        xaxis=dict(
#            tickangle=45,
#            side='bottom',
#            ticks='outside',
#            showline=True,
#            linewidth=1,
#            linecolor='black',
#            mirror=True,
#            automargin=True
#        ),
#        coloraxis_colorbar=dict(
#            thickness=20,      # thinner legend bar
#            len=0.8,          # height relative to plot (80%)
#            y=0.5,            # center vertically
#            yanchor='middle',
#            x=1.02,           # push closer to heatmap (default is ~1.05)
#            ticks='outside',
#            outlinewidth=1,
#            outlinecolor='black'
#        )
#    )
#    
#    fig.update_xaxes(tickson='boundaries')
#    fig.update_yaxes(tickson='boundaries')

    fig.update_layout(
        autosize=True,
        template="plotly_white",
        margin=dict(l=40, r=40, t=10, b=40),
        coloraxis_colorbar=dict(
            thickness=20,
            len=0.8,
            y=0.5,
            yanchor='middle',
            x=1,  # right at edge of plot domain
            ticks='outside',
            outlinewidth=1,
            outlinecolor='black',
        )
    )

    fig.update_xaxes(
        type='category',
        tickson='boundaries',
        constrain='domain',
        ticks='outside',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        tickangle=45,
        automargin=True
    )

    fig.update_yaxes(
        type='category',
        tickson='boundaries',
        constrain='domain',
        autorange='reversed',
        ticks='outside',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        automargin=True
    )


    print("plot_correlation_heatmap out ")
    
    return fig
    
    
def create_2d_correlation_map(stocks_data_ticker1, stocks_data_ticker2):
    """
    Correlation stock vs stock.
    """
    # TODO extend that to other assets
    stocks1 = stocks_data_ticker1.copy()
    stocks2 = stocks_data_ticker2.copy()
    
    # Get name
    name1 = stocks1.name if hasattr(stocks1, 'name') else stocks1.columns[0]
    name2 = stocks2.name if hasattr(stocks2, 'name') else stocks2.columns[0]
    
    # Compute log returns
    returns1 = np.log(stocks1 / stocks1.shift(1))
    returns2 = np.log(stocks2 / stocks2.shift(1))
    
    # Join them into a single DataFrame
    map2d = pd.concat([returns1, returns2], axis=1, join='inner').dropna()
    
    if map2d.empty:
        return go.Figure().add_annotation(text="No overlapping data", showarrow=False)
        
    # Extract values
    x_data = map2d.iloc[:, 0].values
    y_data = map2d.iloc[:, 1].values
    
    fig = go.Figure()

    # Add Scatter Points
    fig.add_trace(go.Scatter(
        x=x_data.tolist(), # Convert to list to avoid conversion into binary data 
        y=y_data.tolist(),
        mode='markers',
        name='Daily Returns',
        marker=dict(color='rgba(0, 123, 255, 0.6)', size=8),
        hovertemplate=f"{name1}: %{{x:.4f}}<br>{name2}: %{{y:.4f}}<extra></extra>"
    ))
    
    # Add regression trendline
    try:
        X_reg = sm.add_constant(x_data)
        model = sm.OLS(y_data, X_reg).fit() # Ordinary Least Squares
        
        # Get statistics
        r_squared = model.rsquared
        
        # Create a smooth line for the trend
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        X_range_reg = sm.add_constant(x_range)
        predictions = model.get_prediction(X_range_reg)
        
        frame = predictions.summary_frame(alpha=0.05) # 95% confidence interval
        y_mean = frame['mean']
        y_upper = frame['mean_ci_upper']
        y_lower = frame['mean_ci_lower']
        
        fig.add_trace(go.Scatter(
            x=x_range.tolist() + x_range.tolist()[::-1],
            y=y_upper.tolist() + y_lower.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence'
        ))
        
        # Plot the regression line
        fig.add_trace(go.Scatter(
            x=x_range.tolist(),
            y=y_mean.tolist(),
            mode='lines',
            name='OLS Trendline',
            line=dict(color='red', width=2)
        ))
        
        # Show the regression coefficients
        corr_coef = map2d.iloc[:, 0].corr(map2d.iloc[:, 1])
        fig.update_layout(title=f"Correlation: {name1} vs {name2} (Pearson r = {corr_coef:.3f} | R² = {r_squared:.3f})")
    except Exception as e:
        print(f"Regression error: {e}")

    # Add the vertical and horizontal crosshair lines (at 0,0)
    fig.add_vline(x=0, line_dash="dash", line_color="grey", line_width=1)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)

    fig.update_layout(
        template="plotly_white",
        xaxis_title=f"{name1} Returns",
        yaxis_title=f"{name2} Returns",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig

def create_price_chart(stocks_data, start_date=None, end_date=None, rolling_windows=None):
    """
    Line chart showing price history for multiple assets.
    
    Args:
        stocks_data (pd.DataFrame): DataFrame with 'Date' index and tickers as columns.
        start_date (str/datetime): Start date for filtering (e.g., '2024-01-01').
        end_date (str/datetime): End date for filtering.
        rolling_windows (list): List of integers for rolling means, e.g., [50, 200].
    Returns:
        go.Figure: The Plotly figure object.
    """

    df = stocks_data.copy()
    
    # Filter data based on dates
    if start_date:
        start_ts = pd.to_datetime(start_date) # Ensure timezone compatibility if the DataFrame is localized to UTC
        if df.index.tz:
            start_ts = start_ts.tz_localize(df.index.tz)
        df = df[df.index >= start_ts]
        
    if end_date:
        end_ts = pd.to_datetime(end_date)
        if df.index.tz:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts]

    # Create the figure
    fig = go.Figure()

    # Add a line for each ticker
    for ticker in df.columns:
        ticker_series = df[ticker].dropna()
        
        if not ticker_series.empty:
            fig.add_trace(go.Scatter(
                x=ticker_series.index.tolist(),        # The filtered dates
                y=ticker_series.values.tolist(),       # The filtered prices
                mode='lines',
                name=ticker,
                hovertemplate=f"<b>{ticker}</b><br>Price: %{{y:.2f}}<extra></extra>"
            ))
            
        # Add rolling averages (optional)
        if rolling_windows:
            for window in rolling_windows:
                rolling_mean = ticker_series.rolling(window=window).mean()
                
                fig.add_trace(go.Scatter(
                    x=rolling_mean.index.tolist(),
                    y=rolling_mean.values.tolist(),
                    mode='lines',
                    name=f"{ticker} ({window}d SMA)",
                    line=dict(dash='dash', width=1.5), # Dashed line for distinction
                    hovertemplate=f"{ticker} {window}d SMA: %{{y:.2f}}<extra></extra>"
                ))

    # Update layout
    fig.update_layout(
        title='Historical Asset Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode="x unified",  # Shows all asset prices for a single date on hover
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=True), # Adds the bottom slide bar
            type='date'
        ),
        yaxis=dict(
            type='log',  # Changes the scale to Logarithmic
            autorange=True,
            title='Price - Log Scale'
        )
    )
    
    return fig
    
def create_returns_distribution_chart(returns, student_t_params=None):
    """
    Distribution plot for returns.
    
    Input: DataFrame with one column of prices
    Output: Plotly figure (Histogram of returns)
    """
    print("create_returns_distribution_chart called")
    
    # Clean data in case it's not done before
    data = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    # If DataFrame, convert to Series by selecting first column
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    
    # If there's no data left, return a blank figure with a message
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for returns", showarrow=False)
        return fig
        
    #print("data: ", data)
    
    # Create histogram
    fig = go.Figure()
    # data_min = data.min()
    # data_max = data.max()
    #print("data_min: ", data_min)
    #print("data_max: ", data_max)
    #print("len data: ", len(data))
    
    fig.add_trace(go.Histogram(
        x=data.tolist(),
        name='Return density',
        histnorm='probability density',
        marker=dict(
            color='#007BFF',
            line=dict(color='white', width=0.5) # Outline ensures visibility
        ),
        opacity=0.75,
        hovertemplate='Return: %{x:.2%}<br>Density: %{y}<extra></extra>'
    ))

    # Add vertical line for mean return
    mean_return = np.mean(data)
    fig.add_vline(x=mean_return, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_return:.2%}")

    # Add normal fit
    x_range = np.linspace(data.min(), data.max(), 100)
    y_pdf = stats.norm.pdf(x_range, loc=data.mean(), scale=data.std())

    fig.add_trace(go.Scatter(
                x=x_range.tolist(),
                y=y_pdf.tolist(),
                mode='lines',
                name='Normal dist.',
                hovertemplate=f"<br>Normal dist.: %{{y:.2f}}<extra></extra>"
            ))
            
    # Add a Student's t fit
    #params = stats.t.fit(data) # Maximum Likelihood Estimation
    if student_t_params is None:
        student_t_params = stats.t.fit(data)

    student_t = stats.t.pdf(x_range,*student_t_params)
    
    fig.add_trace(go.Scatter(
                x=x_range.tolist(),
                y=student_t.tolist(),
                mode='lines',
                name='Students t dist.',
                hovertemplate=f"<br>Student's t dist.: %{{y:.2f}}<extra></extra>"
            ))

    fig.update_layout(
        xaxis_title="Daily returns",
        yaxis_title="Density",
        template="plotly_white",
        bargap=0.05,
        xaxis=dict(tickformat=".2%")
    )
    
    print("create_returns_distribution_chart out")
    
    return fig


if __name__ == '__main__':

    simulated_stock_metrics = [
        {"Ticker": "StockA", "Months_Paid": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]},
        {"Ticker": "StockB", "Months_Paid": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
        {"Ticker": "StockC", "Months_Paid": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
        {"Ticker": "StockD", "Months_Paid": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]},
    ]
    plot_file = create_monthly_dividends_plot(simulated_stock_metrics)
    print(f"Plot saved to: {plot_file}")
