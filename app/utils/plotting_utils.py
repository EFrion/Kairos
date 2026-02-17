import matplotlib
matplotlib.use('Agg') 
# 'Agg' is a non-GUI backend designed for file output (like PNG, JPEG, etc.)

import numpy as np
import os
import statsmodels.api as sm
import plotly.graph_objects as go
import pandas as pd
from scipy import stats

def create_monthly_dividends_figure(stock_metrics, current_shares):
    """
    Interactive Plotly bar chart showing total monthly dividend payout.
    
    Args:
        stock_metrics (list): List of dictionaries with stock data.
        current_shares (dict): Dictionary of current share counts.
        
    Returns:
        str: HTML code for the Plotly graph object.
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_payout = [0.0] * 12
    hover_text = [[] for _ in range(12)] # List of lists to store stock details per month

    # Aggregate data and hover information
    for stock in stock_metrics:
        ticker = stock['Ticker']
        shares = current_shares.get(ticker, 0)
        
        if shares > 0:
            latest_div = stock.get('Latest_Div_EUR', 0.0)
            months_paid = stock.get('Months_Paid', [0] * 12)
            
            payout_amount = latest_div * shares
            
            for i in range(12):
                if months_paid[i] == 1:
                    monthly_payout[i] += payout_amount
                    
                    # Store data
                    hover_text[i].append(
                        f"{ticker}: €{payout_amount:.2f} ({shares} shares)"
                    )

    # Combine the list of stocks into a single string
    final_hover_text = []
    for i, amounts in enumerate(hover_text):
        if monthly_payout[i] > 0:
            # Join the list of stock details with line breaks for the tooltip
            details = "<br>".join(amounts)
            final_hover_text.append(f"<b>Total: €{monthly_payout[i]:.2f}</b><br><br>Contributing Stocks:<br>{details}")
        else:
            final_hover_text.append("No Payout")


    # Create Plotly figure
    fig = go.Figure(data=[
        go.Bar(
            x=months,
            y=monthly_payout,
            marker_color=['#28a745' if amount > 0 else '#cccccc' for amount in monthly_payout],
            hovertext=final_hover_text,
            hovertemplate='%{hovertext}<extra></extra>', # Custom hover text
            name="Monthly Dividend Payout"
        )
    ])

    # Update layout for a non-static look
    fig.update_layout(
        title='Total Expected Monthly Dividend Income (€)',
        xaxis_title='Month',
        yaxis_title='Expected Dividend Payout (€)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
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
    
def create_returns_distribution_chart(ticker_data):
    """
    Distribution plot for log returns.
    
    Input: DataFrame with one column of prices
    Output: Plotly figure (Histogram of log returns)
    """
    print("create_returns_distribution_chart called")
    
    prices = ticker_data.iloc[:, 0].astype(float) # Ensure numbers
    #print("prices: ", prices)
    
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    #ticker_name = log_returns.columns[0]
    
    # Clean data in case it's not done before (should be done in app.py though)
    data = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    #data = log_returns[ticker_name].values
    
    # If there's no data left, return a blank figure with a message
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for returns", showarrow=False)
        return fig
        
    #print("log_returns: ", log_returns)
    #print("ticker_name: ", ticker_name)
    #print("data: ", data)
    
    # Create histogram
    fig = go.Figure()
    data_min = data.min()
    data_max = data.max()
    #print("data_min: ", data_min)
    #print("data_max: ", data_max)
    #print("len data: ", len(data))
    
    fig.add_trace(go.Histogram(
        x=data.tolist(),
        name='Return Frequency',
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
    params = stats.t.fit(data) # Maximum Likelihood Estimation
    student_t = stats.t.pdf(x_range,*params)
    
    fig.add_trace(go.Scatter(
                x=x_range.tolist(),
                y=student_t.tolist(),
                mode='lines',
                name='Students t dist.',
                hovertemplate=f"<br>Student's t dist.: %{{y:.2f}}<extra></extra>"
            ))

    fig.update_layout(
        title=f"Log Returns Distribution",
        #title=f"Log Returns Distribution: {ticker_name}",
        xaxis_title="Daily Log Return",
        yaxis_title="Frequency",
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
