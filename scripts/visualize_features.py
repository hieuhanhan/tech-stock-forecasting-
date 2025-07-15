import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/combined/combined_data_with_technical_indicators.csv', parse_dates=['Date'])

# --- Data Preparation for Visualization ---
# Set 'Date' as the DataFrame index for easier time series plotting
df = df.set_index('Date')

# Identify unique tickers in your dataset
tickers = df['Ticker'].unique()
print(f"Tickers found in dataset: {tickers}")

# Choose one ticker for visualization.
selected_ticker = 'AAPL'
df_ticker = df[df['Ticker'] == selected_ticker].copy()

# Drop rows with NaN values created by rolling windows for cleaner visualization.
df_ticker.dropna(inplace=True)

# Define the range for plotting
# Visualizing the entire 10 years at once can be very dense.
# It's often better to visualize a smaller, representative period (e.g., 2-3 years).
start_date = '2018-01-01'
end_date = '2020-12-31' 
df_plot = df_ticker.loc[start_date:end_date]

# Define the lag periods used in your feature engineering script
lag_periods = [1, 3, 7] 

# --- Visualization Functions ---

def plot_price_and_trend_indicators(data, ticker_name):
    """
    Plots the stock's Close price along with various Moving Averages and Bollinger Bands.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'{ticker_name} Stock Price & Trend/Volatility Indicators ({data.index.min().year}-{data.index.max().year})', fontsize=18)

    # Plot 1: Close Price and Moving Averages (SMA, EMA)
    axes[0].plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.8)
    axes[0].plot(data.index, data['SMA_20'], label='SMA 20', color='orange', linestyle='--')
    axes[0].plot(data.index, data['SMA_50'], label='SMA 50', color='red', linestyle='--')
    axes[0].plot(data.index, data['SMA_200'], label='SMA 200', color='purple', linestyle='--')
    axes[0].plot(data.index, data['EMA_12'], label='EMA 12', color='green', linestyle=':')
    axes[0].plot(data.index, data['EMA_26'], label='EMA 26', color='brown', linestyle=':')
    axes[0].set_ylabel('Price (Log Transformed)')
    axes[0].set_title('Close Price & Moving Averages (Trend Indicators)')
    axes[0].legend(loc='upper left', ncol=3)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Bollinger Bands
    axes[1].plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.7)
    axes[1].plot(data.index, data['SMA_20'], label='Bollinger Middle (SMA 20)', color='orange', linestyle='--')
    axes[1].plot(data.index, data['Bollinger_Upper'], label='Bollinger Upper', color='green', linestyle='-.')
    axes[1].plot(data.index, data['Bollinger_Lower'], label='Bollinger Lower', color='red', linestyle='-.')
    # Fill between the bands for better visualization of the price channel
    axes[1].fill_between(data.index, data['Bollinger_Lower'], data['Bollinger_Upper'], color='grey', alpha=0.1)
    axes[1].set_ylabel('Price (Log Transformed)')
    axes[1].set_title('Bollinger Bands (Volatility Indicator)')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.show()


def plot_momentum_and_volume_indicators(data, ticker_name):
    """
    Plots RSI and OBV, typically in separate subplots for clarity.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'{ticker_name} Momentum & Volume Indicators ({data.index.min().year}-{data.index.max().year})', fontsize=18)

    # Plot 1: RSI 14
    axes[0].plot(data.index, data['RSI_14'], label='RSI 14', color='teal')
    axes[0].axhline(70, color='red', linestyle=':', alpha=0.6, label='Overbought (70)')
    axes[0].axhline(30, color='green', linestyle=':', alpha=0.6, label='Oversold (30)')
    axes[0].set_ylabel('RSI Value')
    axes[0].set_title('RSI 14 (Momentum Indicator)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(0, 100) # RSI typically ranges from 0-100

    # Plot 2: On-Balance Volume (OBV)
    axes[1].plot(data.index, data['OBV'], label='OBV', color='darkviolet', alpha=0.8)
    axes[1].set_ylabel('OBV')
    axes[1].set_title('On-Balance Volume (Volume Indicator)')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def plot_log_returns_distribution(data, ticker_name):
    """
    Plots the histogram and Kernel Density Estimate (KDE) of Log Returns.
    """
    plt.figure(figsize=(12, 7))
    sns.histplot(data['Log_Returns'].dropna(), kde=True, color='skyblue', bins=50)
    plt.title(f'{ticker_name} - Distribution of Log Returns ({data.index.min().year}-{data.index.max().year})', fontsize=16)
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_lagged_features(data, ticker_name, feature_base_name):
    """
    Plots a base feature and its lagged versions side-by-side for comparison.
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Plot the current (non-lagged) feature
    ax.plot(data.index, data[feature_base_name], label=f'{feature_base_name} (Current)', color='blue', alpha=0.8)

    # Plot all defined lagged versions
    for lag in lag_periods:
        lagged_col_name = f'{feature_base_name}_lag_{lag}' 
        if lagged_col_name in data.columns: 
            ax.plot(data.index, data[lagged_col_name], label=f'{feature_base_name} (Lag {lag})', linestyle='--', alpha=0.6)
        else:
            print(f"Warning: Lagged column '{lagged_col_name}' not found for {feature_base_name}. Skipping plot.")


    ax.set_title(f'{ticker_name} - {feature_base_name} and Lagged Versions ({data.index.min().year}-{data.index.max().year})', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel(feature_base_name)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_feature_correlation_heatmap(data, ticker_name):
    """
    Generates a heatmap of correlations between all numerical features.
    """
    # Select only numerical columns for correlation calculation
    # Drop 'Ticker' as it's a categorical identifier here, not a numeric feature
    df_numeric = data.select_dtypes(include=np.number).drop(columns=['Ticker'], errors='ignore')
    
    # Drop rows with NaN values created by rolling windows/lags for correlation calculation
    # Correlation matrix cannot be computed with NaNs.
    df_numeric.dropna(inplace=True)

    plt.figure(figsize=(18, 16))
    # Using 'coolwarm' for diverging colormap, good for correlations from -1 to 1
    # annot=False to avoid cluttering with numbers if many features
    sns.heatmap(df_numeric.corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'{ticker_name} - Feature Correlation Heatmap', fontsize=20)
    plt.show()


# --- Execute Visualizations ---

print(f"\nStarting visualizations for: {selected_ticker} from {start_date} to {end_date}\n")

# 1. Visualize Price and Trend/Volatility Indicators
plot_price_and_trend_indicators(df_plot, selected_ticker)

# 2. Visualize Momentum and Volume Indicators
plot_momentum_and_volume_indicators(df_plot, selected_ticker)

# 3. Visualize Log Returns Distribution
plot_log_returns_distribution(df_plot, selected_ticker)

# 4. Visualize Lagged Features
plot_lagged_features(df_plot, selected_ticker, feature_base_name='Close')
plot_lagged_features(df_plot, selected_ticker, feature_base_name='RSI_14')
plot_lagged_features(df_plot, selected_ticker, feature_base_name='OBV')

# 5. Visualize Overall Feature Correlation Heatmap (using the full ticker data, not just the plot subset)
plot_feature_correlation_heatmap(df_ticker, selected_ticker)


print("\nAll visualizations complete. Remember to change 'selected_ticker' and 'start_date'/'end_date' variables in the script to explore other periods or stocks.")
print("The `dropna()` calls in visualization functions are for plotting clarity and might be different from your actual model's NaN handling.")