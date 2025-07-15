import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# --- Final Visualization Script for Backtest Results ---

# --- 1. CONFIGURATION: Choose which ticker to plot ---
TICKER_TO_PLOT = 'AAPL' # CORRECTED: The ticker for Apple is 'AAPL'

print(f"Generating final backtest plots for {TICKER_TO_PLOT}...")

# --- 2. Load the Saved Backtest Curve Data ---
# CORRECTED: The directory is 'data/backtest_curves' as you specified.
curves_dir = 'data/backtest_curves'
curves_path = os.path.join(curves_dir, f'backtest_curves_{TICKER_TO_PLOT}.csv')

try:
    results_df = pd.read_csv(curves_path)
    results_df['Date'] = pd.to_datetime(results_df['Date'])
except FileNotFoundError:
    print(f"ERROR: Backtest curves file not found at {curves_path}.")
    print("Please run the final backtesting script first to generate this file.")
    exit()

# --- NEW: Load Champion ARIMA Order for accurate labeling ---
try:
    results_dir = 'data/tuning_results'
    champion_models_path = os.path.join(results_dir, 'champion_models.json')
    with open(champion_models_path, 'r') as f:
        champions_list = json.load(f)
    arima_params = next(item['best_params'] for item in champions_list if item['model_type'] == 'ARIMA')
    # Create a dynamic label based on the champion parameters
    champion_arima_label = f"ARIMA ({arima_params['p']},{arima_params['d']},{arima_params['q']}) Strategy"
except (FileNotFoundError, StopIteration):
    # Fallback label if the champion file isn't found
    champion_arima_label = "ARIMA Strategy"

# --- 3. Create the Plots ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(
    nrows=2, 
    ncols=1, 
    figsize=(14, 12), 
    sharex=True, 
    gridspec_kw={'height_ratios': [3, 1]} # Make the top plot taller
)
# Make the title dynamic based on the data's year
start_year = results_df['Date'].dt.year.min()
end_year = results_df['Date'].dt.year.max()
year_str = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"
fig.suptitle(f'Backtest Performance for {TICKER_TO_PLOT} ({year_str})', fontsize=20, weight='bold')

# --- Plot 1: Equity Curve (Cumulative Return) ---
ax1.set_title('Strategy Equity Curves (Cumulative Return)', fontsize=16)

# Calculate cumulative returns from log returns
results_df['prophet_equity'] = np.exp(np.cumsum(results_df['prophet_net_returns']))
results_df['arima_equity'] = np.exp(np.cumsum(results_df['arima_net_returns']))
results_df['buy_and_hold_equity'] = np.exp(np.cumsum(results_df['buy_and_hold_returns']))

# CORRECTED: Use the dynamic label for the ARIMA plot
ax1.plot(results_df['Date'], results_df['arima_equity'], label=champion_arima_label, lw=2.5, color='green')
ax1.plot(results_df['Date'], results_df['prophet_equity'], label='Champion Prophet Strategy', lw=2.5, color='blue')
ax1.plot(results_df['Date'], results_df['buy_and_hold_equity'], label='Buy and Hold Benchmark', lw=2, color='red', linestyle='--')

ax1.set_ylabel('Growth of $1')
ax1.legend(fontsize=12)
ax1.grid(True)


# --- Plot 2: Drawdown Chart ---
ax2.set_title('Strategy Drawdown (Risk)', fontsize=16)

# Calculate drawdown series for each strategy
arima_running_max = np.maximum.accumulate(results_df['arima_equity'])
arima_drawdown_series = (arima_running_max - results_df['arima_equity']) / arima_running_max

prophet_running_max = np.maximum.accumulate(results_df['prophet_equity'])
prophet_drawdown_series = (prophet_running_max - results_df['prophet_equity']) / prophet_running_max

ax2.plot(results_df['Date'], arima_drawdown_series, label=f"ARIMA Drawdown (Max: {arima_drawdown_series.max():.2%})", lw=2, color='green')
ax2.plot(results_df['Date'], prophet_drawdown_series, label=f"Prophet Drawdown (Max: {prophet_drawdown_series.max():.2%})", lw=2, color='blue')

ax2.set_ylabel('Drawdown')
ax2.set_xlabel('Date', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True)
# Format the y-axis ticks as percentages
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
