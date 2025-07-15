import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Read cleaned data for each ticker
tickers = ['AAPL', 'MSFT', 'AMZN', 'META', 'NVDA', 'GOOGL', 'TSLA']

for ticker in tickers:
    # Load data
    df = pd.read_csv(f'data/cleaned/{ticker}_cleaned.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)

    # Use 'Close' for ARIMA
    series = df['Close'].dropna()

    # Define ARIMA order (example: (1,1,1))
    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()

    # Save summary
    with open(f'scripts/arima_summary_{ticker}.txt', 'w') as f:
        f.write(model_fit.summary().as_text())

    # Forecast & compute RMSE (example: 30 days ahead)
    forecast = model_fit.forecast(steps=30)
    rmse = mean_squared_error(series[-30:], forecast, squared=False)

    print(f"ARIMA RMSE for {ticker}: {rmse:.4f}")

print("ARIMA models trained and results saved.")