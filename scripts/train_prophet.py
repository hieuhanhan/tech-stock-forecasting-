import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error

tickers = ['AAPL', 'MSFT', 'AMZN', 'META', 'NVDA', 'GOOGL', 'TSLA']

for ticker in tickers:
    df = pd.read_csv(f'data/cleaned/{ticker}_cleaned.csv', parse_dates=['Date'])
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df[['ds', 'y']])

    # Future dataframe & forecast
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Save forecast
    forecast.to_csv(f'scripts/prophet_forecast_{ticker}.csv', index=False)

    # Compute RMSE on last 30 days
    actual = df['y'].iloc[-30:]
    pred = forecast['yhat'].iloc[-30:]
    rmse = mean_squared_error(actual, pred, squared=False)

    print(f"Prophet RMSE for {ticker}: {rmse:.4f}")

print("Prophet models trained and forecasts saved.")