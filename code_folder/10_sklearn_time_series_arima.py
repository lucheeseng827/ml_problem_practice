"""
Time Series Forecasting with ARIMA/SARIMA
==========================================
Category 10: Time Series Forecasting

This example demonstrates:
- ARIMA (AutoRegressive Integrated Moving Average) modeling
- SARIMA (Seasonal ARIMA) for seasonal patterns
- Forecasting future values
- Model evaluation with time series metrics

Use cases:
- Stock price prediction
- Sales forecasting
- Demand planning
- Weather forecasting
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def generate_time_series_data(n_points=365, seed=42):
    """Generate synthetic time series data with trend and seasonality"""
    np.random.seed(seed)

    dates = pd.date_range(start='2022-01-01', periods=n_points, freq='D')

    # Trend component
    trend = np.linspace(100, 200, n_points)

    # Seasonal component (yearly seasonality)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365)

    # Random noise
    noise = np.random.normal(0, 5, n_points)

    # Combine components
    values = trend + seasonal + noise

    df = pd.DataFrame({'date': dates, 'value': values})
    df.set_index('date', inplace=True)

    return df


def arima_forecast_example():
    """ARIMA model for time series forecasting"""
    print("=" * 60)
    print("ARIMA Time Series Forecasting")
    print("=" * 60)

    # Generate data
    df = generate_time_series_data(n_points=365)

    # Train/test split (80/20)
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

    # Fit ARIMA model
    # Order (p, d, q): p=autoregressive, d=differencing, q=moving average
    print("\nFitting ARIMA(5,1,2) model...")
    model = ARIMA(train['value'], order=(5, 1, 2))
    fitted_model = model.fit()

    print(f"\nModel summary:")
    print(fitted_model.summary())

    # Forecast
    forecast_steps = len(test)
    forecast = fitted_model.forecast(steps=forecast_steps)

    # Evaluate
    mse = mean_squared_error(test['value'], forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test['value'], forecast)

    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['value'], label='Training Data', color='blue')
    plt.plot(test.index, test['value'], label='Actual Test Data', color='green')
    plt.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('ARIMA Time Series Forecast')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/tmp/arima_forecast.png')
    print("\nPlot saved to /tmp/arima_forecast.png")

    return fitted_model, forecast


def sarima_forecast_example():
    """SARIMA model with seasonal components"""
    print("\n" + "=" * 60)
    print("SARIMA Time Series Forecasting (Seasonal)")
    print("=" * 60)

    # Generate data with strong seasonality
    df = generate_time_series_data(n_points=730)  # 2 years

    # Train/test split
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

    # Seasonal decomposition
    print("\nPerforming seasonal decomposition...")
    decomposition = seasonal_decompose(train['value'], model='additive', period=365)

    # Fit SARIMA model
    # Order (p,d,q) x (P,D,Q,s)
    # s = seasonal period (365 for yearly)
    print("\nFitting SARIMA(1,1,1)x(1,1,1,365) model...")
    # Note: Using smaller seasonal period for faster computation
    model = SARIMAX(
        train['value'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 30),  # Using 30 days for demo
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)

    print(f"\nModel AIC: {fitted_model.aic:.2f}")
    print(f"Model BIC: {fitted_model.bic:.2f}")

    # Forecast
    forecast = fitted_model.forecast(steps=len(test))

    # Evaluate
    rmse = np.sqrt(mean_squared_error(test['value'], forecast))
    mae = mean_absolute_error(test['value'], forecast)

    print(f"\nSARIMA Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Plot seasonal decomposition
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.savefig('/tmp/seasonal_decomposition.png')
    print("\nSeasonal decomposition saved to /tmp/seasonal_decomposition.png")

    return fitted_model, forecast


def auto_arima_example():
    """Automatic ARIMA order selection"""
    print("\n" + "=" * 60)
    print("Auto ARIMA - Automatic Order Selection")
    print("=" * 60)

    # Note: pmdarima library provides auto_arima functionality
    # For this example, we'll demonstrate manual grid search

    df = generate_time_series_data(n_points=365)
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    best_aic = np.inf
    best_order = None
    best_model = None

    # Grid search over p, d, q
    print("\nSearching for best ARIMA order...")
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(train['value'], order=(p, d, q))
                    fitted = model.fit()

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted

                except:
                    continue

    print(f"\nBest ARIMA order: {best_order}")
    print(f"Best AIC: {best_aic:.2f}")

    # Forecast with best model
    forecast = best_model.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test['value'], forecast))

    print(f"RMSE with best model: {rmse:.2f}")

    return best_model


def main():
    """Main execution function"""
    print("Time Series Forecasting with ARIMA/SARIMA\n")

    # Example 1: Basic ARIMA
    arima_model, arima_forecast = arima_forecast_example()

    # Example 2: SARIMA with seasonality
    sarima_model, sarima_forecast = sarima_forecast_example()

    # Example 3: Automatic order selection
    auto_model = auto_arima_example()

    print("\n" + "=" * 60)
    print("Time Series Forecasting Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- ARIMA is suitable for non-seasonal time series")
    print("- SARIMA handles seasonal patterns effectively")
    print("- Model selection (p,d,q) impacts forecast accuracy")
    print("- Seasonal decomposition helps understand data structure")
    print("- Always validate on hold-out test set")


if __name__ == "__main__":
    main()
