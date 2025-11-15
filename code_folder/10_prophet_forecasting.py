"""
Time Series Forecasting with Facebook Prophet
==============================================
Category 10: Time Series Forecasting

This example demonstrates:
- Prophet for automated time series forecasting
- Handling seasonality (daily, weekly, yearly)
- Holiday effects and special events
- Change point detection
- Uncertainty intervals

Use cases:
- Business forecasting
- Sales predictions
- Website traffic forecasting
- Capacity planning
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


def generate_business_data(n_days=730):
    """Generate synthetic business time series with multiple seasonalities"""
    np.random.seed(42)

    # Create date range
    dates = pd.date_range(start='2021-01-01', periods=n_days, freq='D')

    # Base trend
    trend = np.linspace(100, 300, n_days)

    # Weekly seasonality (higher on weekends)
    weekly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)

    # Yearly seasonality
    yearly = 30 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)

    # Holiday effects (boosted sales around specific dates)
    holiday_effect = np.zeros(n_days)
    for i in range(n_days):
        # Boost around day 355 (end of year)
        if 350 <= (i % 365) <= 365:
            holiday_effect[i] = 50

    # Random noise
    noise = np.random.normal(0, 10, n_days)

    # Combine all components
    values = trend + weekly + yearly + holiday_effect + noise

    df = pd.DataFrame({
        'ds': dates,  # Prophet requires 'ds' column
        'y': values   # Prophet requires 'y' column
    })

    return df


def basic_prophet_forecast():
    """Basic Prophet forecasting example"""
    print("=" * 60)
    print("Basic Prophet Forecasting")
    print("=" * 60)

    # Generate data
    df = generate_business_data(n_days=730)

    # Split into train and test
    train_size = int(len(df) * 0.8)
    train = df[:train_size]
    test = df[train_size:]

    print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

    # Initialize and fit Prophet model
    print("\nFitting Prophet model...")
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive'
    )

    model.fit(train)

    # Create future dataframe for predictions
    future = model.make_future_dataframe(periods=len(test), freq='D')

    # Make predictions
    forecast = model.predict(future)

    # Extract test predictions
    test_forecast = forecast.iloc[train_size:]

    # Evaluate
    y_true = test['y'].values
    y_pred = test_forecast['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\nTest Set Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Plot forecast
    fig1 = model.plot(forecast)
    plt.title('Prophet Forecast')
    plt.tight_layout()
    plt.savefig('/tmp/prophet_forecast.png')
    print("\nForecast plot saved to /tmp/prophet_forecast.png")

    # Plot components
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('/tmp/prophet_components.png')
    print("Components plot saved to /tmp/prophet_components.png")

    return model, forecast


def prophet_with_holidays():
    """Prophet with custom holiday effects"""
    print("\n" + "=" * 60)
    print("Prophet with Holiday Effects")
    print("=" * 60)

    # Generate data
    df = generate_business_data(n_days=730)

    # Define custom holidays
    holidays = pd.DataFrame({
        'holiday': 'year_end_sale',
        'ds': pd.to_datetime(['2021-12-20', '2022-12-20']),
        'lower_window': -5,
        'upper_window': 5,
    })

    # Additional holidays
    black_friday = pd.DataFrame({
        'holiday': 'black_friday',
        'ds': pd.to_datetime(['2021-11-26', '2022-11-25']),
        'lower_window': 0,
        'upper_window': 3,
    })

    holidays = pd.concat([holidays, black_friday])

    print(f"\nDefined {len(holidays)} holiday events")

    # Model with holidays
    model = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    # Split data
    train_size = int(len(df) * 0.8)
    train = df[:train_size]

    model.fit(train)

    # Forecast
    future = model.make_future_dataframe(periods=180, freq='D')
    forecast = model.predict(future)

    print("\nHoliday effects incorporated into forecast")
    print("Prophet automatically models recurring and one-time events")

    # Plot
    fig = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('/tmp/prophet_holidays.png')
    print("Holiday components saved to /tmp/prophet_holidays.png")

    return model, forecast


def prophet_custom_seasonality():
    """Add custom seasonality patterns"""
    print("\n" + "=" * 60)
    print("Prophet with Custom Seasonality")
    print("=" * 60)

    df = generate_business_data(n_days=730)

    # Initialize model
    model = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=False
    )

    # Add custom seasonalities
    model.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    )

    model.add_seasonality(
        name='quarterly',
        period=91.25,
        fourier_order=3
    )

    # Conditional seasonality (e.g., different patterns on weekends)
    df['is_weekend'] = df['ds'].dt.dayofweek >= 5

    model.add_seasonality(
        name='weekend_seasonality',
        period=7,
        fourier_order=3,
        condition_name='is_weekend'
    )

    print("\nAdded custom seasonalities:")
    print("- Monthly (period=30.5 days)")
    print("- Quarterly (period=91.25 days)")
    print("- Conditional weekend pattern")

    # Fit model
    model.fit(df)

    # Forecast
    future = model.make_future_dataframe(periods=90, freq='D')
    future['is_weekend'] = future['ds'].dt.dayofweek >= 5

    forecast = model.predict(future)

    print("\nCustom seasonality modeling complete")

    return model, forecast


def prophet_cross_validation():
    """Time series cross-validation with Prophet"""
    print("\n" + "=" * 60)
    print("Prophet Cross-Validation")
    print("=" * 60)

    df = generate_business_data(n_days=730)

    # Fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(df)

    # Cross-validation
    # initial: training period
    # period: spacing between cutoff dates
    # horizon: forecast horizon
    print("\nPerforming time series cross-validation...")
    print("This may take a moment...")

    df_cv = cross_validation(
        model,
        initial='365 days',
        period='90 days',
        horizon='90 days'
    )

    # Calculate performance metrics
    df_metrics = performance_metrics(df_cv)

    print(f"\nCross-Validation Metrics:")
    print(df_metrics[['horizon', 'mse', 'rmse', 'mae', 'mape']].head(10))

    # Plot metrics
    fig = plot_cross_validation_metric(df_cv, metric='rmse')
    plt.tight_layout()
    plt.savefig('/tmp/prophet_cv_metrics.png')
    print("\nCV metrics plot saved to /tmp/prophet_cv_metrics.png")

    return df_cv, df_metrics


def prophet_change_points():
    """Detect and visualize change points"""
    print("\n" + "=" * 60)
    print("Prophet Change Point Detection")
    print("=" * 60)

    # Generate data with abrupt changes
    df = generate_business_data(n_days=730)

    # Add some abrupt changes
    df.loc[365:, 'y'] += 50  # Step change at day 365

    # Model with change point detection
    model = Prophet(
        changepoint_prior_scale=0.5,  # Flexibility of trend changes
        n_changepoints=25  # Number of potential change points
    )

    model.fit(df)

    # Get change points
    change_points = model.changepoints

    print(f"\nDetected {len(change_points)} potential change points")
    print(f"First 5 change points:")
    for cp in change_points[:5]:
        print(f"  {cp}")

    # Forecast
    future = model.make_future_dataframe(periods=90, freq='D')
    forecast = model.predict(future)

    # Plot with change points
    fig = model.plot(forecast)
    plt.scatter(model.changepoints,
                forecast.loc[forecast['ds'].isin(model.changepoints), 'yhat'],
                c='red', marker='o', s=50, label='Change Points')
    plt.legend()
    plt.title('Prophet Forecast with Change Points')
    plt.tight_layout()
    plt.savefig('/tmp/prophet_changepoints.png')
    print("Change points plot saved to /tmp/prophet_changepoints.png")

    return model


def main():
    """Main execution function"""
    print("Time Series Forecasting with Facebook Prophet\n")

    # Example 1: Basic forecasting
    model1, forecast1 = basic_prophet_forecast()

    # Example 2: With holidays
    model2, forecast2 = prophet_with_holidays()

    # Example 3: Custom seasonality
    model3, forecast3 = prophet_custom_seasonality()

    # Example 4: Cross-validation
    cv_results, cv_metrics = prophet_cross_validation()

    # Example 5: Change point detection
    model5 = prophet_change_points()

    print("\n" + "=" * 60)
    print("Prophet Forecasting Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- Prophet handles multiple seasonalities automatically")
    print("- Holiday effects can be easily incorporated")
    print("- Custom seasonality patterns are supported")
    print("- Built-in cross-validation for robust evaluation")
    print("- Automatic change point detection")
    print("- Uncertainty intervals provide confidence bounds")
    print("- Great for business forecasting with interpretability")


if __name__ == "__main__":
    main()
