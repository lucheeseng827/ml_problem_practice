"""
Temporal Fusion Transformer for Multi-Horizon Forecasting
=========================================================
Category 10: Time Series Forecasting

This example demonstrates:
- Attention-based time series forecasting
- Multi-horizon predictions
- Variable selection networks
- Static and time-varying features
- Interpretable temporal patterns

Use cases:
- Multi-step ahead forecasting
- Retail demand forecasting
- Energy load prediction
- Financial market forecasting
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


class TemporalFusionTransformer(keras.Model):
    """
    Simplified Temporal Fusion Transformer architecture

    Full implementation requires complex gating mechanisms,
    variable selection networks, and multi-head attention.
    This is a demonstration version showing key concepts.
    """

    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1, forecast_horizon=10):
        super(TemporalFusionTransformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.forecast_horizon = forecast_horizon

        # Embedding layers for static features
        self.static_encoder = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(hidden_dim)
        ])

        # LSTM for encoding past sequences
        self.encoder_lstm = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True
        )

        # Multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout
        )

        # Gated Residual Network (simplified)
        self.grn = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(hidden_dim),
            layers.LayerNormalization()
        ])

        # Output projection
        self.output_layer = layers.Dense(forecast_horizon)

    def call(self, inputs, training=False):
        # inputs: [batch, time_steps, features]

        # Encode sequence with LSTM
        encoded, h_state, c_state = self.encoder_lstm(inputs)

        # Apply attention
        attended = self.attention(encoded, encoded, training=training)

        # Gated residual connection
        gated = self.grn(attended + encoded)

        # Global pooling
        pooled = tf.reduce_mean(gated, axis=1)

        # Output forecast
        output = self.output_layer(pooled)

        return output


class AttentionLSTM(keras.Model):
    """LSTM with attention mechanism for time series"""

    def __init__(self, lstm_units=64, attention_units=32, forecast_horizon=10, dropout=0.2):
        super(AttentionLSTM, self).__init__()

        self.lstm = layers.LSTM(lstm_units, return_sequences=True)
        self.dropout = layers.Dropout(dropout)

        # Attention mechanism
        self.attention_weights_layer = layers.Dense(attention_units, activation='tanh')
        self.attention_score_layer = layers.Dense(1)

        self.output_layer = layers.Dense(forecast_horizon)

    def call(self, inputs, training=False):
        # LSTM encoding
        lstm_out = self.lstm(inputs)
        lstm_out = self.dropout(lstm_out, training=training)

        # Attention mechanism
        attention_weights = self.attention_weights_layer(lstm_out)
        attention_scores = self.attention_score_layer(attention_weights)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)

        # Context vector
        context = tf.reduce_sum(attention_scores * lstm_out, axis=1)

        # Output
        output = self.output_layer(context)

        return output


def generate_multivariate_time_series(n_samples=1000, n_features=5):
    """Generate synthetic multivariate time series"""
    np.random.seed(42)

    t = np.arange(n_samples)

    # Create multiple correlated features
    data = np.zeros((n_samples, n_features))

    for i in range(n_features):
        # Each feature has trend + seasonality + noise
        trend = 0.01 * t * (i + 1)
        seasonal = 10 * np.sin(2 * np.pi * t / (50 + i * 10))
        noise = np.random.randn(n_samples) * 2

        data[:, i] = trend + seasonal + noise + 50 * (i + 1)

    return data


def create_multivariate_sequences(data, lookback=50, forecast_horizon=10):
    """Create sequences for multi-horizon forecasting"""
    X, y = [], []

    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i + lookback])
        # Predict next 'forecast_horizon' steps of first feature
        y.append(data[i + lookback:i + lookback + forecast_horizon, 0])

    return np.array(X), np.array(y)


def train_tft_model():
    """Train Temporal Fusion Transformer"""
    print("=" * 60)
    print("Temporal Fusion Transformer Training")
    print("=" * 60)

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Generate data
    print("\nGenerating multivariate time series...")
    data = generate_multivariate_time_series(n_samples=2000, n_features=5)

    # Normalize
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    # Create sequences
    lookback = 60
    forecast_horizon = 10

    X, y = create_multivariate_sequences(
        data_normalized,
        lookback=lookback,
        forecast_horizon=forecast_horizon
    )

    print(f"Input shape: {X.shape}, Target shape: {y.shape}")

    # Train/val/test split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Build model
    model = TemporalFusionTransformer(
        hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        forecast_horizon=forecast_horizon
    )

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print("\nModel Architecture:")
    model.build(input_shape=(None, lookback, 5))
    model.summary()

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)

    # Calculate metrics for each horizon
    horizons_mae = []
    for h in range(forecast_horizon):
        mae = mean_absolute_error(y_test[:, h], y_pred[:, h])
        horizons_mae.append(mae)
        print(f"Horizon {h + 1}: MAE = {mae:.4f}")

    # Overall metrics
    overall_mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    overall_rmse = np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))

    print(f"\nOverall Test Performance:")
    print(f"MAE: {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training history
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training History')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MAE per horizon
    axes[0, 1].bar(range(1, forecast_horizon + 1), horizons_mae)
    axes[0, 1].set_xlabel('Forecast Horizon')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('MAE by Forecast Horizon')
    axes[0, 1].grid(True, axis='y')

    # Example predictions (first test sample)
    sample_idx = 0
    axes[1, 0].plot(y_test[sample_idx], 'o-', label='Actual', markersize=6)
    axes[1, 0].plot(y_pred[sample_idx], 's-', label='Predicted', markersize=6)
    axes[1, 0].set_xlabel('Forecast Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title(f'Sample Prediction (Test #{sample_idx})')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Multiple predictions
    num_samples = 5
    for i in range(num_samples):
        axes[1, 1].plot(y_pred[i], alpha=0.6, label=f'Pred {i + 1}')

    axes[1, 1].set_xlabel('Forecast Step')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Multiple Sample Predictions')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/tft_results.png')
    print("\nResults saved to /tmp/tft_results.png")

    return model, history


def train_attention_lstm():
    """Train attention-based LSTM"""
    print("\n" + "=" * 60)
    print("Attention LSTM Training")
    print("=" * 60)

    np.random.seed(42)
    tf.random.set_seed(42)

    # Generate and prepare data
    data = generate_multivariate_time_series(n_samples=2000, n_features=3)
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    lookback = 60
    forecast_horizon = 10

    X, y = create_multivariate_sequences(
        data_normalized,
        lookback=lookback,
        forecast_horizon=forecast_horizon
    )

    # Split
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Build model
    model = AttentionLSTM(
        lstm_units=64,
        attention_units=32,
        forecast_horizon=forecast_horizon,
        dropout=0.2
    )

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    print("\nTraining Attention LSTM...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())

    print(f"\nAttention LSTM Test MAE: {mae:.4f}")
    print("Attention mechanism helps focus on relevant time steps")

    return model


def main():
    """Main execution function"""
    print("Temporal Fusion Transformer for Time Series Forecasting\n")

    # Example 1: TFT model
    tft_model, tft_history = train_tft_model()

    # Example 2: Attention LSTM
    attention_model = train_attention_lstm()

    print("\n" + "=" * 60)
    print("Advanced Time Series Forecasting Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- TFT combines attention with variable selection")
    print("- Multi-horizon forecasting predicts multiple steps ahead")
    print("- Attention mechanisms identify important time steps")
    print("- Multivariate features improve forecast accuracy")
    print("- Monitoring performance per horizon is crucial")
    print("- Transformer architectures excel at complex patterns")


if __name__ == "__main__":
    main()
