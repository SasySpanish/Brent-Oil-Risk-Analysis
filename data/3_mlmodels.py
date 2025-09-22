# Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Import dataset
data = pd.read_csv("brentv.csv", parse_dates=["Date"])
data.set_index("Date", inplace=True)
data1 = pd.read_csv("brent.csv", parse_dates=["Date"])
data1.set_index("Date", inplace=True)

# Log-returns
data["Log_Returns"] = np.log(data["Close"] / data["Close"].shift(1))
data['Log_Returns'] = data['Log_Returns'].fillna(method='bfill')
returns = data['Log_Returns']

# Calculate negative returns only (losses)
real_losses = returns.apply(lambda x: -x if x < 0 else 0)

### Machine Learning Models for Value at Risk
# Explanatory variables and target
X = data[['Close', 'Volume', 'Conditional_Volatility']] # explanatory variables
y = data['Log_Returns']  # target variable

# Model structure
qrf = RandomForestQuantileRegressor(
    n_estimators=100, 
    max_depth=6, 
    min_samples_leaf=3, 
    default_quantiles=[0.05],
    random_state=42
)

# Train the Quantile Regression Forest
qrf.fit(X, y)

# Compute the VaR using the lower quantile (5%)
VaR_QRF = qrf.predict(X, quantiles=[0.05])

print(f"95% VaR using Quantile Regression Forest: {VaR_QRF.mean():.2%}")

# Plot real losses vs VaR
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Real losses", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_QRF[5400:], label="95% VaR - Quantile Regression Forest", color='green', alpha=0.5)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  
plt.legend()
plt.show()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Observed returns", color="blue", alpha=0.4)
plt.plot(data.index, VaR_QRF, label="95% VaR - Quantile Regression Forest", color="green", alpha=0.6)
plt.legend()
plt.show()


## Gradient Boosting
# Quantile definition for VaR (e.g. 5% for 95% VaR)
quantile = 0.05

# Gradient Boosting model for the 5% quantile
gbr = GradientBoostingRegressor(
    loss='quantile', 
    alpha=0.05, 
    n_estimators=500, 
    learning_rate=0.01, 
    max_depth=3
)

# Train the model
gbr.fit(X, y)

# Predict VaR
VaR_GB = gbr.predict(X)
print(f"95% VaR using Gradient Boosting: {VaR_GB.mean():.2%}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Observed returns", color="blue", alpha=0.4)
plt.plot(data.index, VaR_GB, label="95% VaR - Gradient Boosting", color="orange", alpha=0.8)
plt.legend()
plt.show()

# Real losses vs VaR
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Real losses", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_GB[5400:], label="95% VaR - Gradient Boosting", color='orange', alpha=0.5)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  
plt.legend()
plt.show()


## XGBoost
import xgboost as xgb

# Quantile regression model
model_xgb = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.05, 
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3
)

# Train the model
model_xgb.fit(X, y)

# Predictions
VaR_XGB = model_xgb.predict(X)
print(f"95% VaR using XGBoost: {VaR_XGB.mean():.2%}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Observed returns", color="blue", alpha=0.4)
plt.plot(data.index, VaR_XGB, label="95% VaR - XGBoost", color="purple", alpha=0.8)
plt.legend()
plt.show()

# Real losses vs VaR
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Real losses", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_XGB[5400:], label="95% VaR - XGBoost", color="purple", alpha=0.8)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  
plt.legend()
plt.show()


## LightGBM
import lightgbm as lgb

# Create LightGBM dataset
train_data = lgb.Dataset(X, label=y)

# LightGBM parameters for quantile regression
params = {
    'objective': 'quantile',
    'alpha': 0.05,
    'metric': 'quantile',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9
}

# Train the model
model_lgb = lgb.train(params, train_data, num_boost_round=100)

# Predictions
VaR_LGB = model_lgb.predict(X)

print(f"95% VaR using LightGBM: {VaR_LGB.mean():.2%}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Observed returns", color="blue", alpha=0.4)
plt.plot(data.index, VaR_LGB, label="95% VaR - LightGBM", color="violet", alpha=0.8)
plt.legend()
plt.show()

# Real losses vs VaR
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Real losses", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_LGB[5400:], label="95% VaR - LightGBM", color="violet", alpha=0.8)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  
plt.legend()
plt.show()


## CatBoost
from catboost import CatBoostRegressor

# CatBoost quantile regression
model_cat = CatBoostRegressor(
    loss_function='Quantile:alpha=0.05',
    iterations=100, 
    learning_rate=0.1, 
    depth=3
)

# Train model
model_cat.fit(X, y)

# Predictions
VaR_CAT = model_cat.predict(X)

print(f"95% VaR using CatBoost: {VaR_CAT.mean():.2%}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Observed returns", color="blue", alpha=0.4)
plt.plot(data.index, VaR_CAT, label="95% VaR - CatBoost", color="deeppink", alpha=0.8)
plt.legend()
plt.show()

# Real losses vs VaR
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Real losses", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_CAT[5400:], label="95% VaR - CatBoost", color="deeppink", alpha=0.8)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  
plt.legend()
plt.show()


## Neural Networks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random

# Set seeds for reproducibility
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

# Explanatory variables and target
X = data[['Close', 'Volume', 'Conditional_Volatility']] 
y = data['Log_Returns']  

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.to_numpy().reshape(-1, 1))

# Quantile loss function
def quantile_loss(q):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return loss

# Neural Network model
model = Sequential([
    Dense(128, activation='relu', 
          kernel_regularizer=tf.keras.regularizers.l2(0.001), 
          input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu', 
          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
quantile = 0.05  

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=100, decay_rate=0.9)

model.compile(optimizer=Adam(learning_rate=lr_schedule), 
              loss=quantile_loss(quantile))

# Early stopping
early_stopping = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)

# Train the model
history = model.fit(X, y, epochs=100, batch_size=64, callbacks=[early_stopping])

# VaR predictions
VaR_NN = scaler.inverse_transform(model.predict(X))

print(f"95% VaR using Neural Networks: {VaR_NN.mean():.2%}")

# Training loss visualization
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss evolution during training')
plt.xlabel('Epochs')
plt.ylabel('Quantile Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot returns vs VaR
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Observed returns", color="blue", alpha=0.4)
plt.plot(data.index, VaR_NN, label="95% VaR - Neural Networks", color="mediumaquamarine", alpha=0.8)
plt.legend()
plt.show()

# Real losses vs VaR
returns = data['Log_Returns']
real_losses = returns.apply(lambda x: -x if x < 0 else 0)

plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Real losses", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_NN[5400:], label="95% VaR - Neural Networks", color="mediumaquamarine", alpha=0.8)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  
plt.legend()
plt.show()
