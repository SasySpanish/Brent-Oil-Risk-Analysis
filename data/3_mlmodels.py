# Librerie 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Import dataset
data = pd.read_csv("brentv.csv", parse_dates=["Date"])
data.set_index("Date", inplace=True)
data1=pd.read_csv('brent.csv', parse_dates=["Date"])
data1.set_index("Date", inplace=True)

# Log-returns
data["Log_Returns"] = np.log(data["Close"] / data["Close"].shift(1))
data['Log_Returns'] = data['Log_Returns'].fillna(method='bfill')
returns=data['Log_Returns']

# Calculate negative returns only (losses)
real_losses = returns.apply(lambda x: -x if x < 0 else 0)

### Machine Learning Models for Value at Risk
# Creazione delle variabili esplicative e della variabile target
X = data[['Close', 'Volume', 'Conditional_Volatility']] # variabili esplicative
y = data['Log_Returns']  #variabile target 

# Struttura del modello
qrf = RandomForestQuantileRegressor(n_estimators=100, 
                                    max_depth=6, 
                                    min_samples_leaf=3, 
                                    default_quantiles=[0.05],
                                    random_state=42)
# Addestramento della Quantile Regression Forest
qrf.fit(X, y)

# Calcolo del VaR con il quantile inferiore (5%)
VaR_QRF = qrf.predict(X, quantiles=[0.05])

print(f"VaR al 95% Tramite Quantile Regression Forest: {VaR_QRF.mean():.2%}")
# Visualizzazione perdite
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Perdite reali", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_QRF[5400:], label="VaR al 95% con Quantile Regression Forest", color='green', alpha=0.5)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  # Mostra solo l'anno
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  # Imposta i tick annuali
plt.legend()
plt.show()

# Plot dei risultati
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Rendimenti osservati", color="blue", alpha=0.4)
plt.plot(data.index, VaR_QRF, label="VaR al 95% con Quantile Regression Forest", color="green", alpha=0.6)
plt.legend()
plt.show()


## Gradient Boosting
# Definizione del quantile per il VaR (es. 5% per il VaR a 95%)
quantile = 0.05

# Modello Gradient Boosting per il quantile 5%
gbr = GradientBoostingRegressor(
      loss='quantile', 
      alpha=0.05, 
      n_estimators=500, 
      learning_rate=0.01, 
      max_depth=3
      )

# Training del modello su tutto il dataset
gbr.fit(X, y)

# Predizione del VaR su tutto il dataset
VaR_GB = gbr.predict(X)
print(VaR_GB.mean())
print(f"VaR al 95% Tramite Gradient Boosting: {VaR_GB.mean():.2%}")
# Plot dei risultati
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Rendimenti osservati", color="blue", alpha=0.4)
plt.plot(data.index, VaR_GB, label="VaR al 95% con Gradient Boosting", color="orange", alpha=0.8)
plt.legend()
plt.show()


# Visualizzazione perdite
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Perdite reali", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_GB[5400:], label="VaR al 95% con Gradient Boosting", color='orange', alpha=0.5)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  # Mostra solo l'anno
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  # Imposta i tick annuali
plt.legend()
plt.show()

## XG BOOST
import xgboost as xgb

# Creazione del modello per la regressione quantilica
model_xgb = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.05, 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3
            )

# Addestramento
model_xgb.fit(X, y)

# Previsioni VaR al 95%
VaR_XGB = model_xgb.predict(X)
print(f"VaR al 95% Tramite XGB: {VaR_XGB.mean():.2%}")

# Plot dei risultati
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Rendimenti osservati", color="blue", alpha=0.4)
plt.plot(data.index, VaR_XGB, label="VaR al 95% con XGB", color="purple", alpha=0.8)
plt.legend()
plt.show()

# Visualizzazione perdite
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Perdite reali", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_XGB[5400:], label="VaR al 95% con XGB", color="purple", alpha=0.8)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  # Mostra solo l'anno
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  # Imposta i tick annuali
plt.legend()
plt.show()

## LightBGM
import lightgbm as lgb

# Creazione del dataset LightGBM
train_data = lgb.Dataset(X, label=y)

# Parametri del modello LGB per la regressione quantilica
params = {
          'objective': 'quantile',
          'alpha': 0.05,
          'metric': 'quantile',
          'boosting_type': 'gbdt',
          'num_leaves': 31,
          'learning_rate': 0.1,
          'feature_fraction': 0.9
          }

# Addestramento del modello
model_lgb = lgb.train(params, train_data, num_boost_round=100)

# Previsioni VaR al 95%
VaR_LGB = model_lgb.predict(X)

print(f"VaR al 95% Tramite LightGBM: {VaR_LGB.mean():.2%}")

# Plot dei risultati
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Rendimenti osservati", color="blue", alpha=0.4)
plt.plot(data.index, VaR_LGB, label="VaR al 95% con LightGBM", color="violet", alpha=0.8)
plt.legend()
plt.show()


# Visualizzazione perdite
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Perdite reali", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_LGB[5400:], label="VaR al 95% con LightGBM", color="violet", alpha=0.8)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  # Mostra solo l'anno
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  # Imposta i tick annuali
plt.legend()
plt.show()


## CatBoost
from catboost import CatBoostRegressor

# Creazione del modello per la regressione quantilica 
model_cat = CatBoostRegressor(
            loss_function='Quantile:alpha=0.05',
            iterations=100, 
            learning_rate=0.1, 
            depth=3
            )

# Addestramento del modello
model_cat.fit(X, y)

# Previsioni VaR al 95%
VaR_CAT = model_cat.predict(X)

print(f"VaR al 95% Tramite CatBoost: {VaR_CAT.mean():.2%}")

# Visualizzazione Risultati
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Rendimenti osservati", color="blue", alpha=0.4)
plt.plot(data.index, VaR_CAT, label="VaR al 95% con CatBoost", color="deeppink", alpha=0.8)
plt.legend()
plt.show()


# Visualizzazione perdite
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Perdite reali", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_CAT[5400:], label="VaR al 95% con CatBoost", color="deeppink", alpha=0.8)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  # Mostra solo l'anno
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  # Imposta i tick annuali
plt.legend()
plt.show()


## Reti Neurali
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random

# Imposta il seed per TensorFlow, NumPy e Python
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)


# Creazione delle variabili esplicative e della variabile target
X = data[['Close', 'Volume', 'Conditional_Volatility']] # variabili esplicative
y = data['Log_Returns']  #variabile target 

# Standardizzazione 
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.to_numpy().reshape(-1, 1))

# Funzione di perdita per la regressione quantilica
def quantile_loss(q):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return loss

# Modello di Reti Neurali
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


# Compilazione del modello
quantile = 0.05  # Quantile per il VaR al 95%

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.002, decay_steps=100, decay_rate=0.9)

model.compile(optimizer=Adam(learning_rate=lr_schedule), 
              loss=quantile_loss(quantile))

# Definizione di callback per early stopping
early_stopping = EarlyStopping(monitor='loss', patience=30, 
                               restore_best_weights=True)

# Addestramento del modello
history = model.fit(X, y, epochs=100, batch_size=64, callbacks=[early_stopping])

# Previsione del VaR
VaR_NN = scaler.inverse_transform(model.predict(X))

print(f"VaR al 95% Tramite Neural Networks: {VaR_NN.mean():.2%}")

# Visualizzazione della loss e della validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')

plt.title('Andamento della Loss durante l\'Addestramento')
plt.xlabel('Epoche')
plt.ylabel('Quantile Loss')
plt.legend()
plt.grid(True)
plt.show()

# Visualizzazione rendimenti
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Returns'], label="Rendimenti osservati", color="blue", alpha=0.4)
plt.plot(data.index, VaR_NN, label="VaR al 95% con Neural Networks", color="mediumaquamarine", alpha=0.8)
plt.legend()
plt.show()


# Visualizzazione perdite

returns= data['Log_Returns']
real_losses= returns.apply(lambda x: -x if x < 0 else 0)

plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Perdite reali", color="lightsteelblue")
plt.plot(data.index[5400:], -VaR_NN[5400:], label="VaR al 95% con Neural Networks", color="mediumaquamarine", alpha=0.8)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  # Mostra solo l'anno
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  # Imposta i tick annuali
plt.legend()
plt.show()