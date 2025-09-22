# Librerie 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import  chi2, norm 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

from arch import arch_model
from sklearn.preprocessing import StandardScaler


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

## Backtesting

# === VaR Variables needed for Backtesting ===

# Parametric VaR (95%)
mean_return = np.mean(returns)
std_dev = np.std(returns)
confidence_level = 0.95
z_score = norm.ppf(1 - confidence_level)
VaR_parametric = mean_return + z_score * std_dev

# Historical VaR (95%)
VaR_historical = np.percentile(returns, (1 - confidence_level) * 100)

# GARCH(1,1) model
garch = arch_model(returns, vol='GARCH', p=1, q=1)
results = garch.fit(disp='off')
volatility = results.conditional_volatility

# Dynamic GARCH-based Parametric VaR
VaRP_dynamic = mean_return + z_score * volatility

# Dynamic GARCH-based Historical VaR
standardized_returns = returns / volatility
standard_p = np.percentile(standardized_returns, (1 - confidence_level) * 100)
VaRH_dynamic = standard_p * volatility

# Quantile Regression VaR
X = data[['Close', 'Volume', 'Conditional_Volatility']]
y = data['Log_Returns']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = sm.add_constant(X_scaled)
regq = sm.QuantReg(y, X_scaled)
resultreg = regq.fit(q=0.05)
VaR_R = resultreg.predict(X_scaled)


#ML Variables
# Features and target
X = data[['Close', 'Volume', 'Conditional_Volatility']]
y = data['Log_Returns']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Quantile Random Forest ---
from sklearn.ensemble import RandomForestRegressor
qrf = RandomForestRegressor(n_estimators=100, random_state=42)
qrf.fit(X_train_scaled, y_train)
VaR_QRF = np.percentile(qrf.predict(X_test_scaled), 5)  # 5th percentile (95% VaR)

# --- Gradient Boosting ---
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(loss="quantile", alpha=0.05, random_state=42)
gbr.fit(X_train_scaled, y_train)
VaR_GB = gbr.predict(X_test_scaled)

# --- XGBoost ---
import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train_scaled, y_train)
VaR_XGB = np.percentile(xgb_model.predict(X_test_scaled), 5)

# --- LightGBM ---
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(objective='quantile', alpha=0.05, random_state=42)
lgb_model.fit(X_train_scaled, y_train)
VaR_LGB = lgb_model.predict(X_test_scaled)

# --- CatBoost ---
from catboost import CatBoostRegressor
cat_model = CatBoostRegressor(loss_function='Quantile:alpha=0.05', verbose=0, random_seed=42)
cat_model.fit(X_train_scaled, y_train)
VaR_CAT = cat_model.predict(X_test_scaled)

# --- Neural Network ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nn = Sequential()
nn.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
nn.add(Dense(32, activation='relu'))
nn.add(Dense(1, activation='linear'))

nn.compile(optimizer='adam', loss='mean_squared_error')
nn.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

VaR_NN = nn.predict(X_test_scaled)  # array shape (n,1)

# Funzione per calcolare le violazioni del VaR e i risultati del test di Kupiec
y_true = -real_losses

def calculate_var_violations(real_losses, models, alpha=0.05):
    backtest1 = {}
    n_obs = len(real_losses)

    for model_name, var_values in models.items():
        # Calcolo delle violazioni
        violations = -real_losses < var_values
        n_violations = np.sum(violations)

        # Hit Ratio
        hit_ratio = n_violations / n_obs

        # Kupiec Test
        pof_stat = -2 * (n_obs * np.log(1 - alpha) + n_violations * np.log(alpha)) + 2 * (n_obs * np.log(1 - hit_ratio) + n_violations * np.log(hit_ratio))
        p_value = 1 - chi2.cdf(pof_stat, df=1)

        # Salva i risultati
        backtest1[model_name] = {
            'Hit Ratio': hit_ratio,
            'Expected Hit Ratio': alpha,
            'Kupiec Test Statistic': pof_stat,
            'p-value': p_value
        }

    return backtest1

# Funzioni per calcolare vari metriche
def calculate_violations(y_true, y_pred):
    return ((np.sum(y_true < y_pred)) / (len(y_true))) * 100

def calculate_coverage(y_true, y_pred):
    return np.mean(y_true >= y_pred)

def quantile_loss(y_true, y_pred, alpha=0.05):
    error = y_true - y_pred
    return np.mean(np.where(error > 0, alpha * error, (alpha - 1) * error))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# Modelli VaR
models1 = {
    'Parametrico': VaR_parametric,
    'Storico': VaR_historical,
    'Parametrico GARCH': VaRP_dynamic,
    'Storico GARCH': VaRH_dynamic,
    'Regressione Quantilica': VaR_R
}
#Trasformo in array gli scalari 
scalar_models = ['Parametrico', 'Storico']

for model_name in scalar_models:
    if np.isscalar(models1[model_name]):  # Se il valore è scalare
        models1[model_name] = np.full_like(y_true, models1[model_name])  # Crea un array riempito con quel valore

# Risultati
backtesting1 = {}

for model_name, y_pred in models1.items():
    var = y_pred.mean() * 100
    violations = calculate_violations(y_true, y_pred)
    kupiec = calculate_var_violations(real_losses, {model_name: y_pred})[model_name]['Kupiec Test Statistic']
    coverage = calculate_coverage(y_true, y_pred)
    q_loss = quantile_loss(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    
    backtesting1[model_name] = {
        'VaR': f"{var:.2f}%",
        'Violations': f"{violations:.2f}%",
        'Kupiec': f"{kupiec:.4f}",
        'Coverage': f"{coverage:.5f}",
        'MAE': f"{mae:.5f}",
        'MSE': f"{rmse:.5f}"
    }

# Creazione di un DataFrame per visualizzare i risultati
backtesting1_df = pd.DataFrame(backtesting1).T
backtesting1_df['Model'] = backtesting1_df.index

# Riordina le colonne per avere 'Model' come prima colonna
cols = ['Model'] + [col for col in backtesting1_df.columns if col != 'Model']
backtesting1_df = backtesting1_df[cols]

# Stampa dei risultati
print(backtesting1_df)

# Subplot della tabella
fig, ax = plt.subplots(figsize=(8, 6))  # Imposta le dimensioni della figura
ax.axis('off')  # Nascondi gli assi
table = ax.table(cellText=backtesting1_df.values, colLabels=backtesting1_df.columns, loc='center')
for i in range(len(backtesting1_df.columns)):
    max_len = max(backtesting1_df[backtesting1_df.columns[i]].astype(str).map(len).max(), len(backtesting1_df.columns[i]))  # Trova la lunghezza massima
    table.auto_set_column_width([i])  # Imposta la larghezza automatica della colonna
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Scala le dimensioni della tabella
plt.show()

## Backtesting e Confronto
# Modelli ML
models2 = {
    'QR Forest': VaR_QRF,
    'Gradient Boosting': VaR_GB,
    'XGBoost': VaR_XGB,
    'LightGBM': VaR_LGB,
    'CatBoost': VaR_CAT,
    'Neural Networks': VaR_NN[:, 0]
}
# Risultati
backtesting2 = {}
print(quantile_loss)

for model_name, y_pred in models2.items():
    var = y_pred.mean() * 100
    violations = calculate_violations(y_true, y_pred)
    kupiec = calculate_var_violations(real_losses, {model_name: y_pred})[model_name]['Kupiec Test Statistic']
    coverage = calculate_coverage(y_true, y_pred)
    q_loss = quantile_loss(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    
    backtesting2[model_name] = {
        'VaR': f"{var:.2f}%",
        'Violations': f"{violations:.2f}%",
        'Kupiec': f"{kupiec:.4f}",
        'Coverage': f"{coverage:.5f}",
        'Quantile Loss': f"{q_loss:.5f}",
        'MAE': f"{mae:.5f}",
        'MSE': f"{rmse:.5f}"
    }

# Creazione di un DataFrame per visualizzare i risultati
backtesting2_df = pd.DataFrame(backtesting2).T
backtesting2_df['Model'] = backtesting2_df.index

# Riordina le colonne per avere 'Model' come prima colonna
cols2 = ['Model'] + [col for col in backtesting2_df.columns if col != 'Model']
backtesting2_df = backtesting2_df[cols]

# Stampa dei risultati
print(backtesting2_df)

# Subplot della Tabella
fig, ax = plt.subplots(figsize=(8, 6))  # Imposta le dimensioni della figura
ax.axis('off')  # Nascondi gli assi
table = ax.table(cellText=backtesting2_df.values, colLabels=backtesting2_df.columns, loc='center')
for i in range(len(backtesting2_df.columns)):
    max_len = max(backtesting2_df[backtesting2_df.columns[i]].astype(str).map(len).max(), len(backtesting2_df.columns[i]))  # Trova la lunghezza massima
    table.auto_set_column_width([i])  # Imposta la larghezza automatica della colonna
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Scala le dimensioni della tabella
plt.show()

#perdite
plt.figure(figsize=(12, 6))
plt.plot(data.index[1200:1500], real_losses[1200:1500], label="Perdite osservate", color="blue", alpha=0.7)
plt.plot(data.index[1200:1500], -VaR_GB[1200:1500]+0.005, label="VaR al 95% con Gradient Boosting", color="orange")
plt.plot(data.index[1200:1500], -VaR_XGB[1200:1500]-0.008, label="VaR al 95% con XGB", color="purple")
plt.plot(data.index[1200:1500], -VaR_LGB[1200:1500]-0.008, label="VaR al 95% con LGB", color="cyan",alpha=0.6)
plt.plot(data.index[1200:1500], -VaR_CAT[1200:1500], label="VaR al 95% con CAT", color="red")
plt.legend(loc='upper right')
plt.show()

#rendimenti
plt.figure(figsize=(12, 6))
plt.plot(data.index[1000:1500], data['Log_Returns'][1000:1500], label="Rendimenti osservati", color="blue", alpha=0.7)
plt.plot(data.index[1000:1500], VaR_GB[1000:1500]-0.005, label="VaR al 95% con Gradient Boosting", color="orange")
plt.plot(data.index[1000:1500], VaR_XGB[1000:1500]+0.008, label="VaR al 95% con XGB", color="purple")
plt.plot(data.index[1000:1500], VaR_LGB[1000:1500]+0.008, label="VaR al 95% con LGB", color="cyan",alpha=0.6)
plt.plot(data.index[1000:1500], VaR_CAT[1000:1500], label="VaR al 95% con CAT", color="red")
plt.plot(data.index[1000:1500], VaR_QRF[1000:1500], label="VaR al 95% con QRF", color="lime",alpha=0.4)
plt.plot(data.index[1000:1500], VaR_NN[1000:1500]+0.003, label="VaR al 95% con NN", color="mediumaquamarine")
plt.legend()
plt.show()


### Confronto Tutti i modelli

# Modelli e le loro predizioni
models = {
    'Parametrico': VaR_parametric,
    'Storico': VaR_historical,
    'Parametrico GARCH': VaRP_dynamic,
    'Storico GARCH': VaRH_dynamic,
    'Regressione Quantilica': VaR_R
    'QR Forest': VaR_QRF,
    'Gradient Boosting': VaR_GB,
    'XGBoost': VaR_XGB,
    'LightGBM': VaR_LGB,
    'CatBoost': VaR_CAT,
    'Neural Networks': VaR_NN[:, 0]
}

# Controllo se i modelli nella lista sono scalari e li trasformo in array della stessa lunghezza di y_true
for model_name in scalar_models:
    if np.isscalar(models[model_name]):  # Se il valore è scalare
        models[model_name] = np.full_like(y_true, models[model_name])  # Crea un array riempito con quel valore

# Risultati
results = {}

for model_name, y_pred in models.items():
    var = y_pred.mean() * 100
    violations = calculate_violations(y_true, y_pred)
    kupiec = calculate_var_violations(real_losses, {model_name: y_pred})[model_name]['Kupiec Test Statistic']
    coverage = calculate_coverage(y_true, y_pred)
    q_loss = quantile_loss(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    
    results[model_name] = {
        'VaR': f"{var:.2f}%",
        'Violations': f"{violations:.2f}%",
        'Kupiec': f"{kupiec:.4f}",
        'Coverage': f"{coverage:.5f}",
        'Quantile Loss': f"{q_loss:.5f}",
        'MAE': f"{mae:.5f}",
        'MSE': f"{rmse:.5f}"
    }

# Creazione di un DataFrame per visualizzare i risultati
results_df = pd.DataFrame(results).T
results_df['Model'] = results_df.index

# Riordina le colonne per avere 'Model' come prima colonna
cols = ['Model'] + [col for col in results_df.columns if col != 'Model']
results_df = results_df[cols]

# Stampa dei risultati
print(results_df)

# Tabella
fig, ax = plt.subplots(figsize=(8, 6))  # Imposta le dimensioni della figura
ax.axis('off')  # Nascondi gli assi
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
for i in range(len(results_df.columns)):
    max_len = max(results_df[results_df.columns[i]].astype(str).map(len).max(), len(results_df.columns[i]))  # Trova la lunghezza massima
    table.auto_set_column_width([i])  # Imposta la larghezza automatica della colonna
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Scala le dimensioni della tabella
plt.show()
