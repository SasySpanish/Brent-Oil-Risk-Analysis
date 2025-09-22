# Librerie 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import norm 
from arch import arch_model
from sklearn.preprocessing import StandardScaler

# Import dataset
data = pd.read_csv("brent.csv", parse_dates=["Date"])
data.set_index("Date", inplace=True)
data1=pd.read_csv('brentv.csv', parse_dates=["Date"])
data1.set_index("Date", inplace=True)

# Log-returns
data["Log_Returns"] = np.log(data["Close"] / data["Close"].shift(1))
data['Log_Returns'] = data['Log_Returns'].fillna(method='bfill')
returns=data['Log_Returns']


### 4.3 VaR Calculation

## Parametric VaR
# Mean and Standard Deviation of Log Returns
mean_return = np.mean(returns)
std_dev = np.std(returns)

# Parametric VaR at 95% confidence level using Z-score
confidence_level = 0.95
z_score = norm.ppf(1 - confidence_level)
VaR_parametric = mean_return + z_score * std_dev
print(f"Parametric VaR (95%): {VaR_parametric:.2%}")

# Normal distribution and VaR threshold
plt.figure(figsize=(12, 6))
x = np.linspace(mean_return - 3*std_dev, mean_return + 3*std_dev, 1000)
y = norm.pdf(x, mean_return, std_dev)
plt.plot(x, y, label='Normal Distribution')
plt.axvline(VaR_parametric, color='red', linestyle='--', label=f'VaR (95%): {VaR_parametric:.2%}')
plt.fill_between(x, 0, y, where=(x <= VaR_parametric), color='red', alpha=0.5)
plt.title('Normal Distribution of Log Returns')
plt.xlabel('Log Returns')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

## Historical VaR
# Historical VaR at 95% confidence level
VaR_historical = np.percentile(returns, (1 - confidence_level) * 100)
print(f"Historical VaR (95%): {VaR_historical:.2%}")

# Plot log-returns distribution with VaR threshold
plt.figure(figsize=(12, 6))
plt.hist(returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.axvline(VaR_historical, color='red', linestyle='--', label=f'VaR (95%): {VaR_historical:.2%}')
plt.title('Historical Brent Returns')
plt.xlabel('Log Returns')
plt.ylabel('Frequency')
plt.legend()
plt.show()

## GARCH
# GARCH(1,1) Model
garch = arch_model(returns, vol='GARCH', p=1, q=1)
results = garch.fit(disp='off')
print(results.summary())
results.plot()
print(results)

### Conditional Volatility
data['Conditional_Volatility'] = results.conditional_volatility

# Parametric VaR using average conditional volatility
volatility = results.conditional_volatility
volatility_mean = volatility.mean()

VaRP_Garch = mean_return + z_score * volatility_mean
print(f"GARCH-based Parametric VaR (95%): {VaRP_Garch:.2%}")

# Historical VaR using average conditional volatility
standardized_returns = returns / volatility
standard_p = np.percentile(standardized_returns, (1 - confidence_level) * 100)

VaRH_Garch = standard_p * volatility_mean
print(f"GARCH-based Historical VaR (95%): {VaRH_Garch:.2%}")

## Quantile Regression VaR
# Define explanatory variables X and target y
X = data[['Close', 'Volume', 'Conditional_Volatility']]
y = data['Log_Returns']

# Standardize X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add constant term
X_scaled = sm.add_constant(X_scaled)

# Quantile regression
regq = sm.QuantReg(y, X_scaled)
resultreg = regq.fit(q=0.05)
print(resultreg.summary())

# Calculate Quantile Regression VaR
VaR_R = resultreg.predict(X_scaled)
print(f"Quantile Regression VaR (95%): {VaR_R.mean():.2%}")

# Plot Quantile Regression VaR
plt.plot(data.index, data['Log_Returns'], label='Returns')
plt.plot(data.index, VaR_R, label='VaR (95%)', color='red')
plt.legend()
plt.show()

## Dynamic VaR
# Calculate negative returns only (losses)
real_losses = returns.apply(lambda x: -x if x < 0 else 0)
plt.figure(figsize=(12, 6))
plt.plot(data.index, real_losses, label="Real Losses", color="lightsteelblue")
plt.legend()
plt.show()

# Dynamic Parametric VaR
VaRP_dynamic = mean_return + z_score * volatility
print(f"Dynamic GARCH-based Parametric VaR (95%): {VaRP_dynamic.mean():.2%}")

# Dynamic Historical VaR
VaRH_dynamic = standard_p * volatility
print(f"Dynamic GARCH-based Historical VaR (95%): {VaRH_dynamic.mean():.2%}")

# Plot all three dynamic VaRs (last 4 years zoom)
plt.figure(figsize=(12, 6))
plt.plot(data.index[5400:], real_losses[5400:], label="Real Losses", color="lightsteelblue")
plt.plot(data.index[5400:], -VaRP_dynamic[5400:], label="Parametric VaR", color="blue", linestyle="-", alpha=0.7)
plt.plot(data.index[5400:], -VaRH_dynamic[5400:], label="Historical VaR", color="red", linestyle="-", alpha=0.8)
plt.plot(data.index[5400:], -VaR_R[5400:], label="Quantile Regression VaR", color="green", linestyle="-", alpha=0.8)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))  # Show only year
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())  # Set yearly ticks
plt.legend()
plt.title("Historical VaR on Real Losses")
plt.show()
