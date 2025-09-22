### 1. EDA
# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera, shapiro
from statsmodels.stats.diagnostic import normal_ad


# Import dataset
data = pd.read_csv("brent.csv", parse_dates=["Date"])
data.set_index("Date", inplace=True)
data1=pd.read_csv('brentv.csv', parse_dates=["Date"])
data1.set_index("Date", inplace=True)

# Log-returns
data["Log_Returns"] = np.log(data["Close"] / data["Close"].shift(1))
data['Log_Returns'] = data['Log_Returns'].fillna(method='bfill')
returns=data['Log_Returns']

# Descriptive statistics
closestat = data["Close"].describe()
print(closestat)
volumestat = data["Volume"].describe()
print(volumestat)
logstat = data["Log_Returns"].describe()
print(logstat)

# Price chart
plt.figure(figsize=(12,6))
plt.plot(data.index, data["Close"], label="Brent Price", color="blue")
plt.title("Historical Trend of Brent Price")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid()
plt.show()

# Histogram of log-returns distribution
plt.figure(figsize=(12,6))
sns.histplot(data["Log_Returns"], bins=50, kde=True, color="blue")
plt.title("Distribution of Log-Returns")
plt.xlabel("Log-Return")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Skewness
skewness = skew(data["Log_Returns"])
print('Skewness:', skewness)
# Kurtosis
curtosi = kurtosis(data["Log_Returns"])
print('Kurtosis:', curtosi)

# Normality tests
jb_stat, jb_p = jarque_bera(data["Log_Returns"])
shapiro_stat, shapiro_p = shapiro(data["Log_Returns"])
print(f"Jarque-Bera Test: Stat={jb_stat:.4f}, p-value={jb_p:.4f}")
print(f"Shapiro-Wilk Test: Stat={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")

alpha = 0.05
if shapiro_p > alpha:
    print('Data appears normally distributed (fail to reject H0)')
else:
    print('Data does not appear normally distributed (reject H0)')

# Kolmogorov-Smirnov Test
ks_test = stats.kstest(data["Log_Returns"], 'norm')
print(f"Kolmogorov-Smirnov Test: statistic={ks_test[0]}, p-value={ks_test[1]:.7f}")
anderson_test = normal_ad(data["Log_Returns"])
print(f"Anderson-Darling Test: statistic={anderson_test[0]}, p-value={anderson_test[1]}")

# QQ-Plot
plt.figure(figsize=(12,6))
sm.qqplot(data["Log_Returns"], line='s')
plt.title("Q-Q Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Empirical Quantiles")
plt.show()

# Boxplot of log-returns
plt.figure(figsize=(12,6))
sns.boxplot(x=data["Log_Returns"])
plt.title("Boxplot of Log-Returns")
plt.grid()
plt.show()

# Empirical quantiles for tail analysis
print(data["Log_Returns"].quantile([0.01, 0.05, 0.95, 0.99]))

# Correlation matrix
correlation_matrix = data[["Close", "Volume", "Log_Returns"]].corr()
# Heatmap of correlations
plt.figure(figsize=(12,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Between Variables")
plt.show()
