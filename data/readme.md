# Data

This folder contains all the datasets used for the Brent VaR analysis, including raw and processed data.  
- `brent.csv`: Historical Brent crude prices.
- `brentv.csv`: Modified version during the process (added log returns and conditional volatility columns).
- Additional data used for feature engineering or volatility calculation.  

### Dataset Features

The dataset includes the following key variables:

- **Close**: Daily closing price of Brent, the primary reference for return and risk calculation.  
- **Open**: Daily opening price, providing insights into intraday dynamics and potential price gaps.  
- **High** / **Low**: Maximum and minimum prices reached during the trading day, useful for assessing daily volatility and identifying abnormal market movements.  
- **Volume**: Number of contracts traded each day, indicating liquidity and investor interest.  
- **Return**: Daily percentage change of the closing price relative to the previous day, offering a direct measure of performance and relative volatility.

## Log-Returns Transformation

To make the dataset suitable for VaR analysis, log-returns were calculated as the natural logarithm of the ratio between the current and previous day’s closing price.
Log-returns are preferred over simple returns because they are additive over time, less sensitive to extreme price variations, and exhibit statistical properties better suited for financial risk analysis. They provide a more stationary series and reduce heteroscedasticity, facilitating the modeling of extreme quantiles essential for VaR estimation. This transformation is also particularly useful for machine learning models, as it produces a variable that is more manageable and representative of market dynamics.
