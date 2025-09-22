# Brent Oil Value at Risk (VaR) Analysis

This repository contains the analysis and results of a study on the estimation of Value at Risk (VaR) for Brent crude oil. The analysis compares classical VaR methods with advanced Machine Learning models based on Quantile Regression, aiming to evaluate the risk associated with this highly volatile and globally significant commodity.

## Repository Structure

- `data/` : Contains raw and processed datasets used for the analysis.
- `notebooks/` : Jupyter notebooks with full analysis, model implementation, and plot generation.
- `src/` : Python scripts for data processing, modeling, and backtesting.
- `results/` : PNG images of the backtesting results and key plots, including:
  1. Comparison of the four boosting models (Gradient Boosting, XGBoost, LightGBM, CatBoost) against actual losses.
  2. Comparison of all Machine Learning models against observed returns.
  3. Comparison of traditional VaR models (including GARCH and Quantile Regression) against actual losses.

## Analysis Summary

The study focused on Brent crude oil due to its high volatility and central role in global financial and energy markets. The main objective was to analyze the risk of potential losses through VaR estimation and evaluate the performance of various models.

### Traditional Models
- Parametric VaR, Historical VaR, GARCH-based models, and Quantile Regression were implemented.
- Quantile Regression was found to be the most effective among classical approaches, showing better calibration and higher precision in VaR estimation due to its flexibility in modeling quantiles.

### Machine Learning Models
- Machine Learning models were built using a Quantile Regression framework to maintain methodological consistency.
- Models implemented: Quantile Regression Forest (QRF), Gradient Boosting, XGBoost, LightGBM, CatBoost, and Neural Networks.
- Boosting-based models (Gradient Boosting, XGBoost, LightGBM, CatBoost) achieved the best overall performance, balancing calibration and accuracy while improving coverage and reducing estimation errors.
- LightGBM, in particular, provided excellent predictive performance for risk quantiles.
- Quantile Regression Forest showed lower adaptability and precision compared to boosting models.
- Neural Networks demonstrated potential but required careful hyperparameter tuning and model optimization to avoid overfitting and convergence issues.

### Key Findings
- Machine Learning models significantly outperform traditional VaR models in robustness and predictive reliability.
- Boosting techniques are particularly effective for risk modeling, confirming their value in financial risk management.
- The analysis highlights the potential for further improvements by integrating additional explanatory variables, using advanced deep learning techniques, and applying models to stress testing or portfolio-level risk evaluation.

