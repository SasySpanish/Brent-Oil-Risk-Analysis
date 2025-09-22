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

The study focused on [Brent crude oil due to its high volatility and central role in global financial and energy markets](Oil_Market). The main objective was to analyze the risk of potential losses through VaR estimation and evaluate the performance of various models.

### Traditional Models
- Parametric VaR: Assumes returns follow a normal distribution, calculating VaR based on mean and standard deviation.
- Historical VaR: Non-parametric method that uses empirical quantiles of past returns.
- GARCH-based VaR: Incorporates time-varying volatility using GARCH(1,1) models, generating both parametric and historical VaR based on conditional volatility.
- Quantile Regression VaR: Estimates VaR directly at a specified quantile using a regression framework, adapting better to the empirical distribution of returns.
- Quantile Regression was found to be the most effective among classical approaches, showing better calibration and higher precision in VaR estimation due to its flexibility in modeling quantiles.

### Machine Learning Models
- Machine Learning models were built using a Quantile Regression framework to maintain methodological consistency.
- Models implemented: Quantile Regression Forest (QRF), Gradient Boosting, XGBoost, LightGBM, CatBoost, and Neural Networks.
- Boosting-based models (Gradient Boosting, XGBoost, LightGBM, CatBoost) achieved the best overall performance, balancing calibration and accuracy while improving coverage and reducing estimation errors.
- LightGBM, in particular, provided excellent predictive performance for risk quantiles.
- Quantile Regression Forest showed lower adaptability and precision compared to boosting models.
- Neural Networks demonstrated potential but required careful hyperparameter tuning and model optimization to avoid overfitting and convergence issues.

### Backtesting Methodologies
To evaluate the performance of VaR models, the following backtesting techniques were applied:

1. Violation Rate / Hit Ratio: Measures the percentage of times the actual loss exceeded the predicted VaR. For a 95% VaR, approximately 5% of observations are expected to exceed the VaR threshold.
2. Kupiec Proportion of Failures Test: A statistical test that compares the observed number of VaR violations to the expected number, assessing the model's calibration.
3. Coverage: Calculates the fraction of observations where actual losses were below the predicted VaR, providing a simple measure of model reliability.
4. Quantile Loss Function: Evaluates the predictive accuracy for quantile regression models, penalizing under- and over-estimation asymmetrically.
5. MAE and RMSE: Standard error metrics (Mean Absolute Error and Root Mean Squared Error) to assess overall prediction accuracy.

The [backtesting analysis](results) confirmed that:
- Boosting-based Machine Learning models achieved superior performance in terms of coverage, calibration, and quantile accuracy.
- Traditional models, while effective, were outperformed by modern Machine Learning approaches in predicting extreme losses.
- Neural Networks required careful tuning but showed potential for further improvement with more advanced architectures.

### Key Findings
- Machine Learning models significantly outperform traditional VaR models in robustness and predictive reliability.
- Boosting techniques are particularly effective for risk modeling, confirming their value in financial risk management.
- The analysis highlights the potential for further improvements by integrating additional explanatory variables, using advanced deep learning techniques, and applying models to stress testing or portfolio-level risk evaluation.

