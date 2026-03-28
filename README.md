# Analyzing Volatility and Trading Volume Across btc's Market Regimes Using a Markov-Switching Vector Autoregression Model

This repository contains Python scripts and generated figures for analyzing the relationship between BTC trading volume and volatility under different market regimes.

## Main Conclusion

The main conclusion of this repository is based on the visual evidence shown in `Figure_1.png`.

Based on `Figure_1.png`, trading volume and volatility show relatively strong correlation in State 1 (upward trend) and State 3 (sideways regime), while in State 2 (downward trend) the relationship is not strong.

## Code Files

- `read.py`: Reads the BTC dataset and prints basic information.
- `coef_ana.py`: Tests simple relationships among volume, return, and volatility, and saves scatter plots.
- `lag_p.py`: Checks stationarity, selects lag length, and runs VAR diagnostics.
- `var_order.py`: Selects the best VAR lag order for the chosen market period.
- `bullish_initial.py`: Fits an initial VAR model for the bullish period.
- `bullish_bestmodel.py`: Builds an improved bullish-period VAR model and prints long-run mean, IRF, and FEVD results.
- `ms_var.py`: Main Markov-Switching VAR script for estimating regime-dependent dynamics.
- `posterior.py`: Uses regime results to examine state classification and asymmetry.
- `relationship.py`: Compares the volume-volatility relationship across different states and visualizes it.
- `lag_eff.py`: Prints the effect of lagged variables based on estimated coefficients.
- `practice.py`: Empty practice file kept in the folder.

## Generated Files

- `Figure_1.png`: Summary chart comparing the relationship between trading volume and volatility across market states.
- `log_volume_vs_return.png`: Scatter plot of log trading volume and return.
- `log_volume_vs_volatility.png`: Scatter plot of log trading volume and volatility.
- `return_vs_log_volume.png`: Scatter plot of return and log trading volume.
- `return_vs_volatility.png`: Scatter plot of return and volatility.
- `volatility_vs_log_volume.png`: Scatter plot of volatility and log trading volume.
- `volatility_vs_return.png`: Scatter plot of volatility and return.

## Data Files

- `btc.csv`: BTC raw data file, not included in version control.
- `eth.csv`: ETH raw data file, not included in version control.
