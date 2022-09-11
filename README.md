# forecasting-SPY
Forecasting future behavior of SPY with ARIMA in R.

## Objective
Predicting the future price of a security is a ubiquitous and fundamental endeavor in financial fields, in which trades and other actions are often taken on the basis of how the market is believed to evolve in the future. The rewards are obvious – windfall profits potentially await those whose predictions significantly outpace those of a coin toss. So too are the risks posed by poor predictive power. 

Of the many techniques successfully used in time series forecasting, autoregressive integrated moving average (ARIMA) models are a popular choice. ARIMA combines autoregression with moving averages to forecast the future. In contrast to other techniques like dynamic regression, ARIMA predicts future values of a variable based on prior values of that variable. In other words, a multitude of different features are not supplied to the model. 

This repository implements ARIMA models to forecast future security behavior. The specific security chosen is the SPDR S&P 500 trust exchange-traded fund (ETF), whose ticker is ‘SPY’. SPY is chosen for its low volatility relative to other securities and for its potential amenability to prediction; as SPY indexes the overall S&P 500 stock market, it seems reasonable that its future behavior may be more easily forecastable than that of an individual security, such as a single holding in SPY. 

This repository provides a complementary forecasting technique to the other available repository branch, which implements an LSTM-based approach to forecasting SPY in TensorFlow. Here, we implement ARIMA using R via Jupyter Notebook in `ARIMA.ipynb`.


