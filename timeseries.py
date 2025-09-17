# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import (ARIMA, ARMA, SimpleExpSmoothing, Holt,
                                 ExponentialSmoothing, VAR, VARMAX)
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------
# 1. SINGLE data source --------------------------------------------------
# ------------------------------------------------------------
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
df = df.asfreq('MS')                # monthly start
y = df['Passengers']
print('Data shape:', y.shape)

# helper: quick in-sample RMSE
def rmse(fit, y_true):
    return np.sqrt(mean_squared_error(y_true, fit.fittedvalues))

# ------------------------------------------------------------
# 2. UNIVARIATE models ---------------------------------------------------
# ------------------------------------------------------------
results = {}  # store AIC / RMSE

# 2.1 AutoReg (AR)
ar_model = AutoReg(y, lags=2).fit()
results['AutoReg'] = {'AIC': ar_model.aic, 'RMSE': rmse(ar_model, y)}

# 2.2 Moving-Average (MA) – invert to ARMA(0,1)
ma_model = ARMA(y, order=(0, 1)).fit()
results['MA'] = {'AIC': ma_model.aic, 'RMSE': rmse(ma_model, y)}

# 2.3 ARMA
arma_model = ARMA(y, order=(2, 1)).fit()
results['ARMA'] = {'AIC': arma_model.aic, 'RMSE': rmse(arma_model, y)}

# 2.4 ARIMA
arima_model = ARIMA(y, order=(2, 1, 1)).fit()
results['ARIMA'] = {'AIC': arima_model.aic, 'RMSE': rmse(arima_model, y)}

# 2.5 SARIMA (1,1,1)(1,0,1,12)
sarima_model = ARIMA(y, order=(1, 1, 1),
                     seasonal_order=(1, 0, 1, 12)).fit()
results['SARIMA'] = {'AIC': sarima_model.aic, 'RMSE': rmse(sarima_model, y)}

# 2.6 SARIMAX – same SARIMA but with time as exog
time = np.arange(len(y))
sarimax_model = ARIMA(y, exog=time, order=(1, 1, 1),
                      seasonal_order=(1, 0, 1, 12)).fit()
results['SARIMAX'] = {'AIC': sarimax_model.aic, 'RMSE': rmse(sarimax_model, y)}

# 2.7 Simple Exponential Smoothing (SES)
ses_model = SimpleExpSmoothing(y).fit()
results['SES'] = {'AIC': ses_model.aic, 'RMSE': rmse(ses_model, y)}

# 2.8 Holt-Winters (multiplicative seasonality)
holt_model = HWES(y, seasonal_periods=12, trend='add', seasonal='mul').fit()
results['HoltWinters'] = {'AIC': holt_model.aic, 'RMSE': rmse(holt_model, y)}

# ------------------------------------------------------------
# 3. MULTIVARIATE models -------------------------------------------------
#    VAR / VARMA / VARMAX  (treat lag-1 & lag-2 as extra series)
# ------------------------------------------------------------
# build a 3-column frame: original, lag1, lag2
data = pd.DataFrame({'y': y})
data['y_l1'] = data['y'].shift(1)
data['y_l2'] = data['y'].shift(2)
data = data.dropna()

# VAR
var_model = VAR(data).fit(maxlags=1)   # lag-order 1
results['VAR'] = {'AIC': var_model.aic, 'RMSE': np.sqrt(var_model.mse)}

# VARMA
varma_model = VARMAX(data, order=(1, 1)).fit(disp=False)
results['VARMA'] = {'AIC': varma_model.aic,
                    'RMSE': np.sqrt(varma_model.mse)}

# VARMAX – add time as exogenous
varmax_model = VARMAX(data, exog=data.index.month.values.reshape(-1, 1),
                      order=(1, 1)).fit(disp=False)
results['VARMAX'] = {'AIC': varmax_model.aic,
                     'RMSE': np.sqrt(varmax_model.mse)}

# ------------------------------------------------------------
# 4. Summary table -------------------------------------------------------
# ------------------------------------------------------------
summary = pd.DataFrame(results).T
summary = summary.sort_values('AIC')
print('\n=== Model comparison (lower AIC / RMSE is better) ===')
print(summary.round(3))
