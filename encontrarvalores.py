import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools

# Leer datos
data = pd.read_csv('bitcoinprices.csv', parse_dates=['Date'], index_col='Date')
prices = data['Price'].str.replace(',', '').astype(float)

# Función para calcular el orden óptimo de diferenciación (d)
def find_optimal_d(series):
    result = adfuller(series)
    for d in range(1, 4):
        diff = np.diff(series, n=d)
        new_result = adfuller(diff)
        if new_result[1] < 0.05:
            return d
    return 0

# Función para calcular los valores óptimos de p y q
def find_optimal_pq(series, p_max, q_max, d):
    best_pq = (0, 0)
    best_mse = float('inf')
    
    for p, q in itertools.product(range(p_max + 1), range(q_max + 1)):
        try:
            model = ARIMA(series, order=(p, d, q))
            results = model.fit()
            mse = mean_squared_error(series[d:], results.fittedvalues[d:])
            if mse < best_mse:
                best_pq = (p, q)
                best_mse = mse
        except:
            continue
            
    return best_pq

# Encontrar valores óptimos de p, d y q
d = find_optimal_d(prices)
p, q = find_optimal_pq(prices, p_max=5, q_max=5, d=d)

print(f"Valores óptimos: p = {p}, d = {d}, q = {q}")