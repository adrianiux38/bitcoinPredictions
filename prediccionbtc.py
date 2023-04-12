import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Leer datos
data = pd.read_csv('bitcoinprices.csv', parse_dates=['Date'], index_col='Date')
data.sort_index(inplace=True)  # Ordenar el índice de fechas
prices = data['Price'].str.replace(',', '').astype(float)

# Establecer la frecuencia del índice de fechas
prices = prices.asfreq('D')

# Crear y ajustar el modelo ARIMA
model = ARIMA(prices, order=(5, 1, 5))
results = model.fit()

# Realizar la predicción para un día después del último registro
forecast = results.forecast(steps=1)
prediction = forecast.iloc[0]

print(f"La predicción para mañana es: {prediction:.2f}")