import pandas as pd
import matplotlib.pyplot as plt

# Leer archivo csv
df = pd.read_csv('bitcoinprices.csv')

# Convertir la columna "Date" a formato de fecha
df['Date'] = pd.to_datetime(df['Date'])

# Establecer la columna "Date" como índice del DataFrame
df.set_index('Date', inplace=True)

# Convertir la columna "Price" a tipo float
df['Price'] = df['Price'].str.replace(',', '').astype(float)

# Graficar los precios de bitcoin
plt.plot(df['Price'])
plt.title('Precios históricos de Bitcoin')
plt.xlabel('Fecha')
plt.ylabel('Precio (USD)')
plt.show()