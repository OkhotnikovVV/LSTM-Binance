import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

#Получение данных о котировках криптовалют с использованием API Binance
url = 'https://api.binance.com/api/v3/klines'
params = {
'symbol': 'BTCUSDT',  # символ криптовалютной пары
'interval': '1m',  # интервал времени свечного графика
'limit': 1000  # число свечей
}
response = requests.get(url, params=params)
data = response.json()

#Сохранение полученных данных в CSV-файл
df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
'Taker buy quote asset volume', 'Ignore'])
df.to_csv('btcusdt_1m.csv', index=False)

#Загрузка CSV-файла с данными
df = pd.read_csv('btcusdt_1m.csv')

#Предварительная обработка данных
df.drop(['Open time', 'Close time', 'Quote asset volume', 'Number of trades',
'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], axis=1, inplace=True)
df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

#Масштабирование данных
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])

#Разделение данных на обучающую и тестовую выборки
train_data = scaled_data[:800]
test_data = scaled_data[800:]

#Создание функции, которая преобразует данные в нужный формат для LSTM-модели
def create_dataset(data, look_back=60):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:i+look_back])
        Y.append(data[i+look_back])
    return np.array(X), np.array(Y)

#Преобразование данных в нужный формат для LSTM-модели
look_back = 60
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

#Создание LSTM-модели
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=5))
model.compile(optimizer='adam', loss='mean_squared_error')

#Обучение модели
model.fit(X_train, Y_train, epochs=50, batch_size=32)

#Предсказание моделью значений на тестовой выборке
predicted_data = model.predict(X_test)

#Обратное масштабирование данных
predicted_data = scaler.inverse_transform(predicted_data)
Y_test = scaler.inverse_transform(Y_test)

#Оценка качества модели
mse = np.mean(np.square(Y_test - predicted_data))
mae = np.mean(np.abs(Y_test - predicted_data))
print('MSE: {}'.format(mse))
print('MAE: {}'.format(mae))

#Визуализация предсказаний модели на графике
plt.plot(Y_test[:, 0], label='Actual price')
plt.plot(predicted_data[:, 0], label='Predicted price')
plt.legend()
plt.show()