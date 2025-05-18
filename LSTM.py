import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


EPOCHS = 5
future_steps = 15
time_step = 15 

file_path = 'lstm_input.csv'
data = pd.read_csv(file_path)

data['DATE'] = pd.to_datetime(data['DATE'], format='%d/%m/%Y')
data = data.sort_values('DATE')
data.set_index('DATE', inplace=True)

river_levels = data[['LEVEL']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(river_levels)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

val_size = int(len(X_train) * 0.2)
X_val, y_val = X_train[-val_size:], y_train[-val_size:]
X_train, y_train = X_train[:-val_size], y_train[:-val_size]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


try:
    model = load_model('modelo_lstm.h5')
    print("Modelo carregado com sucesso.")
except:
    print("Modelo não encontrado. Treinando um novo modelo.")
    
    model = Sequential()
    model.add(LSTM(30, return_sequences=True, input_shape=(time_step, 1), 
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(LSTM(30, return_sequences=False, 
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              batch_size=1, 
              epochs=EPOCHS, 
              callbacks=[early_stopping])
    
    model.save('modelo_lstm.h5')

train_predict = model.predict(X_train)
val_predict = model.predict(X_val)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
val_predict = scaler.inverse_transform(val_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_val = scaler.inverse_transform([y_val])
y_test = scaler.inverse_transform([y_test])

last_time_step = scaled_data[-time_step:]
last_time_step = last_time_step.reshape((1, time_step, 1))

future_predictions = []
for _ in range(future_steps):
    next_pred = model.predict(last_time_step)
    future_predictions.append(next_pred[0, 0])
    next_time_step = np.concatenate((last_time_step[:, 1:, :], next_pred.reshape(1, 1, 1)), axis=1)
    last_time_step = next_time_step

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

plt.figure(figsize=(12, 6))

plt.plot(data.index, scaler.inverse_transform(scaled_data), label='Dados Originais', color='blue')

train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, :] = train_predict
plt.plot(data.index[:len(train_plot)], train_plot, label='Previsão Treinamento', color='green')

val_plot = np.empty_like(scaled_data)
val_plot[:, :] = np.nan
val_start_index = len(train_predict) + time_step
val_end_index = val_start_index + len(val_predict)
val_plot[val_start_index:val_end_index, :] = val_predict
plt.plot(data.index[val_start_index:val_end_index], val_plot[val_start_index:val_end_index], label='Previsão Validação', color='purple')

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_start_index = val_end_index
test_end_index = test_start_index + len(test_predict)
test_plot[test_start_index:test_end_index, :] = test_predict
plt.plot(data.index[test_start_index:test_end_index], test_plot[test_start_index:test_end_index], label='Previsão Teste', color='red')

future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps)
plt.plot(future_dates, future_predictions, label='Previsão Futuro', color='orange')

plt.xlabel('Data')
plt.ylabel('Nível do Rio')
plt.title('Previsão do Nível do Rio com LSTM')
plt.legend()
#plt.show()
plt.savefig('previsao_lstm.png')
