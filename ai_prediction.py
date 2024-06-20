import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
# Load the data
file_path = 'stock_data.csv'
data = pd.read_csv(file_path)

# Define features and target (Indicators and Close price)
# USE INDICATORS OR NOT

features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD'] #Use indicators
#features = ['Close', 'Open', 'High', 'Low'] #Do not use indicators
target = 'Close'

# Normalize the features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + [target]])

# Create sequences of past day data to predict the next day
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and testing sets without shuffling (ordered)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

#Flatten the sequences for SVR and Linear Regression
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

#Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train)#Fit the model
lr_predictions = lr_model.predict(X_test_flat)#Predict the model
lr_next_day_pred = lr_model.predict(X_test_flat[-1].reshape(1, -1))[0]#Predict the next day
lr_mse = mean_squared_error(y_test, lr_predictions)#Mean Squared Error

# SVR Model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_flat, y_train)#Fit the model
svr_predictions = svr_model.predict(X_test_flat)#Predict the model
svr_next_day_pred = svr_model.predict(X_test_flat[-1].reshape(1, -1))[0]#Predict the next day
svr_mse = mean_squared_error(y_test, svr_predictions)#Mean Squared Error

# Reshape data for LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))#Reshape the data for LSTM
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))#Reshape the data for LSTM

# LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=2)#Fit the model

lstm_predictions = lstm_model.predict(X_test_lstm)#Predictions of model
lstm_next_day_pred = lstm_model.predict(X_test_lstm[-1].reshape(1, X_test_lstm.shape[1], X_test_lstm.shape[2]))[0][0]#Predict the next day
lstm_mse = mean_squared_error(y_test, lstm_predictions)#Mean Squared Error
print()
print(f'LR Mean Squared Error: {lr_mse}')#Print the Mean Squared Error
print(f'SVR Mean Squared Error: {svr_mse}')
print(f'LSTM Mean Squared Error: {lstm_mse}')

# Inverse transform the predictions and actual values to get the real prices
scaler_target = MinMaxScaler()
scaler_target.fit(data[[target]])

y_test_inverse = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
lr_predictions_inverse = scaler_target.inverse_transform(lr_predictions.reshape(-1, 1)).flatten()
svr_predictions_inverse = scaler_target.inverse_transform(svr_predictions.reshape(-1, 1)).flatten()
lstm_predictions_inverse = scaler_target.inverse_transform(lstm_predictions).flatten()

# Inverse transform the next day predictions
lr_next_day_pred_inverse = scaler_target.inverse_transform([[lr_next_day_pred]])[0, 0]
svr_next_day_pred_inverse = scaler_target.inverse_transform([[svr_next_day_pred]])[0, 0]
lstm_next_day_pred_inverse = scaler_target.inverse_transform([[lstm_next_day_pred]])[0, 0]

today_price = y_test_inverse[-2]#Today price
next_day_real_price = y_test_inverse[-1]#Next day price

print()
print("Today price: ", today_price)
print("Next Day price: ", next_day_real_price)
print()
print(f'LR Prediction: {lr_next_day_pred_inverse}')
print(f'SVR Next Day Prediction: {svr_next_day_pred_inverse}')
print(f'LSTM Next Day Prediction: {lstm_next_day_pred_inverse}')
print()

# Calculate the percentage difference
def percentage_difference(actual, predicted):
    return ((predicted - actual) / actual) * 100

lr_percentage_diff = percentage_difference(y_test_inverse, lr_predictions_inverse)
svr_percentage_diff = percentage_difference(y_test_inverse, svr_predictions_inverse)
lstm_percentage_diff = percentage_difference(y_test_inverse, lstm_predictions_inverse)

# Plot percentage differences
plt.figure(figsize=(14, 7))

plt.subplot(2, 2, 1)
plt.plot(lr_percentage_diff, label='LR Percentage Difference')
plt.title('Linear Regression Percentage Difference')
plt.xlabel('Days')
plt.ylabel('Percentage Difference')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(svr_percentage_diff, label='SVR Percentage Difference')
plt.title('SVR Percentage Difference')
plt.xlabel('Days')
plt.ylabel('Percentage Difference')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(lstm_percentage_diff, label='LSTM Percentage Difference')
plt.title('LSTM Percentage Difference')
plt.xlabel('Days')
plt.ylabel('Percentage Difference')
plt.legend()

# Plot the mean squared errors
plt.subplot(2, 2, 4)
models = ['Linear Regression', 'SVR', 'LSTM']
mses = [lr_mse, svr_mse, lstm_mse]
plt.bar(models, mses)
plt.title('Mean Squared Error Comparison')
plt.ylabel('Mean Squared Error')

plt.tight_layout()
plt.show()

# Plot next day predictions actual values with annotations
plt.figure(figsize=(10, 6))
labels = ['Today', 'Real Next Day', 'LR Prediction', 'SVR Prediction', 'LSTM Prediction']
values = [today_price, next_day_real_price, lr_next_day_pred_inverse, svr_next_day_pred_inverse, lstm_next_day_pred_inverse]

plt.bar(labels, values, color=['blue', 'green', 'red', 'orange', 'purple'])

for i, v in enumerate(values):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

plt.title('Next Day Predictions Actual Values')
plt.ylabel('Price')
plt.show()
