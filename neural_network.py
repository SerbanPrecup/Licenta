import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import functions as f


data = f.select_file()

input_columns, output_columns = f.select_data(data)

input_data = data[input_columns]
output_data = data[output_columns]

train_percent = int(input("Procentaj date antrenare: "))
train_input, train_output, test_input, test_output = f.shuffle_division(data, train_percent, input_columns,
                                                                        output_columns)
train_input_scaled, train_output_scaled, test_input_scaled, test_output_scaled = f.normalize_min_max(train_input,
                                                                                                     train_output,
                                                                                                     test_input,
                                                                                                     test_output)

train_input_standardizat, train_output_standardizat, test_input_standardizat, test_output_standardizat = f.normalize_standard(
    train_input,
    train_output,
    test_input,
    test_output)

model = Sequential()
model.add(Dense(8, input_dim=3, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.fit(train_input_scaled,train_output_scaled, epochs=30, batch_size=1, verbose=1)
predictions = model.predict(test_input_scaled)
# predicted_label = (prediction > 0.5).astype(int)


print("\nRegresie Liniara (min-max norm):")
mse_scaled,mae_scaled,rmse_scaled,r2_scaled,mape_scaled = f.calculate_metrics(test_output_scaled,predictions)


model_standard = Sequential()
model_standard.add(Dense(8, input_dim=3, activation='relu'))
model_standard.add(Dense(2, activation='linear'))
model_standard.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model_standard.fit(train_input_standardizat,train_output_standardizat, epochs=30, batch_size=1, verbose=1)
predictions = model_standard.predict(test_input_standardizat)
# predicted_label = (prediction > 0.5).astype(int)


print("\nRegresie Liniara (min-max norm):")
mse_standard,mae_standard,rmse_standard,r2_standard,mape_standard = (f.calculate_metrics(test_output_standardizat,predictions ))


