from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import functions as f
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

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


# svr = SVR(kernel='rbf')

# svr.fit(train_input_scaled, train_output_scaled)

# predictions = svr.predict(test_input_scaled)


# SVR folosind mai multe coloane de output

svr = MultiOutputRegressor(SVR(kernel='rbf'))
svr.fit(train_input_scaled, train_output_scaled)
predictions = svr.predict(test_input_scaled)

mse = mean_squared_error(test_output_scaled, predictions)
mae = mean_absolute_error(test_output_scaled, predictions)
rmse = root_mean_squared_error(test_output_scaled, predictions)
r2 = r2_score(test_output_scaled, predictions)

print("\nSupport Vector Regression (min-max norm):")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)

svr_standard = MultiOutputRegressor(SVR(kernel='rbf'))
svr_standard.fit(train_input_standardizat, train_output_standardizat)
predictions = svr_standard.predict(test_input_standardizat)

mse = mean_squared_error(test_output_standardizat, predictions)
mae = mean_absolute_error(test_output_standardizat, predictions)
rmse = root_mean_squared_error(test_output_standardizat, predictions)
r2 = r2_score(test_output_standardizat, predictions)

print("\nSupport Vector Regression (standard norm):")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)


# SVR folosit pe fiecare coloana de output separat

svr_fthg = SVR(kernel='rbf')
svr_ftag = SVR(kernel='rbf')

svr_fthg.fit(train_input_scaled, train_output_scaled[:, 0])
svr_ftag.fit(train_input_scaled, train_output_scaled[:, 1])

predictions_fthg = svr_fthg.predict(test_input_scaled)
predictions_ftag = svr_ftag.predict(test_input_scaled)

predictions = np.column_stack((predictions_fthg, predictions_ftag))

mse = mean_squared_error(test_output_scaled, predictions)
mae = mean_absolute_error(test_output_scaled, predictions)
rmse = root_mean_squared_error(test_output_scaled, predictions)
r2 = r2_score(test_output_scaled, predictions)

print("\nSupport Vector Regression (min-max norm):")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)


svr_fthg_standard = SVR(kernel='rbf')
svr_ftag_standard = SVR(kernel='rbf')

svr_fthg_standard.fit(train_input_standardizat, train_output_standardizat[:, 0])
svr_ftag_standard.fit(train_input_standardizat, train_output_standardizat[:, 1])


predictions_fthg_standard = svr_fthg_standard.predict(test_input_standardizat)
predictions_ftag_standard = svr_ftag_standard.predict(test_input_standardizat)

predictions = np.column_stack((predictions_fthg_standard, predictions_ftag_standard))

mse = mean_squared_error(test_output_standardizat, predictions)
mae = mean_absolute_error(test_output_standardizat, predictions)
rmse = root_mean_squared_error(test_output_standardizat, predictions)
r2 = r2_score(test_output_standardizat, predictions)

print("\nSupport Vector Regression (standard norm):")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)
