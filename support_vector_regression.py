from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import functions as f
import numpy as np
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

# SVR folosind mai multe coloane de output

svr = MultiOutputRegressor(SVR(kernel='rbf'))
svr.fit(train_input_scaled, train_output_scaled)
predictions = svr.predict(test_input_scaled)


print("\nSupport Vector Regression (min-max norm):")
mse_scaled,mae_scaled,rmse_scaled,r2_scaled,mape_scaled = f.calculate_metrics(test_output_scaled,predictions)


svr_standard = MultiOutputRegressor(SVR(kernel='rbf'))
svr_standard.fit(train_input_standardizat, train_output_standardizat)
predictions = svr_standard.predict(test_input_standardizat)

print("\nSupport Vector Regression (standard norm):")
mse_standard,mae_standard,rmse_standard,r2_standard,mape_standard = (f.calculate_metrics(test_output_standardizat,predictions ))

# SVR folosit pe fiecare coloana de output separat

svr_fthg = SVR(kernel='rbf')
svr_ftag = SVR(kernel='rbf')

svr_fthg.fit(train_input_scaled, train_output_scaled[:, 0])
svr_ftag.fit(train_input_scaled, train_output_scaled[:, 1])

predictions_fthg = svr_fthg.predict(test_input_scaled)
predictions_ftag = svr_ftag.predict(test_input_scaled)

predictions = np.column_stack((predictions_fthg, predictions_ftag))


print("\n\nSupport Vector Regression pt fiecare coloana din output separat")
print("Support Vector Regression (min-max norm):")
mse_scaled,mae_scaled,rmse_scaled,r2_scaled,mape_scaled = f.calculate_metrics(test_output_scaled,predictions)


svr_fthg_standard = SVR(kernel='rbf')
svr_ftag_standard = SVR(kernel='rbf')

svr_fthg_standard.fit(train_input_standardizat, train_output_standardizat[:, 0])
svr_ftag_standard.fit(train_input_standardizat, train_output_standardizat[:, 1])

predictions_fthg_standard = svr_fthg_standard.predict(test_input_standardizat)
predictions_ftag_standard = svr_ftag_standard.predict(test_input_standardizat)

predictions = np.column_stack((predictions_fthg_standard, predictions_ftag_standard))


print("\nSupport Vector Regression (standard norm):")
mse_standard,mae_standard,rmse_standard,r2_standard,mape_standard = (f.calculate_metrics(test_output_standardizat,predictions ))

