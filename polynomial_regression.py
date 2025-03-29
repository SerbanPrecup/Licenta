from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

import functions as f
from sklearn.preprocessing import PolynomialFeatures


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


poly = PolynomialFeatures(degree=4)
train_input_poly = poly.fit_transform(train_input_scaled)
test_input_poly = poly.fit_transform(test_input_scaled)

lin = LinearRegression()
lin.fit(train_input_poly, train_output_scaled)
predictions_poly = lin.predict(test_input_poly)


print("\nRegresie Polinomiala (min-max norm):")
mse_scaled,mae_scaled,rmse_scaled,r2_scaled,mape_scaled = f.calculate_metrics(test_output_scaled,predictions_poly)


poly_standard = PolynomialFeatures(degree=4)
train_input_poly_standard = poly_standard.fit_transform(train_input_standardizat)
test_input_poly_standard = poly_standard.transform(test_input_standardizat)


lin_standard = LinearRegression()
lin_standard.fit(train_input_poly_standard, train_output_standardizat)
predictions_poly_standard = lin_standard.predict(test_input_poly_standard)


print("\nRegresie Polinomiala (Standard norm):")
mse_standard,mae_standard,rmse_standard,r2_standard,mape_standard = (f.calculate_metrics(test_output_standardizat,predictions_poly_standard ))


