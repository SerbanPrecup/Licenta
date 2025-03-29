from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import functions as f
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

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

regr1 = DecisionTreeRegressor(max_depth=2)
regr2 = DecisionTreeRegressor(max_depth=5)
regr1.fit(train_input_scaled, train_output_scaled)
regr2.fit(train_input_scaled, train_output_scaled)

predictions = regr1.predict(test_input_scaled)
predictions2 = regr2.predict(test_input_scaled)

print("\nDecision Tree Regression (min-max norm):")
print("->max_depth=2")
mse_scaled,mae_scaled,rmse_scaled,r2_scaled,mape_scaled = f.calculate_metrics(test_output_scaled,predictions)

print("\nDecision Tree Regression (min-max norm):")
print("->max_depth=5")
mse_scaled,mae_scaled,rmse_scaled,r2_scaled,mape_scaled = f.calculate_metrics(test_output_scaled,predictions)


regr1_standard = DecisionTreeRegressor(max_depth=2)
regr2_standard = DecisionTreeRegressor(max_depth=5)
regr1_standard.fit(train_input_standardizat, train_output_standardizat)
regr2_standard.fit(train_input_standardizat, train_output_standardizat)

predictions = regr1_standard.predict(test_input_standardizat)
predictions2 = regr2_standard.predict(test_input_standardizat)

print("\nDecision Tree Regression (standard norm):")
print("->max_depth=2")
mse_standard,mae_standard,rmse_standard,r2_standard,mape_standard = (f.calculate_metrics(test_output_standardizat,predictions ))


print("\nDecision Tree Regression (standard norm):")
print("->max_depth=5")
mse_standard,mae_standard,rmse_standard,r2_standard,mape_standard = (f.calculate_metrics(test_output_standardizat,predictions ))

