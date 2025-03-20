from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

import functions as f
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline


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


model = make_pipeline(
    SplineTransformer(n_knots=5, degree=3),
    MultiOutputRegressor(PoissonRegressor(alpha=0.01))
)

model.fit(train_input_scaled, train_output_scaled)
predictions = model.predict(test_input_scaled)


print("\nRegresie Poisson (min-max norm):")
mse_scaled,mae_scaled,rmse_scaled,r2_scaled,mape_scaled = f.calculate_metrics(test_output_scaled,predictions)


model_standard = make_pipeline(
    SplineTransformer(n_knots=5, degree=3),
    MultiOutputRegressor(PoissonRegressor(alpha=0.01))
)

# train_output_standardizat are valori cu minus care nu sunt permise de PoissonRegression
model_standard.fit(train_input_standardizat, train_output_scaled)
predictions = model_standard.predict(test_input_standardizat)

print("\nRegresie Poisson (standard norm):")
mse_standard,mae_standard,rmse_standard,r2_standard,mape_standard = (f.calculate_metrics(test_output_standardizat,predictions ))
