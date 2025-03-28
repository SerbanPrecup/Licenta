import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import functions as f


def antrenare(neuronsHL, LR, nrEpoci, train_input_scaled, train_output_scaled):
    vectorErori = []
    nrNeuronsHL = neuronsHL
    LR = LR
    nrEpoci = nrEpoci

    HL = np.random.uniform(-0.1, 0.1, size=(train_input_scaled.shape[1], nrNeuronsHL))
    OUT = np.random.uniform(-0.1, 0.1, size=(nrNeuronsHL, train_output_scaled.shape[1]))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivat(x):
        return sigmoid(x) * (1 - sigmoid(x))

    for e in range(nrEpoci):
        all_predictions = []
        all_targets = []

        for idx in range(len(train_input_scaled)):
            linie = train_input_scaled[idx]
            target = train_output_scaled[idx]

            gInHL = np.dot(linie, HL)
            actHL = sigmoid(gInHL)
            gInOut = np.dot(actHL, OUT)
            actOut = sigmoid(gInOut)

            all_predictions.append(actOut)
            all_targets.append(target)

            deltaOUT = 2 * (actOut - target) * sigmoid_derivat(gInOut)
            delta_hidden = np.dot(deltaOUT, OUT.T) * sigmoid_derivat(gInHL)

            OUT -= LR * np.outer(actHL, deltaOUT)
            HL -= LR * np.outer(linie, delta_hidden)

        mse = mean_squared_error(all_targets, all_predictions)
        vectorErori.append(mse)

    return HL, OUT, vectorErori


def testare(test_input_scaled, test_output_scaled, HL, OUT):
    all_predictions = []
    all_targets = []

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for idx in range(len(test_input_scaled)):
        linie = test_input_scaled[idx]
        target = test_output_scaled[idx]

        gInHL = np.dot(linie, HL)
        actHL = sigmoid(gInHL)
        gInOut = np.dot(actHL, OUT)
        actOut = sigmoid(gInOut)

        all_predictions.append(actOut)
        all_targets.append(target)

    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)

    return mse, mae, rmse, r2


data = f.select_file()
input_columns, output_columns = f.select_data(data)
train_percent = int(input("Procentaj date antrenare: "))

train_input, train_output, test_input, test_output = f.shuffle_division(data, train_percent, input_columns, output_columns)
train_input_scaled, train_output_scaled, test_input_scaled, test_output_scaled = f.normalize_min_max(train_input, train_output, test_input, test_output)

HL, OUT, erori_antrenare = antrenare(
    neuronsHL=15,
    LR=0.01,
    nrEpoci=200,
    train_input_scaled=train_input_scaled,
    train_output_scaled=train_output_scaled
)

mse, mae, rmse, r2 = testare(
    test_input_scaled=test_input_scaled,
    test_output_scaled=test_output_scaled,
    HL=HL,
    OUT=OUT
)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)
