import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def antrenare(neuronsHL, LR, nrEpoci, train_input, train_output):
    vectorErori = []
    nrNeuronsHL = neuronsHL
    LR = LR
    nrEpoci = nrEpoci

    HL = np.random.uniform(-0.1, 0.1, size=(train_input.shape[1], nrNeuronsHL))
    OUT = np.random.uniform(-0.1, 0.1, size=(nrNeuronsHL, train_output.shape[1]))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivat(x):
        return sigmoid(x) * (1 - sigmoid(x))

    for e in range(nrEpoci):
        all_predictions = []
        all_targets = []

        for idx in range(len(train_input)):
            linie = train_input[idx]
            target = train_output[idx]

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


def testare(test_input, test_output, HL, OUT):
    all_predictions = []
    all_targets = []

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for idx in range(len(test_input)):
        linie = test_input[idx]
        target = test_output[idx]

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