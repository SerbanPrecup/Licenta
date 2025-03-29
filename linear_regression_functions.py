import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class LinearRegressionFromZero:
    def __init__(self):
        self.parameters = {}

    def forward_propagation(self, train_input):
        return np.dot(train_input, self.parameters['m']) + self.parameters['c']

    def cost_function(self, predictions, train_output):
        return np.mean((train_output - predictions) ** 2)

    def backward_propagation(self, train_input, train_output, predictions):
        df = predictions - train_output
        dm = 2 * np.dot(train_input.T, df) / len(train_input)
        dc = 2 * np.mean(df)
        return {'dm': dm, 'dc': dc}

    def update_parameters(self, derivatives, learning_rate):
        self.parameters['m'] -= learning_rate * derivatives['dm']
        self.parameters['c'] -= learning_rate * derivatives['dc']

    def train(self, train_input, train_output, learning_rate, iters):
        num_features = train_input.shape[1]
        num_outputs = train_output.shape[1]

        self.parameters['m'] = np.random.uniform(-1, 1, size=(num_features, num_outputs))
        self.parameters['c'] = np.random.uniform(-1, 1, size=(num_outputs,))

        self.loss = []
        for i in range(iters):
            predictions = self.forward_propagation(train_input)
            cost = self.cost_function(predictions, train_output)
            derivatives = self.backward_propagation(train_input, train_output, predictions)
            self.update_parameters(derivatives, learning_rate)

            self.loss.append(cost)
            if i % 50 == 0:
                print(f"Iteration {i + 1}/{iters} - Loss: {cost:.6f}")

        return self.parameters, self.loss

    def predict(self, test_input):
        return self.forward_propagation(test_input)

    def evaluate(self, test_input, test_output):
        predictions = self.predict(test_input)
        mse = mean_squared_error(test_output, predictions)
        mae = mean_absolute_error(test_output, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_output, predictions)

        print(f"\nEvaluation Metrics:")
        print(f"MSE  (Mean Squared Error): {mse:.6f}")
        print(f"MAE  (Mean Absolute Error): {mae:.6f}")
        print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
        print(f"R²   (R-squared): {r2:.6f}")

        return mse,mae,rmse,r2