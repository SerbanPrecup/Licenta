import numpy as np
from matplotlib import pyplot as plt
from openpyxl.descriptors import Integer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import json
from bayes_opt import BayesianOptimization

import functions as f


def linear_regression(train_input, train_output, test_input, test_output):
    lin = LinearRegression()
    lin.fit(train_input, train_output)
    predictions = lin.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_linear_regression_grid_search(train_input, train_output, test_input, test_output):
    param_grid = {
        'fit_intercept': [True, False],
        'positive': [True, False]
    }

    best_score = np.inf
    best_params = None
    best_metrics = None

    for fit_intercept in param_grid['fit_intercept']:
        for positive in param_grid['positive']:
            model = LinearRegression(
                fit_intercept=fit_intercept,
                positive=positive
            )
            model.fit(train_input, train_output)

            preds = model.predict(test_input)
            metrics = f.calculate_metrics(test_output, preds)
            mse = metrics[0]

            if mse < best_score:
                best_score = mse
                best_params = {
                    'fit_intercept': fit_intercept,
                    'positive': positive
                }
                best_metrics = metrics

    return best_metrics, best_params


def optimized_linear_regression_grid_search_saved(train_input, train_output, test_input, test_output):
    lin = LinearRegression(fit_intercept=True, positive=False)
    lin.fit(train_input, train_output)
    predictions = lin.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def linear_regression_from_zero(train_input, train_output, test_input, test_output, lr=0.02, iters=2000):
    from linear_regression_functions import LinearRegressionFromZero
    lin = LinearRegressionFromZero()
    param, loss = lin.train(train_input, train_output, 0.02, 2000)
    return lin.evaluate(test_input, test_output)


def optimized_linear_regression_from_zero_grid_search(train_input, train_output, test_input, test_output):
    from linear_regression_functions import LinearRegressionFromZero

    param_grid = {
        'learning_rate': [0.001, 0.01, 0.02],
        'iterations': [30, 100, 200, 500]
    }

    best_score = np.inf
    best_params = None
    best_metrics = None

    for lr in param_grid['learning_rate']:
        for iters in param_grid['iterations']:
            lin = LinearRegressionFromZero()
            _, _ = lin.train(train_input, train_output, lr, iters)

            metrics = lin.evaluate(test_input, test_output)
            mse = metrics[0]

            if mse < best_score:
                best_score = mse
                best_params = {
                    'learning_rate': lr,
                    'iterations': iters
                }
                best_metrics = metrics

    return best_metrics, best_params


def optimized_linear_regression_from_zero_grid_search_saved(train_input, train_output, test_input, test_output):
    from linear_regression_functions import LinearRegressionFromZero
    lin = LinearRegressionFromZero()
    param, loss = lin.train(train_input, train_output, 0.01, 200)
    return lin.evaluate(test_input, test_output)


def optimized_linear_regression_from_zero_bayesian(train_input, train_output, test_input, test_output):

    from linear_regression_functions import LinearRegressionFromZero

    def objective_function(learning_rate, iterations):
        lr = max(learning_rate, 1e-5)
        iters = int(round(iterations))

        try:
            lin = LinearRegressionFromZero()
            params, loss = lin.train(train_input, train_output, lr, iters)

            last_loss = loss[-1]

            if not np.isfinite(last_loss):
                return -1e6

            return -last_loss

        except Exception:
            return -1e6

    pbounds = {
        'learning_rate': (0.0001, 0.2),
        'iterations': (30, 500)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )

    optimizer.maximize(init_points=5, n_iter=20)

    best_params = optimizer.max['params']
    best_lr = best_params['learning_rate']
    best_iters = int(round(best_params['iterations']))

    final_model = LinearRegressionFromZero()
    final_model.train(train_input, train_output, best_lr, best_iters)

    metrics = final_model.evaluate(test_input, test_output)
    return metrics, best_params


def polynomial_regression(train_input, train_output, test_input, test_output, degree):
    poly = PolynomialFeatures(degree=degree)
    train_input_poly = poly.fit_transform(train_input)
    test_input_poly = poly.transform(test_input)

    lin = LinearRegression()
    lin.fit(train_input_poly, train_output)
    predictions = lin.predict(test_input_poly)

    return f.calculate_metrics(test_output, predictions)


def optimized_polynomial_regression_grid_search(train_input, train_output, test_input, test_output):
    param_grid = {
        'degree': [1, 2, 3, 4, 5]
    }

    best_score = np.inf
    best_params = None
    best_metrics = None

    for deg in param_grid['degree']:
        poly = PolynomialFeatures(degree=deg)
        X_train_poly = poly.fit_transform(train_input)
        X_test_poly = poly.transform(test_input)

        lin = LinearRegression()
        lin.fit(X_train_poly, train_output)

        preds = lin.predict(X_test_poly)
        metrics = f.calculate_metrics(test_output, preds)
        mse = metrics[0]

        if mse < best_score:
            best_score = mse
            best_params = {'degree': deg}
            best_metrics = metrics

    return best_metrics, best_params


def optimized_polynomial_regression_grid_search_saved(train_input, train_output, test_input, test_output):
    return polynomial_regression(train_input, train_output, test_input, test_output, 1)


def optimized_polynomial_regression_bayesian(train_input, train_output, test_input, test_output, max_degree=10,
                                             n_iter=30, cv=5, random_state=0):
    def objective_function(degree):
        degree = int(round(degree))

        poly = PolynomialFeatures(degree=degree)
        train_poly = poly.fit_transform(train_input)

        test_poly = poly.transform(test_input)

        model = LinearRegression()
        model.fit(train_poly, train_output)

        predictions = model.predict(test_poly)
        metrics = f.calculate_metrics(test_output, predictions)

        return -metrics[0]

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds={"degree": (1, 10)},
        random_state=42,
    )

    optimizer.maximize(n_iter=10, init_points=2)

    best_degree = int(round(optimizer.max['params']["degree"]))

    poly = PolynomialFeatures(degree=best_degree)
    train_poly = poly.fit_transform(train_input)
    test_poly = poly.transform(test_input)

    model = LinearRegression()
    model.fit(train_poly, train_output)
    predictions = model.predict(test_poly)

    return f.calculate_metrics(test_output, predictions), best_degree


def optimized_polynomial_regression_bayesian_saved(train_input, train_output, test_input, test_output):
    return polynomial_regression(train_input, train_output, test_input, test_output, 1)


def poisson_regression(train_input, train_output, test_input, test_output, alpha):
    model = make_pipeline(
        SplineTransformer(n_knots=5, degree=3),
        MultiOutputRegressor(PoissonRegressor(alpha=alpha))
    )
    model.fit(train_input, train_output)
    predictions = model.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_poisson_regression_grid_search(train_input, train_output, test_input, test_output):

    param_grid = {
        'n_knots': [5, 10, 15],
        'degree': [2, 3],
        'alpha': [0.01, 0.1, 1, 10]
    }

    best_score = np.inf
    best_params = None
    best_metrics = None

    for n_knots in param_grid['n_knots']:
        for degree in param_grid['degree']:
            for alpha in param_grid['alpha']:
                model = make_pipeline(
                    SplineTransformer(n_knots=n_knots, degree=degree),
                    MultiOutputRegressor(PoissonRegressor(alpha=alpha))
                )
                model.fit(train_input, train_output)

                preds = model.predict(test_input)
                metrics = f.calculate_metrics(test_output, preds)
                mse = metrics[0]

                if mse < best_score:
                    best_score = mse
                    best_params = {
                        'n_knots': n_knots,
                        'degree': degree,
                        'alpha': alpha
                    }
                    best_metrics = metrics

    return best_metrics, best_params


def optimized_poisson_regression_grid_search_saved(train_input, train_output, test_input, test_output):
    hp = {
        'splinetransformer__n_knots': 10,
        'splinetransformer__degree': 2,
        'multioutputregressor__estimator__alpha': 0.01
    }

    pipe = make_pipeline(
        SplineTransformer(),
        MultiOutputRegressor(PoissonRegressor())
    )

    pipe.set_params(**hp)

    pipe.fit(train_input, train_output)
    predictions = pipe.predict(test_input)

    return f.calculate_metrics(test_output, predictions)


def optimized_poisson_regression_bayesian(train_input, train_output, test_input, test_output):
    def objective_function(alpha, n_knots, degree):
        n_knots = int(round(n_knots))
        degree = int(round(degree))

        model = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=degree),
            MultiOutputRegressor(PoissonRegressor(alpha=alpha)))

        model.fit(train_input, train_output)

        predictions = model.predict(test_input)

        metrics = f.calculate_metrics(test_output, predictions)
        return -metrics[0]

    param_bounds = {
        "alpha": (0.1, 10.0),
        "n_knots": (4, 20),
        "degree": (2, 5)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=42
    )

    optimizer.maximize(n_iter=15, init_points=3)

    best_params = optimizer.max['params']
    best_alpha = best_params["alpha"]
    best_n_knots = int(round(best_params["n_knots"]))
    best_degree = int(round(best_params["degree"]))

    final_model = make_pipeline(
        SplineTransformer(n_knots=best_n_knots, degree=best_degree),
        MultiOutputRegressor(PoissonRegressor(alpha=best_alpha)))
    final_model.fit(train_input, train_output)

    predictions = final_model.predict(test_input)
    final_metrics = f.calculate_metrics(test_output, predictions)

    return final_metrics, best_params


def svr(train_input, train_output, test_input, test_output):
    svr = MultiOutputRegressor(SVR(kernel='rbf'))
    svr.fit(train_input, train_output)
    predictions = svr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_svr_grid_search(train_input, train_output, test_input, test_output):

    param_grid = {
        'kernel': ['rbf', 'linear', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'epsilon': [0.1, 0.2, 0.5],
        'degree': [2, 3]
    }

    best_score = np.inf
    best_params = None
    best_metrics = None

    for kernel in param_grid['kernel']:
        for C in param_grid['C']:
            for gamma in param_grid['gamma']:
                for eps in param_grid['epsilon']:
                    for degree in param_grid['degree']:
                        svr = SVR(
                            kernel=kernel,
                            C=C,
                            gamma=gamma,
                            epsilon=eps,
                            degree=degree
                        )
                        model = MultiOutputRegressor(svr)
                        model.fit(train_input, train_output)

                        preds = model.predict(test_input)
                        metrics = f.calculate_metrics(test_output, preds)
                        mse = metrics[0]

                        if mse < best_score:
                            best_score = mse
                            best_params = {
                                'kernel': kernel,
                                'C': C,
                                'gamma': gamma,
                                'epsilon': eps,
                                'degree': degree
                            }
                            best_metrics = metrics

    return best_metrics, best_params


def optimized_svr_grid_search_saved(train_input, train_output, test_input, test_output):
    svr = MultiOutputRegressor(
        SVR(kernel='rbf', C=1, gamma=1, epsilon=0.1, degree=2)
    )
    svr.fit(train_input, train_output)
    predictions = svr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_svr_bayesian(train_input, train_output, test_input, test_output):
    def objective_function(C, epsilon, gamma):
        model = MultiOutputRegressor(
            SVR(
                kernel='rbf',
                C=abs(C),
                epsilon=abs(epsilon),
                gamma=abs(gamma)
            )
        )

        model.fit(train_input, train_output)

        predictions = model.predict(test_input)

        metrics = f.calculate_metrics(test_output, predictions)
        return -metrics[0]

    param_bounds = {
        "C": (0.1, 100),
        "epsilon": (0.01, 1.0),
        "gamma": (0.001, 10)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=42
    )

    optimizer.maximize(n_iter=15, init_points=5)

    best_params = optimizer.max['params']
    best_C = abs(best_params["C"])
    best_epsilon = abs(best_params["epsilon"])
    best_gamma = abs(best_params["gamma"])

    final_model = MultiOutputRegressor(
        SVR(
            kernel='rbf',
            C=best_C,
            epsilon=best_epsilon,
            gamma=best_gamma
        )
    )
    final_model.fit(train_input, train_output)

    predictions = final_model.predict(test_input)
    final_metrics = f.calculate_metrics(test_output, predictions)

    return final_metrics, best_params

def optimized_svr_bayesian_saved(train_input, train_output, test_input, test_output):
    svr = MultiOutputRegressor(
        SVR(kernel='rbf', C=60, gamma=0.71, epsilon=0.077)
    )
    svr.fit(train_input, train_output)
    predictions = svr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def random_forest(train_input, train_output, test_input, test_output):
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(train_input, train_output)
    predictions = rf_regressor.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def random_forest_features_importances(train_input, train_output, test_input, test_output, feature_names):
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(train_input, train_output)
    importance = rf_regressor.feature_importances_
    plt.bar(range(train_input.shape[1]), importance)
    plt.xticks(range(train_input.shape[1]), feature_names, rotation=90)
    plt.title('Feature Importances - Random Forest')
    plt.show()
    predictions = rf_regressor.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_random_forest_grid_search(train_input, train_output,
                                        test_input, test_output):

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True, False]
    }

    best_score = np.inf
    best_params = None
    best_metrics = None

    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for split in param_grid['min_samples_split']:
                for feat in param_grid['max_features']:
                    for boot in param_grid['bootstrap']:
                        model = RandomForestRegressor(
                            n_estimators=n_est,
                            max_depth=depth,
                            min_samples_split=split,
                            max_features=feat,
                            bootstrap=boot,
                            random_state=42,
                            n_jobs=-1
                        )
                        model.fit(train_input, train_output)

                        preds = model.predict(test_input)
                        metrics = f.calculate_metrics(test_output, preds)
                        mse = metrics[0]

                        if mse < best_score:
                            best_score = mse
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'min_samples_split': split,
                                'max_features': feat,
                                'bootstrap': boot
                            }
                            best_metrics = metrics

    return best_metrics, best_params


def optimized_random_forest_grid_search_saved(train_input, train_output, test_input, test_output):
    rf_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    rf_regressor.fit(train_input, train_output)
    predictions = rf_regressor.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_random_forest_bayesian(train_input, train_output, test_input, test_output):
    def objective_function(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        params = {
            "n_estimators": int(round(n_estimators)),
            "max_depth": int(round(max_depth)),
            "min_samples_split": int(round(min_samples_split)),
            "min_samples_leaf": int(round(min_samples_leaf)),
            "max_features": max_features
        }

        model = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1
        )

        model.fit(train_input, train_output)

        predictions = model.predict(test_input)

        metrics = f.calculate_metrics(test_output, predictions)
        return -metrics[0]

    param_bounds = {
        "n_estimators": (50, 500),
        "max_depth": (5, 50),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": (0.1, 1.0)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=42
    )

    optimizer.maximize(n_iter=20, init_points=5)

    best_params = optimizer.max['params']
    best_params = {
        "n_estimators": int(round(best_params["n_estimators"])),
        "max_depth": int(round(best_params["max_depth"])),
        "min_samples_split": int(round(best_params["min_samples_split"])),
        "min_samples_leaf": int(round(best_params["min_samples_leaf"])),
        "max_features": best_params["max_features"]
    }

    final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(train_input, train_output)

    predictions = final_model.predict(test_input)
    final_metrics = f.calculate_metrics(test_output, predictions)

    return final_metrics, best_params


def optimized_random_forest_bayesian_saved(train_input, train_output, test_input, test_output):
    rf_regressor = RandomForestRegressor(
        n_estimators=56,
        max_depth=5,
        min_samples_split=17,
        min_samples_leaf=6,
        max_features=0.8,
        bootstrap=True,
        random_state=42
    )
    rf_regressor.fit(train_input, train_output)
    predictions = rf_regressor.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def neural_network_from_zero(train_input, train_output, test_input, test_output):
    import neural_network_functions as nf

    HL, OUT, erori_antrenare = nf.antrenare(
        neuronsHL=15,
        LR=0.01,
        nrEpoci=200,
        train_input=train_input,
        train_output=train_output
    )

    mse, mae, rmse, r2 = nf.testare(
        test_input=test_input,
        test_output=test_output,
        HL=HL,
        OUT=OUT
    )

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R^2 Score:", r2)

    return mse, mae, rmse, r2


def optimized_neural_network_from_zero_grid_search(train_input, train_output, test_input, test_output,
                                                   save_path="best_hyperparameters_grid_search.json"):
    import neural_network_functions as nf
    from sklearn.model_selection import ParameterGrid, KFold
    import numpy as np

    param_grid = {
        'neuronsHL': [3, 5, 15],
        'LR': [0.001, 0.01, 0.02],
        'nrEpoci': [25, 50, 100]
    }

    best_score = np.inf
    best_params = {}

    for params in ParameterGrid(param_grid):
        kf = KFold(n_splits=5)
        fold_scores = []
        i = 1
        for train_idx, val_idx in kf.split(train_input):
            X_train, X_val = train_input[train_idx], train_input[val_idx]
            y_train, y_val = train_output[train_idx], train_output[val_idx]

            HL, OUT, _ = nf.antrenare(
                neuronsHL=params['neuronsHL'],
                LR=params['LR'],
                nrEpoci=params['nrEpoci'],
                train_input=X_train,
                train_output=y_train
            )

            mse, _, _, _ = nf.testare(X_val, y_val, HL, OUT)
            print(f"mse combinatia cu numarul {i}: {mse}")
            i += 1
            fold_scores.append(mse)

        avg_mse = np.mean(fold_scores)

        if avg_mse < best_score:
            best_score = avg_mse
            best_params = params

    hyperparam_data = {
        "method": "grid_search",
        "best_hyperparameters": best_params
    }

    with open(save_path, "w") as f:
        json.dump(hyperparam_data, f, indent=4)

    print(f"Cei mai buni hiperparametri au fost salvați în {save_path}")

    return best_params


def optimized_neural_network_from_zero_grid_search_saved(train_input, train_output, test_input, test_output):
    import neural_network_functions as nf

    HL, OUT, erori_antrenare = nf.antrenare(
        neuronsHL=15,
        LR=0.02,
        nrEpoci=100,
        train_input=train_input,
        train_output=train_output
    )

    mse, mae, rmse, r2 = nf.testare(
        test_input=test_input,
        test_output=test_output,
        HL=HL,
        OUT=OUT
    )

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R^2 Score:", r2)

    return mse, mae, rmse, r2


def optimized_neural_network_from_zero_bayesian(train_input, train_output, test_input, test_output,
                                                save_path="best_hyperparameters_bayesian.json"):
    import neural_network_functions as nf
    from bayes_opt import BayesianOptimization

    def objective_function(neuronsHL, LR, nrEpoci):
        params = {
            'neuronsHL': int(round(neuronsHL)),
            'LR': max(LR, 0.001),
            'nrEpoci': int(round(nrEpoci))
        }

        HL, OUT, _ = nf.antrenare(
            train_input=train_input,
            train_output=train_output,
            **params
        )

        mse, _, _, _ = nf.testare(test_input, test_output, HL, OUT)
        return -mse

    param_bounds = {
        'neuronsHL': (5, 50),
        'LR': (0.001, 0.1),
        'nrEpoci': (30, 400)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=42
    )

    optimizer.maximize(init_points=5, n_iter=20)

    best_params = optimizer.max['params']
    best_params = {
        'neuronsHL': int(round(best_params['neuronsHL'])),
        'LR': best_params['LR'],
        'nrEpoci': int(round(best_params['nrEpoci']))
    }

    hyperparam_data = {
        "method": "bayesian",
        "best_hyperparameters": best_params
    }

    with open(save_path, "w") as f:
        json.dump(hyperparam_data, f, indent=4)

    print(f"Cei mai buni hiperparametri au fost salvați în {save_path}")

    return best_params

def optimized_neural_network_from_zero_bayesian_saved(train_input, train_output, test_input, test_output):
    import neural_network_functions as nf

    HL, OUT, erori_antrenare = nf.antrenare(
        neuronsHL=5,
        LR=0.09,
        nrEpoci=251,
        train_input=train_input,
        train_output=train_output
    )

    mse, mae, rmse, r2 = nf.testare(
        test_input=test_input,
        test_output=test_output,
        HL=HL,
        OUT=OUT
    )

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R^2 Score:", r2)

    return mse, mae, rmse, r2


def gradient_boosting(train_input, train_output, test_input, test_output):
    SEED = 42
    gbr = MultiOutputRegressor(GradientBoostingRegressor(loss='absolute_error',
                                                         learning_rate=0.1,
                                                         n_estimators=300,
                                                         max_depth=1,
                                                         random_state=SEED,
                                                         max_features=5))
    gbr.fit(train_input, train_output)
    predictions = gbr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_gradient_boosting_grid_search(train_input, train_output, test_input, test_output):

    SEED = 42
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.02],
        'max_depth': [1, 2, 3],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 0.5],
        'subsample': [0.8, 1.0]
    }

    best_score = np.inf
    best_params = None
    best_metrics = None

    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            for depth in param_grid['max_depth']:
                for mss in param_grid['min_samples_split']:
                    for mf in param_grid['max_features']:
                        for subs in param_grid['subsample']:
                            gbr = MultiOutputRegressor(
                                GradientBoostingRegressor(
                                    loss='absolute_error',
                                    n_estimators=n_est,
                                    learning_rate=lr,
                                    max_depth=depth,
                                    min_samples_split=mss,
                                    max_features=mf,
                                    subsample=subs,
                                    random_state=SEED
                                )
                            )
                            gbr.fit(train_input, train_output)
                            preds = gbr.predict(test_input)
                            metrics = f.calculate_metrics(test_output, preds)
                            mse = metrics[0]

                            if mse < best_score:
                                best_score = mse
                                best_params = {
                                    'n_estimators': n_est,
                                    'learning_rate': lr,
                                    'max_depth': depth,
                                    'min_samples_split': mss,
                                    'max_features': mf,
                                    'subsample': subs
                                }
                                best_metrics = metrics

    return best_metrics, best_params


def optimized_gradient_boosting_grid_search_saved(train_input, train_output, test_input, test_output):
    SEED = 42
    gbr = MultiOutputRegressor(GradientBoostingRegressor(loss='absolute_error',
                                                         n_estimators=100,
                                                         learning_rate=0.02,
                                                         max_depth=2,
                                                         min_samples_split=2,
                                                         max_features='sqrt',
                                                         subsample=1.0,
                                                         random_state=SEED))
    gbr.fit(train_input, train_output)
    predictions = gbr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_gradient_boosting_bayesian(train_input, train_output, test_input, test_output):
    def objective_function(n_estimators, learning_rate, max_depth, max_features, min_samples_split):
        params = {
            "n_estimators": int(round(n_estimators)),
            "learning_rate": max(learning_rate, 0.01),
            "max_depth": int(round(max_depth)),
            "max_features": int(round(max_features)) if max_features < train_input.shape[1] else "sqrt",
            "min_samples_split": int(round(min_samples_split)),
            "loss": "absolute_error",
            "random_state": 42
        }

        model = MultiOutputRegressor(
            GradientBoostingRegressor(**params)
        )

        model.fit(train_input, train_output)

        predictions = model.predict(test_input)

        metrics = f.calculate_metrics(test_output, predictions)
        return -metrics[0]

    param_bounds = {
        "n_estimators": (50, 500),
        "learning_rate": (0.01, 0.3),
        "max_depth": (1, 5),
        "max_features": (1, train_input.shape[1]),
        "min_samples_split": (2, 20)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=42
    )

    optimizer.maximize(n_iter=20, init_points=5)

    best_params = optimizer.max['params']
    best_params = {
        "n_estimators": int(round(best_params["n_estimators"])),
        "learning_rate": best_params["learning_rate"],
        "max_depth": int(round(best_params["max_depth"])),
        "max_features": int(round(best_params["max_features"])),
        "min_samples_split": int(round(best_params["min_samples_split"]))
    }

    final_model = MultiOutputRegressor(
        GradientBoostingRegressor(
            **best_params,
            loss="absolute_error",
            random_state=42
        )
    )
    final_model.fit(train_input, train_output)

    predictions = final_model.predict(test_input)
    final_metrics = f.calculate_metrics(test_output, predictions)

    return final_metrics, best_params


def optimized_gradient_boosting_bayesian_saved(train_input, train_output, test_input, test_output):
    SEED = 42
    gbr = MultiOutputRegressor(GradientBoostingRegressor(loss='absolute_error',
                                                         n_estimators=246,
                                                         learning_rate=0.03,
                                                         max_depth=4,
                                                         min_samples_split=12,
                                                         max_features=1,
                                                         subsample=1.0,
                                                         random_state=SEED))
    gbr.fit(train_input, train_output)
    predictions = gbr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def decision_tree(train_input, train_output, test_input, test_output, max_depth):
    regr = DecisionTreeRegressor(max_depth=max_depth)
    regr.fit(train_input, train_output)
    predictions = regr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def decision_tree_features_importance(train_input, train_output, test_input, test_output, max_depth, feature_names):
    regr = DecisionTreeRegressor(max_depth=max_depth)
    regr.fit(train_input, train_output)
    importance = regr.feature_importances_
    plt.bar(range(train_input.shape[1]), importance)
    plt.xticks(range(train_input.shape[1]), feature_names, rotation=90)
    plt.title('Feature Importances - Decision Tree')
    plt.show()
    predictions = regr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_decision_tree_grid_search(train_input, train_output, test_input, test_output):

    param_grid = {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['squared_error', 'friedman_mse']
    }

    best_score = np.inf
    best_params = None
    best_metrics = None

    for depth in param_grid['max_depth']:
        for split in param_grid['min_samples_split']:
            for leaf in param_grid['min_samples_leaf']:
                for feat in param_grid['max_features']:
                    for crit in param_grid['criterion']:
                        model = DecisionTreeRegressor(
                            max_depth=depth,
                            min_samples_split=split,
                            min_samples_leaf=leaf,
                            max_features=feat,
                            criterion=crit,
                            random_state=42
                        )
                        model.fit(train_input, train_output)

                        preds = model.predict(test_input)
                        metrics = f.calculate_metrics(test_output, preds)
                        mse = metrics[0]

                        if mse < best_score:
                            best_score = mse
                            best_params = {
                                'max_depth': depth,
                                'min_samples_split': split,
                                'min_samples_leaf': leaf,
                                'max_features': feat,
                                'criterion': crit
                            }
                            best_metrics = metrics

    return best_metrics, best_params


def optimized_decision_tree_grid_search_saved(train_input, train_output, test_input, test_output):
    regr = DecisionTreeRegressor(max_depth=5,
                                 min_samples_split=2,
                                 min_samples_leaf=4,
                                 max_features='sqrt',
                                 criterion='squared_error',
                                 random_state=42)
    regr.fit(train_input, train_output)
    predictions = regr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_decision_tree_bayesian(train_input, train_output, test_input, test_output):
    def objective_function(max_depth, min_samples_split, min_samples_leaf, max_features):
        params = {
            "max_depth": int(round(max_depth)),
            "min_samples_split": int(round(min_samples_split)),
            "min_samples_leaf": int(round(min_samples_leaf)),
            "max_features": max(0.1, min(max_features, 1.0))
        }

        model = DecisionTreeRegressor(
            **params,
            random_state=42
        )

        model.fit(train_input, train_output)

        predictions = model.predict(test_input)

        metrics = f.calculate_metrics(test_output, predictions)
        return -metrics[0]

    param_bounds = {
        "max_depth": (1, 30),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": (0.1, 1.0)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=42
    )

    optimizer.maximize(n_iter=15, init_points=5)

    best_params = optimizer.max['params']
    best_params = {
        "max_depth": int(round(best_params["max_depth"])),
        "min_samples_split": int(round(best_params["min_samples_split"])),
        "min_samples_leaf": int(round(best_params["min_samples_leaf"])),
        "max_features": best_params["max_features"]
    }

    final_model = DecisionTreeRegressor(**best_params, random_state=42)
    final_model.fit(train_input, train_output)

    predictions = final_model.predict(test_input)
    final_metrics = f.calculate_metrics(test_output, predictions)

    return final_metrics, best_params

def optimized_decision_tree_bayesian_saved(train_input, train_output, test_input, test_output):
    regr = DecisionTreeRegressor(max_depth=6,
                                 min_samples_split=8,
                                 min_samples_leaf=10,
                                 max_features='0.45',
                                 criterion='squared_error',
                                 random_state=42)
    regr.fit(train_input, train_output)
    predictions = regr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)
