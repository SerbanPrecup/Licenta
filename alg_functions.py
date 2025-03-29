import numpy as np
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

from scipy.stats import loguniform, randint, uniform
from bayes_opt import BayesianOptimization

import functions as f
from linear_regression_functions import LinearRegressionFromZero


def linear_regression(train_input, train_output, test_input, test_output):
    lin = LinearRegression()
    lin.fit(train_input, train_output)
    predictions = lin.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_linear_regression_grid_search(train_input, train_output, test_input, test_output):
    model = LinearRegression()
    param_grid = {
        'fit_intercept': [True, False],
        'positive': [True, False]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_input, train_output)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_input)
    return f.calculate_metrics(test_output, predictions)

def linear_regression_from_zero(train_input, train_output, test_input, test_output, lr=0.02, iters=2000):
    lin = LinearRegressionFromZero()
    param,loss = lin.train(train_input, train_output,0.02,2000)
    return lin.evaluate(test_input, test_output)


def optimized_linear_regression_from_zero_grid_search(train_input, train_output, test_input, test_output):
    best_mse = float('inf')
    best_params = {}

    param_grid = {
        'learning_rate': [0.001, 0.01, 0.02, 0.1],
        'iterations': [500, 1000, 2000, 3000]
    }

    for lr in param_grid['learning_rate']:
        for iters in param_grid['iterations']:
            model = LinearRegressionFromZero()
            params, loss = model.train(train_input, train_output, lr, iters)

            predictions = model.predict(train_input)
            current_mse = mean_squared_error(train_output, predictions)

            if current_mse < best_mse:
                best_mse = current_mse
                best_params = {'learning_rate': lr, 'iterations': iters}

    final_model = LinearRegressionFromZero()
    params, loss = final_model.train(train_input, train_output,
                                     best_params['learning_rate'],
                                     best_params['iterations'])

    return final_model.evaluate(test_input, test_output), best_params


def optimized_linear_regression_from_zero_bayesian(train_input, train_output, test_input, test_output):
    def objective_function(learning_rate, iterations):
        lr = max(learning_rate, 1e-5) #evitare valori negative
        iters = int(round(iterations))

        model = LinearRegressionFromZero()
        _, loss = model.train(train_input, train_output, lr, iters)

        return -loss[-1]

    pbounds = {
        'learning_rate': (0.0001, 0.2),
        'iterations': (100, 5000)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True
    )

    optimizer.maximize(
        init_points=5,
        n_iter=20
    )

    best_params = optimizer.max['params']
    best_lr = best_params['learning_rate']
    best_iters = int(round(best_params['iterations']))

    final_model = LinearRegressionFromZero()
    final_model.train(train_input, train_output, best_lr, best_iters)

    return final_model.evaluate(test_input, test_output), best_params

def polynomial_regression(train_input, train_output, test_input, test_output, degree):
    poly = PolynomialFeatures(degree=degree)
    train_input_poly = poly.fit_transform(train_input)
    test_input_poly = poly.transform(test_input)

    lin = LinearRegression()
    lin.fit(train_input_poly, train_output)
    predictions = lin.predict(test_input_poly)

    return f.calculate_metrics(test_output, predictions)


def optimized_polynomial_regression_grid_search(train_input, train_output, test_input, test_output):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression())
    ])

    param_grid = {
        'poly__degree': [1, 2, 3, 4, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_input, train_output)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_polynomial_regression_bayesian(train_input, train_output, test_input, test_output):
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


def poisson_regression(train_input, train_output, test_input, test_output, alpha):
    model = make_pipeline(
        SplineTransformer(n_knots=5, degree=3),
        MultiOutputRegressor(PoissonRegressor(alpha=alpha))
    )
    model.fit(train_input, train_output)
    predictions = model.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_poisson_regression_grid_search(train_input, train_output, test_input, test_output):
    pipeline = make_pipeline(
        SplineTransformer(),
        MultiOutputRegressor(PoissonRegressor())
    )

    param_grid = {
        'splinetransformer__n_knots': [5, 10, 15],
        'splinetransformer__degree': [2, 3],
        'multioutputregressor__estimator__alpha': [0.01, 0.1, 1, 10]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(train_input, train_output)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_input)
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
    pipeline = Pipeline([
        ('svr', MultiOutputRegressor(SVR()))
    ])

    param_grid = {
        'svr__estimator__kernel': ['rbf', 'linear', 'poly'],
        'svr__estimator__C': [0.1, 1, 10, 100],
        'svr__estimator__gamma': ['scale', 'auto', 0.1, 1],
        'svr__estimator__epsilon': [0.1, 0.2, 0.5],
        'svr__estimator__degree': [2, 3]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(train_input, train_output)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_input)

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


def random_forest(train_input, train_output, test_input, test_output):
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(train_input, train_output)
    predictions = rf_regressor.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_random_forest_grid_search(train_input, train_output, test_input, test_output):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(train_input, train_output)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_input)

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


def neural_network(train_input, train_output, test_input, test_output):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential()
    model.add(Dense(8, input_dim=3, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.fit(train_input, train_output, epochs=30, batch_size=1, verbose=1)
    predictions = model.predict(test_input)
    return f.calculate_metrics(test_output, predictions)

def optimized_neural_network_grid_search(train_input, train_output, test_input, test_output):
    from tensorflow.python.keras import Sequential
    from tensorflow.python.keras.layers import Dense
    # from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import GridSearchCV

    def create_model(units=8, activation='relu', optimizer='adam'):
        model = Sequential()
        model.add(Dense(units, input_dim=3, activation=activation))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
        return model

    # model = KerasRegressor(build_fn=create_model, verbose=0)

    param_grid = {
        'units': [8, 16, 32],
        'activation': ['relu', 'tanh'],
        'optimizer': ['adam', 'sgd'],
        'batch_size': [1, 10, 20],
        'epochs': [30, 50]
    }

    grid_search = GridSearchCV(
        # estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )

    grid_search.fit(train_input, train_output)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(test_input)

    return f.calculate_metrics(test_output, predictions)


def optimized_neural_network_bayesian(train_input, train_output, test_input, test_output):
    from tensorflow.python.keras import Sequential
    from tensorflow.python.keras.layers import Dense
    # from tensorflow.python.keras.optimizers import Adam, SGD
    from bayes_opt import BayesianOptimization
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        train_input, train_output,
        test_size=0.2,
        random_state=42
    )

    def objective_function(units, lr, batch_size, epochs, activation_param, optimizer_param):
        units = int(round(units))
        batch_size = int(round(batch_size))
        epochs = int(round(epochs))

        activation = "relu" if activation_param < 0.5 else "tanh"
        # optimizer = Adam(learning_rate=lr) if optimizer_param < 0.5 else SGD(learning_rate=lr)

        model = Sequential([
            Dense(units, input_dim=3, activation=activation),
            Dense(2, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizer)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        return -history.history['val_loss'][-1]

    param_bounds = {
        'units': (8, 64),
        'lr': (0.0001, 0.1),
        'batch_size': (8, 64),
        'epochs': (20, 100),
        'activation_param': (0, 1),
        'optimizer_param': (0, 1)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        random_state=42
    )

    optimizer.maximize(init_points=10, n_iter=20)

    best_params = optimizer.max['params']
    final_params = {
        'units': int(round(best_params['units'])),
        'activation': 'relu' if best_params['activation_param'] < 0.5 else 'tanh',
        'optimizer': 'adam' if best_params['optimizer_param'] < 0.5 else 'sgd',
        'lr': best_params['lr'],
        'batch_size': int(round(best_params['batch_size'])),
        'epochs': int(round(best_params['epochs']))
    }

    # final_optimizer = Adam(lr=final_params['lr']) if final_params['optimizer'] == 'adam' else SGD(lr=final_params['lr'])

    final_model = Sequential([
        Dense(final_params['units'], input_dim=3, activation=final_params['activation']),
        Dense(2, activation='linear')
    ])
    # final_model.compile(loss='mse', optimizer=final_optimizer)
    final_model.fit(train_input, train_output, epochs=final_params['epochs'], batch_size=final_params['batch_size'],
                    verbose=0)

    predictions = final_model.predict(test_input)
    metrics = f.calculate_metrics(test_output, predictions)

    return metrics, final_params

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


def optimized_neural_network_from_zero_grid_search(train_input, train_output, test_input, test_output):
    import neural_network_functions as nf
    from sklearn.model_selection import ParameterGrid, KFold
    import numpy as np

    param_grid = {
        'neuronsHL': [3, 5, 15, 25],
        'LR': [0.001, 0.01, 0.02, 0.1],
        'nrEpoci': [50, 100, 200, 300]
    }

    best_score = np.inf
    best_params = {}

    for params in ParameterGrid(param_grid):

        kf = KFold(n_splits=5)
        fold_scores = []

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
            fold_scores.append(mse)

        avg_mse = np.mean(fold_scores)

        if avg_mse < best_score:
            best_score = avg_mse
            best_params = params

    HL, OUT, _ = nf.antrenare(
        neuronsHL=best_params['neuronsHL'],
        LR=best_params['LR'],
        nrEpoci=best_params['nrEpoci'],
        train_input=train_input,
        train_output=train_output
    )

    mse, mae, rmse, r2 = nf.testare(test_input, test_output, HL, OUT)

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R^2 Score:", r2)

    return mse, mae, rmse, r2


def optimized_neural_network_from_zero_bayesian(train_input, train_output, test_input, test_output):
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

    HL, OUT, _ = nf.antrenare(
        train_input=train_input,
        train_output=train_output,
        **best_params
    )

    mse, mae, rmse, r2 = nf.testare(test_input, test_output, HL, OUT)

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
    gbr = MultiOutputRegressor(
        GradientBoostingRegressor(loss='absolute_error', random_state=SEED)
    )

    param_grid = {
        'estimator__n_estimators': [100, 200, 300],
        'estimator__learning_rate': [0.01, 0.1, 0.02, 0.2],
        'estimator__max_depth': [1, 2, 3],
        'estimator__min_samples_split': [2, 5],
        'estimator__max_features': ['sqrt', 0.5],
        'estimator__subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        gbr,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(train_input, train_output)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_input)

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


def decision_tree(train_input, train_output, test_input, test_output, max_depth):
    regr = DecisionTreeRegressor(max_depth=max_depth)
    regr.fit(train_input, train_output)
    predictions = regr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)


def optimized_decision_tree_grid_search(train_input, train_output, test_input, test_output):
    dt = DecisionTreeRegressor(random_state=42)

    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['squared_error', 'friedman_mse']
    }

    grid_search = GridSearchCV(
        dt,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(train_input, train_output)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(test_input)

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
