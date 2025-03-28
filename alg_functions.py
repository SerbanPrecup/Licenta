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
    from tensorflow.python.keras import Sequential
    from tensorflow.python.keras.layers import Dense
    model = Sequential()
    model.add(Dense(8, input_dim=3, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.fit(train_input, train_output, epochs=30, batch_size=1, verbose=1)
    predictions = model.predict(test_input)
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

    return mse,mae,rmse,r2


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