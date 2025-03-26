from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

import functions as f

def linear_regression(train_input,train_output,test_input,test_output):
    lin = LinearRegression()
    lin.fit(train_input, train_output)
    predictions = lin.predict(test_input)
    return f.calculate_metrics(test_output, predictions)

def polynomial_regression(train_input,train_output,test_input,test_output,degree):
    poly = PolynomialFeatures(degree=degree)
    train_input_poly = poly.fit_transform(train_input)
    test_input_poly = poly.fit_transform(test_input)

    lin = LinearRegression()
    lin.fit(train_input_poly, train_output)
    predictions = lin.predict(test_input_poly)

    return f.calculate_metrics(test_output, predictions)

def poisson_regression(train_input,train_output,test_input,test_output,alpha):
    model = make_pipeline(
        SplineTransformer(n_knots=5, degree=3),
        MultiOutputRegressor(PoissonRegressor(alpha=alpha))
    )
    model.fit(train_input, train_output)
    predictions = model.predict(test_input)
    return f.calculate_metrics(test_output, predictions)

def svr(train_input,train_output,test_input,test_output):
    svr = MultiOutputRegressor(SVR(kernel='rbf'))
    svr.fit(train_input, train_output)
    predictions = svr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)

def random_forest(train_input,train_output,test_input,test_output):
    rf_regressor = RandomForestRegressor(n_estimators=100,random_state=42)
    rf_regressor.fit(train_input, train_output)
    predictions = rf_regressor.predict(test_input)
    return f.calculate_metrics(test_output, predictions)

def neural_network(train_input,train_output,test_input,test_output):
    model = Sequential()
    model.add(Dense(8, input_dim=3, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.fit(train_input,train_output, epochs=30, batch_size=1, verbose=1)
    predictions = model.predict(test_input)
    return f.calculate_metrics(test_output, predictions)

def gradient_boosting(train_input,train_output,test_input,test_output,SEED):
    gbr = MultiOutputRegressor(GradientBoostingRegressor(loss='absolute_error',
                                                         learning_rate=0.1,
                                                         n_estimators=300,
                                                         max_depth=1,
                                                         random_state=SEED,
                                                         max_features=5))
    gbr.fit(train_input,train_output)
    predictions = gbr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)

def decision_tree(train_input,train_output,test_input,test_output,max_depth):
    regr = DecisionTreeRegressor(max_depth=max_depth)
    regr.fit(train_input,train_output)
    predictions = regr.predict(test_input)
    return f.calculate_metrics(test_output, predictions)

