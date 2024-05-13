from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error

class DTRegression:
  def __init__(self):
    pass

  def fit(self, train_x, train_y):
    self.model = DecisionTreeRegressor()
    self.model.fit(train_x, train_y)
    return self.model

  def predict(self, test_X):
    pred = self.model.predict(test_X)
    return pred

  def evaluate(self, test_X, test_y):
    pred = self.model.predict(test_X)
    self.errors = abs(pred - test_y)
    self.average_error = round(np.mean(self.errors), 2)
    return self.average_error

class RandomForest:
  def __init__(self, n_estimators=1000):
    self.n_estimators = n_estimators

  def fit(self, train_x, train_y):
    self.model = RandomForestRegressor(n_estimators = self.n_estimators, random_state = 42)
    self.model.fit(train_x, train_y)
    return self.model

  def predict(self, test_X):
    pred = self.model.predict(test_X)
    return pred

  def evaluate(self, test_X, test_y):
    pred = self.model.predict(test_X)
    self.errors = abs(pred - test_y)
    self.average_error = round(np.mean(self.errors), 2)
    return self.average_error
class XGBoosting:
  def __init__(self, booster='gbtree', objective = 'reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, n_estimators = 10):
    self.booster = booster
    self.objective= objective
    self.colsample_bytree = colsample_bytree
    self.learning_rate = learning_rate
    self.max_depth = max_depth
    self.n_estimators = n_estimators

  def fit(self, train_X, train_y):
    self.model = xgb.XGBRegressor(booster=self.booster, objective =self.objective,
                              colsample_bytree = self.colsample_bytree, learning_rate = self.learning_rate,
                              max_depth = self.max_depth, n_estimators = self.n_estimators)
    self.model.fit(train_X, train_y)
    return self.model

  def predict(self, test_X):
    pred = self.model.predict(test_X)
    return pred

  def evaluate(self, test_X, test_y):
    pred = self.model.predict(test_X)
    self.errors = abs(pred - test_y)
    self.average_error = round(np.mean(self.errors), 2)
    return self.average_error

def build_neural_network(train_X, train_y, test_X, test_y):
    model = Sequential()
    model.add(Dense(1000, input_shape=(train_X.shape[1],), activation='relu'))  # (features,)
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='linear'))  # output node
    model.summary()  # see what your model looks like

    # compile the model
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    # early stopping callback
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       patience=50,
                       restore_best_weights=True)

    # fit the model!
    # attach it to a new variable called 'history' in case
    # to look at the learning curves
    history = model.fit(train_X, train_y,
                        validation_data=(test_X, test_y),
                        callbacks=[es],
                        epochs=10,
                        batch_size=50,
                        verbose=1)

    return model, history


def train_and_evaluate_models(train_X, test_X, train_y, test_y, data_dmatrix):
    # Train and evaluate different regression models.
    # Decision Tree Regression
    dt_reg = DTRegression()
    dt_reg.fit(train_X, train_y)
    dt_mae = dt_reg.evaluate(test_X, test_y)
    print('Decision Tree Regression Mean Absolute Error:', dt_mae, 'degrees.')

    # Random Forest Regression
    rf_reg = RandomForest(n_estimators=1000)
    rf_reg.fit(train_X, train_y)
    rf_mae = rf_reg.evaluate(test_X, test_y)
    print('Random Forest Regression Mean Absolute Error:', rf_mae, 'degrees.')

    # XGBoost Regression
    xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    xgb_reg.fit(train_X, train_y)
    xgb_mae = mean_absolute_error(test_y, xgb_reg.predict(test_X))
    print('XGBoost Regression Mean Absolute Error:', xgb_mae, 'degrees.')

    # Plot feature importance for XGBoost
    xgb.plot_importance(xgb_reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    
    current_directory = os.getcwd()

    plt.savefig(os.path.join(current_directory,'images', 'feature.png'))

    plt.close()

    return rf_reg
