from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import xgboost as xgb


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RandomForestRegressor()

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)