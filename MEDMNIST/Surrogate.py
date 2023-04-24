# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:48:33 2022

@author: IRMAS
"""

import random
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import numpy as np

from scipy.stats import uniform, randint

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

#This class implements different surrogates to speed up the evolutionary algorithm

class Surrogate:

    def predict(self, test_data):
        model = load('gbr.pkl')
        prediction = model.predict(test_data)
        return prediction
    def test(self,d,l):
        print(d)
        print(l)

    def display_scores(self,scores):
        print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

    def report_best_scores(self,results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def gbm_regressor(self, train_data, train_label):
        GBR = GradientBoostingRegressor()
        parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04],
                      'subsample': [0.9, 0.5, 0.2, 0.1],
                      'n_estimators': [100, 500, 1000, 1500],
                      'max_depth': [4, 6, 8, 10]
                      }
        grid_GBR = GridSearchCV(estimator=GBR, param_grid=parameters, cv=2, n_jobs=-1)
        grid_GBR.fit(train_data, train_label)
        dump(grid_GBR, 'gbr.pkl')
    def xg_boost(self,train_data, train_label):
        xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)

        xgb_model.fit(train_data, train_label)

        y_pred = xgb_model.predict(train_data)
    def xg_boost_kfold(self,train_data,train_label):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        scores = []

        for train_index, test_index in kfold.split(train_data):
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_label[train_index], train_label[test_index]

            xgb_model = xgb.XGBRegressor(objective="reg:linear")
            xgb_model.fit(X_train, y_train)

            y_pred = xgb_model.predict(X_test)

            scores.append(mean_squared_error(y_test, y_pred))

        self.display_scores(np.sqrt(scores))
        dump(xgb_model, 'xgb_model.pkl')

    def predict_xgb(self,test_data):

        model = load('xgb_model.pkl')
        prediction = model.predict(test_data)
        return prediction
    def xgb_hpo(self,train_data,train_label):

        xgb_model = xgb.XGBRegressor()

        params = {
            "colsample_bytree": uniform(0.7, 0.3),
            "gamma": uniform(0, 0.5),
            "learning_rate": uniform(0.03, 0.3),  # default 0.1
            "max_depth": randint(2, 6),  # default 3
            "n_estimators": randint(100, 150),  # default 100
            "subsample": uniform(0.6, 0.4)
        }

        search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1,
                                    n_jobs=1, return_train_score=True)

        search.fit(X, y)
        dump(search, 'xgb_model.pkl')
        self.report_best_scores(search.cv_results_, 1)