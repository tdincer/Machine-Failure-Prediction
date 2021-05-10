import os
import time
import joblib
from functools import wraps

import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

import warnings
warnings.filterwarnings("ignore")


def timer(func):
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print('Elapsed time: %0.4f seconds' % elapsed_time)
        return value
    return wrapper_timer


def read_data(data_dir, file, target, cols):
    df = pd.read_csv(os.path.join(data_dir, file), usecols=cols)
    y = df[target]
    x = df.drop([target], axis=1)
    return x, y


def get_train_test_data():
    DATA_DIR = './DATA'
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test.csv'
    COLS = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]', 'Machine failure']
    CATS = ['Type']
    NUMS = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]']
    TARGET = 'Machine failure'

    x_train, y_train = read_data(DATA_DIR, TRAIN_FILE, TARGET, COLS)
    x_test, y_test = read_data(DATA_DIR, TEST_FILE, TARGET, COLS)
    return x_train, y_train, x_test, y_test, NUMS, CATS


def get_preprocessor(est_name, nums, cats):
    if est_name == 'cb':
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), nums)],
                                         remainder='passthrough')
    elif est_name == 'lgb':
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), nums),
                                                       ('cat', OrdinalEncoder(), cats)],
                                         remainder='passthrough')
    elif est_name == 'xgb':
        preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), nums),
                                                       ('cat', OneHotEncoder(), cats)],
                                         remainder='passthrough')
    return preprocessor


def get_estimator(est_name, params):
    if est_name == 'xgb':
        estimator = xgb.XGBClassifier(**params)
    elif est_name == 'lgb':
        estimator = lgb.LGBMClassifier(**params)
    elif est_name == 'cb':
        estimator = cb.CatBoostClassifier(**params)
    return estimator


def cross_validate(est_name, x, y, params, cv=None, method='predict_proba', return_model=False):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=84)

    oof_preds = np.zeros(len(y))
    models = []

    for fold, (train_index, validation_index) in enumerate(cv.split(x, y)):
        x_train = x[train_index]
        y_train = y[train_index]

        x_validation = x[validation_index]
        y_validation = y[validation_index]

        estimator = get_estimator(est_name, params)
        estimator.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], early_stopping_rounds=10,
                      verbose=False)

        if method == 'predict_proba':
            validation_pred = estimator.predict_proba(x_validation)
        elif method == 'predict':
            validation_pred = estimator.predict(x_validation)

        oof_preds[validation_index] = validation_pred[:, 1]

        if return_model:
            models.append(estimator)

    if return_model:
        return {'oof_preds': oof_preds, 'models': models}
    else:
        return {'oof_preds': oof_preds}


def get_best_params(est_name='xgb'):
    study_file = os.path.join('STUDY', 'study_' + est_name + '.pkl')
    study = joblib.load(study_file)
    params = study.best_params
    return params


def get_test_score(estimator, x_test, y_test):
    test_pred = estimator.predict_proba(x_test)
    test_pred = test_pred[:, 1]
    return roc_auc_score(y_test, test_pred)


class MeanClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, transformer, est_name, params):
        self.transformer = transformer
        self.est_name = est_name
        self.params = params

    def fit(self, X, y=None, save_oof_preds=False):
        self.classes_ = np.unique(y)

        X = self.transformer.fit_transform(X)
        results = cross_validate(self.est_name, X, y, self.params, cv=None, method='predict_proba',
                                 return_model=True)
        self.models_ = results['models']

        if save_oof_preds:
            file_name = self.est_name + '_oof_preds.npy'
            np.save(file_name, results['oof_preds'])

        return self

    def predict_proba(self, X, y=None):
        X = self.transformer.transform(X)
        y_pred = np.zeros(len(X))
        for model in self.models_:
            y_pred += model.predict_proba(X)[:, 1] / len(self.models_)
        return y_pred

    def predict(self, X, y=None, threshold=0.5):
        y_pred = self.predict_proba(X)
        return np.where(y_pred < threshold, 0, 1)

    def score(self, x_true, y_true):
        y_pred = self.predict_proba(x_true)
        return roc_auc_score(y_true, y_pred)


class BlendClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y=None):
        for clf in self.clfs:
            clf.fit(X, y)
        return self

    def predict_proba(self, X):
        y_pred = np.zeros(len(X))

        if self.weights is None:
            for i, clf in enumerate(self.clfs):
                y_pred += clf.predict_proba(X) / len(self.clfs)
            return y_pred
        else:
            for i, clf in enumerate(self.clfs):
                y_pred += clf.predict_proba(X) * self.weights[i]
            return y_pred

    def predict(self, X, y=None, threshold=0.5):
        y_pred = self.predict_proba(X)
        return np.where(y_pred < threshold, 0, 1)

    def score(self, x_true, y_true):
        y_pred = self.predict_proba(x_true)
        return roc_auc_score(y_true, y_pred)
