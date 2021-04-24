import os
import joblib
from functools import partial

import optuna
from sklearn.metrics import roc_auc_score

from stacking import get_train_test_data, get_preprocessor, cross_validate

import warnings
warnings.filterwarnings("ignore")


def define_objective(trial, x, y, est_name):
    if est_name == 'xgb':
        params = {
            'eval_metric': trial.suggest_categorical('eval_metric', ['auc']),
            'n_estimators': trial.suggest_int('n_estimators', 300, 300),
            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 1, 5),
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
            'min_child_weight': trial.suggest_float('min_child_weight', 0, 5),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),
            'colsample_bynode': trial.suggest_discrete_uniform('colsample_bynode', 0.1, 1, 0.01),
            'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.1, 1, 0.01),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.05)}

    elif est_name == 'lgb':
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
            'metric': trial.suggest_categorical('metric', ['auc']),
            'feature_pre_filter': trial.suggest_categorical('feature_pre_filter', [False]),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
            'num_leaves': trial.suggest_int('num_leaves', 2, 32),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.01),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.01),
            'subsample_freq': trial.suggest_int('subsample_freq', 7, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 30),
            'early_stopping_round': trial.suggest_int('early_stopping_round', 10, 10),
            'n_estimators': trial.suggest_int('n_estimators', 100, 100),
            'verbosity': trial.suggest_categorical('verbosity', [-1])}

    elif est_name == 'cb':
        params = {
            'loss_function': trial.suggest_categorical('loss_function', ['Logloss']),
            'eval_metric': trial.suggest_categorical('eval_metric', ['AUC']),
            'iterations': trial.suggest_int('iterations', 50, 50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            'depth': trial.suggest_int('depth', 6, 12),
            'verbose': trial.suggest_categorical('verbose', [False]),
            'early_stopping_rounds': trial.suggest_categorical('early_stopping_rounds', [10]),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 100),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.8, 1),
            'cat_features': trial.suggest_categorical('cat_features', [[5]])}

    results = cross_validate(est_name, x, y, params, method='predict_proba')

    return roc_auc_score(y_train, results['oof_preds'])


if __name__ == '__main__':
    EST_NAME = 'cb'
    N_TRIALS = 20

    # Read data
    x_train, y_train, x_test, y_test, nums, cats = get_train_test_data()

    # Preprocessor
    preprocessor = get_preprocessor(EST_NAME, nums, cats)
    x_train = preprocessor.fit_transform(x_train)

    # Parameter Optimization
    study_name = 'study_' + EST_NAME + '.pkl'
    if os.path.exists(study_name):
        study = joblib.load(study_name)
    else:
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(sampler=sampler, direction='maximize')

    objective = partial(define_objective, x=x_train, y=y_train, est_name=EST_NAME)
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    joblib.dump(study, study_name)
