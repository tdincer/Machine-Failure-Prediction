import os
import joblib
import utility as ut
import warnings
warnings.filterwarnings("ignore")


def main():
    x_train, y_train, x_test, y_test, NUMS, CATS = ut.get_train_test_data()

    lgbparams = ut.get_best_params('lgb')
    xgbparams = ut.get_best_params('xgb')
    cbparams = ut.get_best_params('cb')

    transformer_lgb = ut.get_preprocessor('lgb', NUMS, CATS)
    transformer_xgb = ut.get_preprocessor('xgb', NUMS, CATS)
    transformer_cb = ut.get_preprocessor('cb', NUMS, CATS)

    clf_lgb = ut.MeanClassifier(transformer_lgb, 'lgb', lgbparams)
    clf_xgb = ut.MeanClassifier(transformer_xgb, 'xgb', xgbparams)
    clf_cb = ut.MeanClassifier(transformer_cb, 'cb', cbparams)

    clf_lgb.fit(x_train, y_train)
    clf_xgb.fit(x_train, y_train)
    clf_cb.fit(x_train, y_train)

    # Save models
    for clf in [clf_lgb, clf_xgb, clf_cb]:
        file_name = os.path.join('MODELS', clf.est_name + '.model')
        joblib.dump(clf, file_name, compress=1)

    print('xgb test score: %f' % clf_lgb.score(x_test, y_test))
    print('lgb test score: %f' % clf_xgb.score(x_test, y_test))
    print('cb test score: %f' % clf_cb.score(x_test, y_test))


if __name__ == '__main__':
    main()
