import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier


def load_porto():
    df = pd.read_csv('./data/porto_train.csv')

    labels = df.columns[2:]

    X = df[labels]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def average_precision(y_true, y_pred):
    score = average_precision_score(y_true, y_pred)
    return 'average_precision', score, True


def recall_at_precision(y_true, y_pred):
    threshold = 0.3
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return 'recall_at_precision', np.max(recall[precision >= threshold]), True


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_porto()

    models = []

    for _ in range(10):
        pos_idx = X_train[y_train == 1].index
        neg_idx = X_train[y_train == 0].sample(pos_idx.shape[0]).index

        X_resample = pd.concat(
            [X_train.loc[pos_idx], X_train.loc[neg_idx]], axis=0
        )

        y_resample = pd.concat(
            [y_train.loc[pos_idx], y_train.loc[neg_idx]], axis=0
        )

        model = lgb.LGBMClassifier(n_estimators=1000)
        model.fit(X_resample, y_resample, eval_set=[
                  (X_valid, y_valid)], eval_metric=average_precision, early_stopping_rounds=50, verbose=10)

        models.append(model)

    res = pd.DataFrame()
    res['label'] = y_valid
    res['score'] = 0

    for model in models:
        res['score'] += model.predict_proba(X_valid)[:, 1]
    res['score'] /= len(models)
    print(res['score'])

    print(average_precision(res['label'], res['score']))

    res = pd.DataFrame()
    res['label'] = y_test
    res['score'] = 0

    for model in models:
        res['score'] += model.predict_proba(X_test)[:, 1]
    res['score'] /= len(models)
    print(res['score'])

    print(average_precision(res['label'], res['score']))
