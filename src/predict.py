from src import constants
from src.extract import process_url
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def predict(X, sess):
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name],
                        {input_name: X.astype(np.float32)})
    return pred_onx[0]


def predict_url(url, sess):
    code, features = process_url(url)
    if code != constants.CODE_OK:
        return 'Model failed', constants.CODE_FAILED
    prediction = predict(features, sess)[0]
    return {'answer': prediction}, constants.CODE_OK


def predict_urls(urls, sess):
    X = []
    ok = []

    for url in urls:
        print(url)
        answer = process_url(url)
        if answer[0] == constants.CODE_OK:
            X.append(answer[1])
            ok.append(True)
        else:
            ok.append(False)

    X = np.array(X)
    y_pred = predict(X, sess)
    length = len(y_pred)
    answer = np.array([y_pred[i] if ok[i] else np.NaN for i in range(length)])
    return answer


def make_str(arr):
    return [str(x) for x in arr]


def predict_batch(df, sess):
    answer = predict_urls(df.url, sess)
    return {'answer': make_str(answer)}, constants.CODE_OK


def calculate_metrics(y_pred, y_test):
    return {'accuracy': accuracy_score(y_pred, y_test),
            'f1': f1_score(y_pred, y_test),
            'roc_auc': roc_auc_score(y_pred, y_test)}


def evaluate(df, sess):
    y_test = df[constants.FIELD_IS_TEXT]
    y_pred = predict_urls(df.url, sess)
    ids = ~np.isnan(y_pred)
    metrics = calculate_metrics(y_pred[ids], y_test[ids])
    return {'answer': make_str(y_pred), 'metrics': metrics}, constants.CODE_OK
