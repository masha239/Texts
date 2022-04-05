from src import constants
from src.extract import process_url
import numpy as np


def transform_answer(ans):
    if ans == 1:
        return 'Yes'
    return 'No'


def predict_url(url, sess):
    code, features = process_url(url)
    if code != constants.CODE_OK:
        return 'Model failed', constants.CODE_FAILED

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name],
                        {input_name: features.astype(np.float32)})
    return transform_answer(pred_onx), constants.CODE_OK


def predict_batch(infile, sess):
    pass


def calculate_metrics(y_pred, y_test, sess):
    pass


def evaluate(infile, sess):
    pass