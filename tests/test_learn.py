import pytest
import numpy as np
import os
import tempfile
import pandas as pd
import shutil
import onnxruntime as rt
from src.learn import create_model
import src.constants


class Args:
    def __init__(self, folder_path):
        self.outfile = os.path.join(folder_path, 'model.onnx')
        dull_data = np.zeros((30, 14), dtype=float)
        self.infile = os.path.join(folder_path, 'df.csv')
        dull_columns = ['a'] * 13 + [src.constants.FIELD_IS_TEXT]
        df = pd.DataFrame(dull_data, columns=dull_columns)
        df.to_csv(self.infile)
        self.hash = 1111
        self.seed = 0
        self.experiment = 'dull experiment'


@pytest.fixture(scope='module')
def folders():
    folder_path = tempfile.mkdtemp()
    args = Args(folder_path)
    create_model(args)
    sess = rt.InferenceSession(args.outfile)
    yield {
        'sess': sess,
        'args': args
    }
    shutil.rmtree(folder_path)
    assert not os.path.exists(folder_path)


def test_file_created(folders):
    assert os.path.exists(folders['args'].outfile)


def test_predictions_zeros(folders):
    sess = folders['sess']
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    new_dull_data = np.zeros((5, 14), dtype=float)
    pred_onx = sess.run([label_name],
                        {input_name: new_dull_data.astype(np.float32)})[0]
    assert sum(pred_onx) == 0


def test_metadata(folders):
    ans = folders['sess']._model_meta.custom_metadata_map
    assert ans['hash'] == folders['args']['hash']
    assert ans['data'] == folders['args']['data']
    assert ans['experiment'] == folders['args']['experiment']
