import onnxruntime as rt
from argparse import ArgumentParser
import numpy as np
from flask import Flask, request, jsonify
from extract import process_url
import constants


def transform_answer(ans):
    if ans == 1:
        return 'Yes'
    return 'No'


def parse_args():
    parser = ArgumentParser('ML web app')
    parser.add_argument('--model',
                        type=str,
                        help='model to use')
    parser.add_argument('--port',
                        type=int,
                        default=5000,
                        help='Port to use')
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0',
                        help='Host to use')
    return parser.parse_args()


args = parse_args()
sess = rt.InferenceSession(args.model)
app = Flask('Test app')


@app.route('/metadata')
def meta():
    metadata_map = sess._model_meta.custom_metadata_map
    metadata_json = jsonify(metadata_map)
    return metadata_json


@app.route('/forward', methods=['POST'])
def predict():
    try:
        url = request.get_json()['url']
    except KeyError:
        return 'bad request', constants.CODE_BAD_REQUEST

    code, features = process_url(url)

    if code != constants.CODE_OK:
        return 'Model failed', constants.CODE_FAILED

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name],
                        {input_name: features.astype(np.float32)})

    return transform_answer(pred_onx[0][0])


if __name__ == '__main__':
    print(sess._model_meta.custom_metadata_map)
    app.run(port=args.port, host=args.host)
