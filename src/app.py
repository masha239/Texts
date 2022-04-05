import onnxruntime as rt
from argparse import ArgumentParser
from flask import Flask, request, jsonify
from predict import *


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
    return predict_url(url, sess)


@app.route('/forward_batch', methods=['POST'])
def predict_batch():
    try:
        infile = request.get_json()['url']
    except KeyError:
        return 'bad request', constants.CODE_BAD_REQUEST
    return predict_batch(infile, sess)


@app.route('/evaluate', methods=['GET'])
def predict_batch():
    try:
        infile = request.get_json()['url']
    except KeyError:
        return 'bad request', constants.CODE_BAD_REQUEST
    return evaluate(infile, sess)


if __name__ == '__main__':
    print(sess._model_meta.custom_metadata_map)
    app.run(port=args.port, host=args.host)
