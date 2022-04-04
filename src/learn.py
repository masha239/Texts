import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import argparse
import random
import constants


def state_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Makes ML model in .onnx format')
    parser.add_argument('--infile',
                        type=str,
                        required=True,
                        help='Input .csv dataframe for learning')
    parser.add_argument('--outfile',
                        type=str,
                        required=True,
                        help='Output, .onnx file for pretrained classification '
                             'model')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed for model')

    args = parser.parse_args()
    df = pd.read_csv(args.infile)

    X = df.drop(constants.FIELD_IS_TEXT, axis=1).values
    y = df[constants.FIELD_IS_TEXT].values

    parameters = {'max_depth': [3, 4, 5], 'min_samples_split': range(2, 10)}
    rf = RandomForestClassifier(random_state=args.seed)
    clf = GridSearchCV(rf, parameters)
    clf.fit(X, y)

    print(clf.best_score_, clf.best_params_)

    with open(args.outfile, 'wb') as file:
        pickle.dump(clf.best_estimator_, file)
