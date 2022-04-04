import bs4.element
import pandas as pd
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
import numpy as np
from argparse import ArgumentParser
import constants

all_percentiles = [60, 70, 80, 90, 100]

elements = [bs4.element.Comment,
            bs4.element.Doctype,
            bs4.element.NavigableString,
            bs4.element.ProcessingInstruction,
            bs4.element.Script,
            bs4.element.Stylesheet,
            bs4.element.Tag]


def process_url(url):
    try:
        data = requests.get(url)
        if data.status_code == 200:
            soup = BeautifulSoup(data.text, 'html.parser')
            lens = []
            types = defaultdict(int)
            for child in soup.recursiveChildGenerator():
                types[type(child)] += 1
                if type(child) != bs4.element.Tag:
                    lens.append(len(child))

            percentiles = [np.percentile(lens, q) for q in all_percentiles]
            el_arr = [float(types[el]) for el in elements]
            return percentiles + el_arr + [float(len(soup.text))]
        else:
            return None
    except BaseException as e:
        print(e.__class__)


def prepare_dataframe(infile, outfile, with_labels=False):
    df_urls = pd.read_csv(infile)
    features_arr = [process_url(url) for url in df_urls.url]

    percentile_names = [f'p{q}' for q in all_percentiles]
    element_names = [str(el).split('.')[-1][:-2] for el in elements]

    columns = percentile_names + element_names + ['length']
    df_features = pd.DataFrame(features_arr, columns=columns)

    if with_labels:
        df_features[constants.FIELD_IS_TEXT] = df_urls[constants.FIELD_IS_TEXT]

    df_features.to_csv(outfile, index=False)


if __name__ == '__main__':
    parser = ArgumentParser('Prepares features from URL')
    parser.add_argument('--infile',
                        type=str,
                        required=True,
                        help='Input .csv file with urls')
    parser.add_argument('--outfile',
                        type=str,
                        required=True,
                        help='Output .csv file with features')
    parser.add_argument('-l', '--labels',
                        help='For rewriting labels from infile')

    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
