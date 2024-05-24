import argparse

from app import App


parser = argparse.ArgumentParser()

parser.add_argument('mode', type=str, help='train or predict')
parser.add_argument('--data', type=str, help='dataset')
parser.add_argument('--test', type=str, help='test dataset')
parser.add_argument('--split', type=float, help='split size')
parser.add_argument('--model', type=str, help='model')

args = parser.parse_args()


if __name__ == '__main__':
    App(args).run()
