from hmmlearn.hmm import GaussianHMM
import pickle
import DataGenerator
import sys


def main():

    mode = "ema"
    start_date = '2010-01-01'
    end_date = '2021-04-01'

    training_x, _ = DataGenerator.make_features(start_date, end_date, mode=mode, input_days=1, span=14  ,is_training=True)

    # TODO: set model parameters
    n_components = 14
    model = GaussianHMM(n_components, n_iter=100)
    model.fit(training_x)

    # TODO: fix pickle file name
    filename = 'team11_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))

if __name__ == "__main__":
    main()

