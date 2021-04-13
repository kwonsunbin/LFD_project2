from hmmlearn.hmm import GaussianHMM
import pickle
import DataGenerator
import sys


def main():
    try :
        mode = sys.argv[1]
    except IndexError: 
        print("need moving average mode!!")
    start_date = '2010-01-01'
    end_date = '2021-03-18'

    training_x, _ = DataGenerator.make_features(start_date, end_date, mode=mode, input_days=1, span=3  ,is_training=True)

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

def train(start_date, end_date, mode, input_days, span, n_components):

    training_x, _ = DataGenerator.make_features(start_date, end_date, mode=mode, input_days=input_days, span=span, is_training=True)

    # TODO: set model parameters
    model = GaussianHMM(n_components, n_iter=100)
    model.fit(training_x)

    # TODO: fix pickle file name
    filename = 'team11_model.pkl'
    pickle.dump(model, open(filename, 'wb'))




