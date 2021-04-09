from hmmlearn.hmm import GaussianHMM
import pickle
import DataGenerator


def main():

    start_date = '2010-01-01'
    end_date = '2020-04-18'

    training_x = DataGenerator.make_features(start_date, end_date, is_training=True)

    # TODO: set model parameters
    n_components = 3
    model = GaussianHMM(n_components)
    model.fit(training_x)

    # TODO: fix pickle file name
    filename = 'team11_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))


if __name__ == "__main__":
    main()



