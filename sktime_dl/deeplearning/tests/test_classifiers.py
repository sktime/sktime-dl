from sktime.datasets import load_italy_power_demand, load_basic_motions

from sktime_dl.deeplearning import CNNClassifier
from sktime_dl.utils import construct_all_classifiers, SMALL_NB_EPOCHS


def test_basic_univariate(network=CNNClassifier(nb_epochs=SMALL_NB_EPOCHS)):
    '''
    just a super basic test with gunpoint,
        load data,
        construct classifier,
        fit,
        score
    '''

    print("Start test_basic()")

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    hist = network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))
    print("End test_basic()")


def test_pipeline(network=CNNClassifier(nb_epochs=SMALL_NB_EPOCHS)):
    '''
    slightly more generalised test with sktime pipelines
        load data,
        construct pipeline with classifier,
        fit,
        score
    '''

    print("Start test_pipeline()")

    from sktime.pipeline import Pipeline

    # just a simple (useless) pipeline for the purposes of testing
    # that the keras network is compatible with that system
    steps = [
        ('clf', network)
    ]
    clf = Pipeline(steps)

    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    hist = clf.fit(X_train[:10], y_train[:10])

    print(clf.score(X_test[:10], y_test[:10]))
    print("End test_pipeline()")


def test_highLevelsktime(network=CNNClassifier(nb_epochs=SMALL_NB_EPOCHS)):
    '''
    truly generalised test with sktime tasks/strategies
        load data, build task
        construct classifier, build strategy
        fit,
        score
    '''

    print("start test_highLevelsktime()")

    from sktime.highlevel.tasks import TSCTask
    from sktime.highlevel.strategies import TSCStrategy
    from sklearn.metrics import accuracy_score

    train = load_italy_power_demand(split='TRAIN')
    test = load_italy_power_demand(split='TEST')
    task = TSCTask(target='class_val', metadata=train)

    strategy = TSCStrategy(network)
    strategy.fit(task, train.iloc[:10])

    y_pred = strategy.predict(test.iloc[:10])
    y_test = test.iloc[:10][task.target]
    print(accuracy_score(y_test, y_pred))

    print("End test_highLevelsktime()")


def test_basic_multivariate(network=CNNClassifier(nb_epochs=SMALL_NB_EPOCHS)):
    '''
    just a super basic test with basicmotions,
        load data,
        construct classifier,
        fit,
        score
    '''
    print("Start test_multivariate()")
    X_train, y_train = load_basic_motions(split='TRAIN', return_X_y=True)
    X_test, y_test = load_basic_motions(split='TEST', return_X_y=True)

    hist = network.fit(X_train, y_train)

    print(network.score(X_test, y_test))
    print("End test_multivariate()")


def test_all_networks():
    for network in construct_all_classifiers(SMALL_NB_EPOCHS):
        print('\n\t\t' + network.__class__.__name__ + ' testing started')
        # test_basic_univariate(network)
        test_basic_multivariate(network)
        # test_pipeline(network)
        test_highLevelsktime(network)
        print('\t\t' + network.__class__.__name__ + ' testing finished')


if __name__ == "__main__":
    test_all_networks()
