from sktime.datasets import load_italy_power_demand

from sktime_dl.deeplearning import CNNClassifier
from sktime_dl.meta import TunedDeepLearningClassifier


def test_basic_tuning(
        network=TunedDeepLearningClassifier(
            base_model=CNNClassifier(),
            param_grid=dict(nb_epochs=[50, 100], ),
            cv_folds=3,
        )
):
    """
    just a super basic test of the tuner
    """

    print("Start test_basic_tuning()")

    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)

    network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))
    print("End test_basic_tuning()")


if __name__ == "__main__":
    test_basic_tuning()
