from pathlib import Path

from sktime.datasets import load_italy_power_demand

from sktime_dl.deeplearning import CNNClassifier
from sktime_dl.meta import DeepLearnerEnsembleClassifier


def test_basic_inmem(
        network=DeepLearnerEnsembleClassifier(
            base_model=CNNClassifier(nb_epochs=50),
            nb_iterations=2,
            keep_in_memory=True,
            model_save_directory=None,
            verbose=True,
        )
):
    """
    just a super basic test with gunpoint,
        load data,
        construct classifier,
        fit,
        score
    """

    print("Start test_basic()")

    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)

    network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))
    print("End test_basic()")


def test_basic_saving(
        network=DeepLearnerEnsembleClassifier(
            base_model=CNNClassifier(nb_epochs=50),
            nb_iterations=2,
            keep_in_memory=False,
            model_save_directory="testResultsDELETE",
            verbose=True,
        )
):
    """
    just a super basic test with gunpoint,
        load data,
        construct classifier,
        fit,
        score
    """

    print("Start test_basic()")

    path = Path(network.model_save_directory)
    # if the directory doesn't get cleaned up because of error in testing
    if not path.exists():
        path.mkdir()

    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)

    network.fit(X_train[:10], y_train[:10])

    print(network.score(X_test[:10], y_test[:10]))

    (
            path / (network.base_model.model_name + "_0.hdf5")
    ).unlink()  # delete file
    (
            path / (network.base_model.model_name + "_1.hdf5")
    ).unlink()  # delete file
    path.rmdir()  # directory should now be empty, fails if not

    print("End test_basic()")


if __name__ == "__main__":
    test_basic_inmem()
    test_basic_saving()
