import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Assuming the updated functions from the previous message are saved in train.py
from train import split_data, train_model, get_model_metrics

"""A set of simple unit tests for protecting against regressions in train.py"""

def test_split_data():
    test_data = {
        'key': [0, 1, 2, 3, 4],
        'fare_amount': [1, 2, 3, 5, 5],
        'pickup_datetime':[0, 0, 1, 1, 1],
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 1, 1, 2, 1]
    }

    data_df = pd.DataFrame(data=test_data)
    features_train, features_valid, labels_train, labels_valid = split_data(data_df)

    # verify that columns were removed correctly
    assert "fare_amount" not in features_train.columns
    assert "key" not in features_train.columns
    assert "col1" in features_train.columns

    # verify that data was split as desired
    assert features_train.shape == (4, 2)
    assert features_valid.shape == (1, 2)

def test_train_model():
    data = __get_test_datasets()

    params = {
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": 3
    }

    model = train_model(data, params)

    # verify that parameters are passed in to the model correctly
    for param_name, param_value in params.items():
        assert getattr(model, param_name) == param_value

def test_get_model_metrics():
    class MockModel:
        @staticmethod
        def predict(data):
            return np.array([0.5, 0.5])

    data = __get_test_datasets()
    metrics = get_model_metrics(MockModel(), data)

    # verify that metrics is a dictionary containing the regression metrics
    assert "mean_squared_error" in metrics
    assert "mean_absolute_error" in metrics
    assert "r2_score" in metrics

    mse = metrics["mean_squared_error"]
    mae = metrics["mean_absolute_error"]
    r2 = metrics["r2_score"]

    np.testing.assert_almost_equal(mse, 0.25)
    np.testing.assert_almost_equal(mae, 0.5)
    np.testing.assert_almost_equal(r2, -3.0)

def __get_test_datasets():
    """This is a helper function to set up some test data"""
    X_train = np.array([[1], [2], [3], [4], [5], [6]])
    y_train = np.array([1, 2, 3, 4, 5, 6])
    X_valid = np.array([[7], [8]])
    y_valid = np.array([7, 8])

    return (X_train, X_valid, y_train, y_valid)
