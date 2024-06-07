import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Split the dataframe into test and train data
def split_data(data_df):
    """Split a dataframe into training and validation datasets"""

    features = data_df.drop(['fare_amount','pickup_datetime', 'key'], axis=1)
    labels = np.array(data_df['fare_amount'])
    features_train, features_valid, labels_train, labels_valid = \
        train_test_split(features, labels, test_size=0.2, random_state=0)

    return (features_train, features_valid, labels_train, labels_valid)

# Train the model, return the model
def train_model(data, parameters):
    """Train a model with the given datasets and parameters"""
    features_train, features_valid, labels_train, labels_valid = data

    model = GradientBoostingRegressor(**parameters)
    model.fit(features_train, labels_train)

    return model

# Evaluate the metrics for the model
def get_model_metrics(model, data):
    """Construct a dictionary of metrics for the model"""
    _, features_valid, _, labels_valid = data
    predictions = model.predict(features_valid)
    
    mse = metrics.mean_squared_error(labels_valid, predictions)
    mae = metrics.mean_absolute_error(labels_valid, predictions)
    r2 = metrics.r2_score(labels_valid, predictions)
    
    model_metrics = {
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r2_score": r2
    }
    print(model_metrics)

    return model_metrics

# Example usage:
# Assuming you have a DataFrame `df` with the required structure

# parameters = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
# data = split_data(df)
# model = train_model(data, parameters)
# metrics = get_model_metrics(model, data)
