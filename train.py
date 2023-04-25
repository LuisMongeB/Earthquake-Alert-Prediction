from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from imblearn.over_sampling import SMOTE


def split_data(df, ratio=(0.7, 0.2, 0.1), random_state=42):
    """
    Splits a pandas dataframe into train, validation, and test sets based on the provided ratio.

    Returns:
        Tuple of three dataframes: (train_df, val_df, test_df)
    """
    assert round(sum(ratio), 0) == 1.0, "Split ratio must add up to 1"

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('alert', axis=1),
                                                        df['alert'],
                                                        test_size=ratio[2],
                                                        random_state=random_state)

    # Split the remaining data into train and validation sets
    val_ratio = ratio[1] / (ratio[0] + ratio[1])
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=val_ratio,
                                                      random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_categorical_features(df, encoder):
    """
    Encode categorical features.
    """
    # Encode categorical features
    alert_enc = pd.DataFrame({'alert': encoder.fit_transform(df['alert'])})
    tsunami_enc = pd.DataFrame(
        {'tsunami': encoder.fit_transform(df['tsunami'])})
    cdi_enc = pd.get_dummies(
        df['cdi'], prefix='cdi', drop_first=True).reset_index(drop=True)

    # join both dataframes
    cat_df = pd.concat([cdi_enc, tsunami_enc, alert_enc], axis=1)

    return cat_df

def scale_numerical_features(df, scaler):
    """
    Drops categorical variables, selects numerical ones returns scaled dataframe.
    """
    num_df = df.select_dtypes(include=np.number).drop(
        ['tsunami', 'cdi'], axis=1)
    scaled_features = scaler.fit_transform(num_df)
    num_df = pd.DataFrame(scaled_features, columns=num_df.columns)

    return num_df


def preprocess(df):
    """
    Preprocess data for training.
    1. Split data into train, validation, and test sets
    2. Encode categorical features
    3. Scale numerical features
    """
    # Encode categorical features
    cat_df = encode_categorical_features(df, LabelEncoder())
    num_df = scale_numerical_features(df, StandardScaler())
    preprocessed = pd.concat([num_df, cat_df], axis=1)

    return preprocessed


def train_and_predict(X_train, y_train, model):
    """
    Train a model on a split and return the predictions.
    """
    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the validation set
    preds = model.predict(X_train)

    return preds

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, average='weighted')
    # roc_auc = roc_auc_score(actual, pred)
    return accuracy, f1


def cv_upsampled(model, X, y, upsample=True):
    if upsample:
        over = SMOTE()
        X, y = over.fit_resample(X, y)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
