from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import mlflow.sklearn

from preparation import prepare_data
from train import preprocess, split_data

# Load data
dataset_path = Path.cwd() / 'data' / 'earthquake-2001-2023.csv'
features_to_remove = ['title', 'date_time', 'magType', 'location', 'continent', 'country']
earthquakes = prepare_data(dataset_path, features_to_remove)

# Preprocess data: encoding cat features, scaling num features
preprocessed = preprocess(earthquakes)

# Instantiate models to try
##  Decision Tree does not require scaling or normalization and will therefore use preprocessed dataset for training
dtc = DecisionTreeClassifier(criterion='gini', max_depth=50, max_features=0.75,
                            min_impurity_decrease=0.0, min_samples_leaf=0.01, min_samples_split=0.05,
                            random_state=42)
## Random Forests are not affected by outliers and balance bias and variance trade-off. Performs well on imbalanced datasets.
rfc = RandomForestClassifier(n_estimators=200, min_samples_split=2,
                            min_samples_leaf=1, max_features='sqrt',
                            max_depth=40, bootstrap=False)
## XGBoost does not need scaling, is not affected by outliers but needs a lot of hyperparameter tuning.
xgbc = xgb.XGBClassifier(objective='multiclass:softmax', learning_rate=0.2, gamma=0.4, max_depth=9, min_child_weight=3, colsample_bytree=0.7)

# Oversample data
oversample = SMOTE()
X, y = oversample.fit_resample(preprocessed.drop('alert', axis=1), preprocessed['alert'])
upsampled = pd.concat([X, y], axis=1)

# Split imbalanced data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(preprocessed)

# Split balanced data
X_train_up, X_val_up, X_test_up, y_train_up, y_val_up, y_test_up = split_data(upsampled)

# print lengths of all splits
print(f'X_train: {len(X_train)} | '
      f'X_val: {len(X_val)} | '
      f'X_test: {len(X_test)}')

print(f'y_train: {len(y_train)} | '
      f'y_val: {len(y_val)} | '
      f'y_test: {len(y_test)}')

models_to_try = [dtc, rfc, xgbc]

print("Model fitting with imbalanced data: ")
for model in models_to_try:
    with mlflow.start_run(run_name=f'{model.__class__.__name__}_imbalanced'):
        print(f'\nFitting {model.__class__.__name__}')
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accuracy = model.score(X_val, y_val)
        f1 = f1_score(y_val, preds, average='weighted')
        # Log metrics to MLflow
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1', f1)
        mlflow.log_params(model.get_params())
        print(f'Accuracy: {accuracy} | F1: {f1}\n')

print("Model fitting with upsampled data: ")
for model in models_to_try:
    with mlflow.start_run(run_name=f'{model.__class__.__name__}_upsampled'):
        print(f'\nFitting {model.__class__.__name__}')
        model.fit(X_train_up, y_train_up)
        preds = model.predict(X_val_up)
        accuracy = model.score(X_val_up, y_val_up)
        f1 = f1_score(y_val_up, preds, average='weighted')
        # Log metrics to MLflow
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1', f1)
        mlflow.log_params(model.get_params())
        print(f'Accuracy: {accuracy} | F1: {f1}\n')