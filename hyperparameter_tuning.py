import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def rf_tuning(rfc, X_train, y_train):
    # randomforest hyperparameter tuning

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 4)]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 5, 10, 20, 50]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 5, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    rf_random = GridSearchCV(estimator=rfc, param_grid=random_grid, scoring='balanced_accuracy', cv=3, verbose=1, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    return rf_random.best_params_

def dt_tuning(dtc, X_train, y_train):
    params = {
        'max_depth': [None, 2, 3, 5, 10, 20, 40, 50],
        'min_samples_split': [0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [0.01, 0.05, 0.1, 0.2],
        'max_features': [None, 'sqrt', 'log2', 0.3, 0.5, 0.75, 1.0],
        'criterion': ["gini", "entropy"],

    }
    dt_random = GridSearchCV(estimator=dtc, param_grid=params, scoring='balanced_accuracy', cv=3, verbose=2, n_jobs=-1)
    dt_random.fit(X_train, y_train)
    return dt_random.best_params_
    
def xgboost_tuning(xgb, X_train, y_train):
        
        params = {
        'learning_rate': [0.001, 0.01, 0.1, 0.15, 0.2],
        'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
        }

        xgb_model = xgb.XGBClassifier()
        xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=params, cv=5, n_jobs=-1, verbose=3)
        xgb_grid.fit(X_train, y_train)
        print("xgb best params: ")
        print(xgb_grid.best_params_)
        return xgb_grid.best_params_