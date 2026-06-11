from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# XGBoost param grid
param_grid = {
    'classifier__n_estimators':     [100, 200, 300],
    'classifier__max_depth':        [3, 5, 7],
    'classifier__learning_rate':    [0.01, 0.05, 0.1],
    'classifier__subsample':        [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__scale_pos_weight': [1, 1.86],  # handles class imbalance
}

# RandomizedSearch faster than GridSearch for large space
search = RandomizedSearchCV(
    pipe, param_grid,
    n_iter=50, cv=cv, scoring='roc_auc',
    n_jobs=-1, verbose=1, random_state=42
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Best AUC:", search.best_score_)
print("Best Params:", search.best_params_)