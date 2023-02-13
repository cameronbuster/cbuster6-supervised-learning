def dt_grid():
    hyperparameter_grid = {
        'classifier__criterion': ['gini', 'entropy', 'log_loss'],
        'classifier__splitter': ['best', 'random'],
        'classifier__max_depth': [2, 4, 8, 16, 17, 18, 19, 20],
        'classifier__min_samples_split': [1, 2, 4, 8, 10, 12, 14, 16],
        'classifier__min_samples_leaf': [1, 2, 4, 5, 6, 8],
        'classifier__min_weight_fraction_leaf': [0.0, 0.01, 0.05],
        'classifier__max_leaf_nodes': [0, 1, 2, 4, 8, 9, 10],
        'classifier__min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],
        'classifier__ccp_alpha': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    }

    return hyperparameter_grid


def bc_mlp_grid():
    hyperparameter_grid = {
        'classifier__hidden_layer_sizes': [
            # breast cancer network
            (18, 9, 4,),
            (9, 4, 2,),
            (18, 9,),
            (9, 4,),
            (4, 2,),
        ],
        'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'classifier__solver': ['lbfgs', 'sgd', 'adam'],
        'classifier__alpha': [0.00005, 0.0001, 0.0005],
        'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'classifier__learning_rate_init': [0.0005, 0.001, 0.005],
        'classifier__power_t': [0.4, 0.5, 0.6],
        'classifier__max_iter': [100, 200, 400],
        'classifier__tol': [0.00005, 0.0001, 0.0005],
        'classifier__momentum': [0.89, 0.9, 0.99],
        'classifier__validation_fraction': [0.05, 0.1, 0.25, 0.5, 0.9],
        'classifier__beta_1': [0.89, 0.9, 0.99],
        'classifier__beta_2': [0.9985, 0.9990, 0.9995],
        'classifier__epsilon': [0.000000005, 0.000000010, 0.000000015],
        'classifier__n_iter_no_change': [5, 10, 15],
    }

    return hyperparameter_grid


def sp_mlp_grid():
    hyperparameter_grid = {
        'classifier__hidden_layer_sizes': [
            # steel plate network
            (54, 27, 13,),
            (27, 13, 6,),
            (54, 27,),
            (27, 13,),
            (13, 6,),
        ],
        'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'classifier__solver': ['lbfgs', 'sgd', 'adam'],
        'classifier__alpha': [0.00005, 0.0001, 0.0005],
        'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'classifier__learning_rate_init': [0.0005, 0.001, 0.005],
        'classifier__power_t': [0.4, 0.5, 0.6],
        'classifier__max_iter': [100, 200, 400],
        'classifier__tol': [0.00005, 0.0001, 0.0005],
        'classifier__momentum': [0.89, 0.9, 0.99],
        'classifier__validation_fraction': [0.05, 0.1, 0.25, 0.5, 0.9],
        'classifier__beta_1': [0.89, 0.9, 0.99],
        'classifier__beta_2': [0.9985, 0.9990, 0.9995],
        'classifier__epsilon': [0.000000005, 0.000000010, 0.000000015],
        'classifier__n_iter_no_change': [5, 10, 15],
    }

    return hyperparameter_grid


def gb_grid():
    hyperparameter_grid = {
        'classifier__loss': ['log_loss', 'deviance', 'exponential'],
        'classifier__learning_rate': [0.050, 0.1, 0.15],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__subsample': [0.1, 0.25, 0.5, 1.0],
        'classifier__criterion': ['friedman_mse', 'squared_error'],
        'classifier__min_samples_leaf': [1, 2, 4, 5, 6, 8],
        'classifier__min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 6, 12, 15, 17, 20, 24],
        'classifier__min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],
        'classifier__max_leaf_nodes': [2, 4, 8, 9, 10, 15],
        'classifier__validation_fraction': [0.1, 0.25, 0.5, 0.9],
        'classifier__n_iter_no_change': [5, 10, 15],
        'classifier__tol': [0.00005, 0.0001, 0.0005],
        'classifier__ccp_alpha': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    }

    return hyperparameter_grid


def svc_grid():
    hyperparameter_grid = {
        'classifier__C': [1.0, 2.0, 4.0],
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'classifier__degree': [2, 3, 4],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__coef0': [0.0, 0.1, 0.5, 1.0, 2.0],
        'classifier__tol': [0.0005, 0.001, 0.0015],
        'classifier__cache_size': [200, 400, 800],
        'classifier__decision_function_shape': ['ovo', 'ovr']
    }

    return hyperparameter_grid


def knn_grid():
    hyperparameter_grid = {
        'classifier__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'classifier__leaf_size': [15, 30, 45, 60],
        'classifier__p': [1, 2, 3, 4, 5],
        'classifier__metric': [
            'jaccard', 'cosine', 'precomputed', 'manhattan', 'l1', 'euclidean', 'minkowski', 'l2'
        ],
    }

    return hyperparameter_grid
