dt_config = {
    'criterion': {
        'type': str,
        'default': 'gini',
        'pipeline_path': 'classifier__',
        'space': ['gini', 'entropy', 'log_loss']
    },
    'splitter': {
        'type': str,
        'default': 'best',
        'pipeline_path': 'classifier__',
        'space': ['best', 'random']
    },
    'max_depth': {
        'type': int,
        'default': 1,
        'pipeline_path': 'classifier__',
        'space': [2, 4, 8, 16, 17, 18, 19, 20]
    },
    'min_samples_split': {
        'type': int,
        'default': 2,
        'pipeline_path': 'classifier__',
        'space': [1, 2, 4, 8, 10, 12, 14, 16]
    },
    'min_samples_leaf': {
        'type': int,
        'default': 1,
        'pipeline_path': 'classifier__',
        'space': [1, 2, 4, 5, 6, 8]
    },
    'min_weight_fraction_leaf': {
        'type': float,
        'default': 0.0,
        'pipeline_path': 'classifier__',
        'space': [0.0, 0.01, 0.05]
    },
    'max_leaf_nodes': {
        'type': int,
        'default': 0,
        'pipeline_path': 'classifier__',
        'space': [0, 1, 2, 4, 8, 9, 10]
    },
    'min_impurity_decrease': {
        'type': float,
        'default': 0.0,
        'pipeline_path': 'classifier__',
        'space': [0.0, 0.01, 0.05, 0.1]
    },
    'ccp_alpha': {
            'type': float,
            'default': 0.0,
            'pipeline_path': 'classifier__',
            'space': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        }
}


bc_mlp_config = {
    'hidden_layer_sizes': {
        'type': tuple,
        'default': (9, 4, 2,),
        'pipeline_path': 'classifier__',
        'space': [
            # breast cancer network
            (18, 9, 4,),
            (9, 4, 2,),
            (18, 9,),
            (9, 4,),
            (4, 2,),
        ]
    },
    'activation': {
            'type': str,
            'default': 'relu',
            'pipeline_path': 'classifier__',
            'space': [
                'identity',
                'logistic',
                'tanh',
                'relu'
            ]
        },
    'solver': {
        'type': str,
        'default': 'adam',
        'pipeline_path': 'classifier__',
        'space': ['lbfgs', 'sgd', 'adam']
    },
    'alpha': {
        'type': float,
        'default': 0.0001,
        'pipeline_path': 'classifier__',
        'space': [0.00005, 0.0001, 0.0005]
    },
    'learning_rate': {
        'type': str,
        'default': 'constant',
        'pipeline_path': 'classifier__',
        'space': ['constant', 'invscaling', 'adaptive']
    },
    'learning_rate_init': {
        'type': float,
        'default': 0.001,
        'pipeline_path': 'classifier__',
        'space': [0.0005, 0.001, 0.005]
    },
    'power_t': {
        'type': float,
        'default': 0.5,
        'pipeline_path': 'classifier__',
        'space': [0.4, 0.5, 0.6]
    },
    'max_iter': {
        'type': int,
        'default': 200,
        'pipeline_path': 'classifier__',
        'space': [100, 200, 400]
    },
    'tol': {
        'type': float,
        'default': 0.0001,
        'pipeline_path': 'classifier__',
        'space': [0.00005, 0.0001, 0.0005]
    },
    'momentum': {
        'type': float,
        'default': 0.9,
        'pipeline_path': 'classifier__',
        'space': [0.89, 0.9, 0.99]
    },
    'validation_fraction': {
        'type': float,
        'default': 0.1,
        'pipeline_path': 'classifier__',
        'space': [0.05, 0.1, 0.25, 0.5, 0.9]
    },
    'beta_1': {
        'type': float,
        'default': 0.9,
        'pipeline_path': 'classifier__',
        'space': [0.89, 0.9, 0.99]
    },
    'beta_2': {
        'type': float,
        'default': 0.999,
        'pipeline_path': 'classifier__',
        'space': [0.9985, 0.9990, 0.9995]
    },
    'epsilon': {
        'type': float,
        'default': 0.00000001,
        'pipeline_path': 'classifier__',
        'space': [0.000000005, 0.000000010, 0.000000015]
    },
    'n_iter_no_change': {
        'type': int,
        'default': 10,
        'pipeline_path': 'classifier__',
        'space': [5, 10, 15]
    }
}


sp_mlp_config = {
    'hidden_layer_sizes': {
        'type': tuple,
        'default': (27, 13, 6,),
        'pipeline_path': 'classifier__',
        'space': [
            # steel plate network
            (54, 27, 13,),
            (27, 13, 6,),
            (54, 27,),
            (27, 13,),
            (13, 6,),
        ]
    },
    'activation': {
            'type': str,
            'default': 'relu',
            'pipeline_path': 'classifier__',
            'space': [
                'identity',
                'logistic',
                'tanh',
                'relu'
            ]
        },
    'solver': {
        'type': str,
        'default': 'adam',
        'pipeline_path': 'classifier__',
        'space': ['lbfgs', 'sgd', 'adam']
    },
    'alpha': {
        'type': float,
        'default': 0.0001,
        'pipeline_path': 'classifier__',
        'space': [0.00005, 0.0001, 0.0005]
    },
    'learning_rate': {
        'type': str,
        'default': 'constant',
        'pipeline_path': 'classifier__',
        'space': ['constant', 'invscaling', 'adaptive']
    },
    'learning_rate_init': {
        'type': float,
        'default': 0.001,
        'pipeline_path': 'classifier__',
        'space': [0.0005, 0.001, 0.005]
    },
    'power_t': {
        'type': float,
        'default': 0.5,
        'pipeline_path': 'classifier__',
        'space': [0.4, 0.5, 0.6]
    },
    'max_iter': {
        'type': int,
        'default': 200,
        'pipeline_path': 'classifier__',
        'space': [100, 200, 400]
    },
    'tol': {
        'type': float,
        'default': 0.0001,
        'pipeline_path': 'classifier__',
        'space': [0.00005, 0.0001, 0.0005]
    },
    'momentum': {
        'type': float,
        'default': 0.9,
        'pipeline_path': 'classifier__',
        'space': [0.89, 0.9, 0.99]
    },
    'validation_fraction': {
        'type': float,
        'default': 0.1,
        'pipeline_path': 'classifier__',
        'space': [0.05, 0.1, 0.25, 0.5, 0.9]
    },
    'beta_1': {
        'type': float,
        'default': 0.9,
        'pipeline_path': 'classifier__',
        'space': [0.89, 0.9, 0.99]
    },
    'beta_2': {
        'type': float,
        'default': 0.999,
        'pipeline_path': 'classifier__',
        'space': [0.9985, 0.9990, 0.9995]
    },
    'epsilon': {
        'type': float,
        'default': 0.00000001,
        'pipeline_path': 'classifier__',
        'space': [0.000000005, 0.000000010, 0.000000015]
    },
    'n_iter_no_change': {
        'type': int,
        'default': 10,
        'pipeline_path': 'classifier__',
        'space': [5, 10, 15]
    }
}


gb_config = {
    'loss': {
        'type': str,
        'default': 'log_loss',
        'pipeline_path': 'classifier__',
        'space': ['log_loss', 'deviance', 'exponential']
    },
    'learning_rate': {
        'type': float,
        'default': 0.1,
        'pipeline_path': 'classifier__',
        'space': [0.050, 0.1, 0.15]
    },
    'n_estimators': {
        'type': int,
        'default': 100,
        'pipeline_path': 'classifier__',
        'space': [50, 100, 200]
    },
    'subsample': {
        'type': float,
        'default': 1.0,
        'pipeline_path': 'classifier__',
        'space': [0.1, 0.25, 0.5, 1.0]
    },
    'criterion': {
        'type': str,
        'default': 'friedman_mse',
        'pipeline_path': 'classifier__',
        'space': ['friedman_mse', 'squared_error']
    },
    'min_samples_leaf': {
        'type': int,
        'default': 2,
        'pipeline_path': 'classifier__',
        'space': [1, 2, 4, 5, 6, 8]
    },
    'min_weight_fraction_leaf': {
        'type': float,
        'default': 0.0,
        'pipeline_path': 'classifier__',
        'space': [0.0, 0.01, 0.05, 0.1]
    },
    'max_depth': {
        'type': int,
        'default': 3,
        'pipeline_path': 'classifier__',
        'space': [3, 6, 12, 15, 17, 20, 24]
    },
    'min_impurity_decrease': {
        'type': float,
        'default': 0.0,
        'pipeline_path': 'classifier__',
        'space': [0.0, 0.01, 0.05, 0.1]
    },
    'max_leaf_nodes': {
        'type': int,
        'default': 2,
        'pipeline_path': 'classifier__',
        'space': [2, 4, 8, 9, 10, 15]
    },
    'validation_fraction': {
        'type': float,
        'default': 0.1,
        'pipeline_path': 'classifier__',
        'space': [0.1, 0.25, 0.5, 0.9]
    },
    'n_iter_no_change': {
        'type': int,
        'default': 15,
        'pipeline_path': 'classifier__',
        'space': [5, 10, 15]
    },
    'tol': {
        'type': float,
        'default': 0.0001,
        'pipeline_path': 'classifier__',
        'space': [0.00005, 0.0001, 0.0005]
    },
    'ccp_alpha': {
            'type': float,
            'default': 0.0,
            'pipeline_path': 'classifier__',
            'space': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        }
}


svc_config = {
    'C': {
            'type': float,
            'default': 1.0,
            'pipeline_path': 'classifier__',
            'space': [1.0, 2.0, 4.0]
        },
    'kernel': {
        'type': str,
        'default': 'linear',
        'pipeline_path': 'classifier__',
        'space': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    },
    'degree': {
        'type': int,
        'default': 3,
        'pipeline_path': 'classifier__',
        'space': [2, 3, 4]
    },
    'gamma': {
        'type': str,
        'default': 'scale',
        'pipeline_path': 'classifier__',
        'space': ['scale', 'auto']
    },
    'coef0': {
        'type': float,
        'default': 0.0,
        'pipeline_path': 'classifier__',
        'space': [0.0, 0.1, 0.5, 1.0, 2.0]
    },
    'tol': {
        'type': float,
        'default': 0.001,
        'pipeline_path': 'classifier__',
        'space': [0.0005, 0.001, 0.0015]
    },
    'cache_size': {
        'type': float,
        'default': 200,
        'pipeline_path': 'classifier__',
        'space': [200, 400, 800]
    },
    'decision_function_shape': {
            'type': str,
            'default': 'ovr',
            'pipeline_path': 'classifier__',
            'space': ['ovo', 'ovr']
        }
}


knn_config = {
    'n_neighbors': {
            'type': int,
            'default': 5,
            'pipeline_path': 'classifier__',
            'space': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
    'weights': {
            'type': str,
            'default': 'uniform',
            'pipeline_path': 'classifier__',
            'space': ['uniform', 'distance']
        },
    'algorithm': {
            'type': str,
            'default': 'auto',
            'pipeline_path': 'classifier__',
            'space': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
    'leaf_size': {
            'type': int,
            'default': 30,
            'pipeline_path': 'classifier__',
            'space': [15, 30, 45, 60]
        },
    'p': {
            'type': int,
            'default': int,
            'pipeline_path': 'classifier__',
            'space': [1, 2, 3, 4, 5]
        },
    'metric': {
            'type': str,
            'default': 'minkowski',
            'pipeline_path': 'classifier__',
            'space': ['jaccard', 'cosine', 'precomputed', 'manhattan', 'l1', 'euclidean', 'minkowski', 'l2']
        }
}