import pandas as pd
import os
import shutil
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import time

from helpers.helpers import bc_feature_create, bc_feature_drop, sp_feature_create, sp_feature_drop, get_data, \
    hyperparam_hist, hyperparam_categorical, generate_learning_curves
from helpers.dummy import DummiesCreate
from helpers.hyperparameter_grids import dt_grid, bc_mlp_grid, sp_mlp_grid, gb_grid, svc_grid, knn_grid
from helpers.hyperparameter_config import dt_config, bc_mlp_config, sp_mlp_config, gb_config, svc_config, knn_config

from bokeh.plotting import save, output_file
from bokeh.models import TabPanel, Tabs
from pandas.api.types import is_numeric_dtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score


dataset_choice = [
    'breast-cancer',
    'steel-plate-faults'
]

algorithms = [
    'DecisionTreeClassifier',
    'MLPClassifier',
    'GradientBoostingClassifier',
    'SVC',
    'KNeighborsClassifier'
]


def load_data(dataset):
    # create model directories
    if os.path.exists(f'models') and os.path.isdir(f'models'):
        if os.path.exists(f'models/{dataset}') and os.path.isdir(f'models/{dataset}'):
            shutil.rmtree(f'models/{dataset}')
            os.mkdir(f'models/{dataset}')
        else:
            os.mkdir(f'models/{dataset}')
    else:
        os.mkdir(f'models')
        os.mkdir(f'models/{dataset}')

    # create figure directories
    if os.path.exists(f'figures') and os.path.isdir(f'figures'):
        if os.path.exists(f'figures/{dataset}') and os.path.isdir(f'figures/{dataset}'):
            shutil.rmtree(f'figures/{dataset}')
            os.mkdir(f'figures/{dataset}')
            os.mkdir(f'figures/{dataset}/hyperparameter_plots')
            os.mkdir(f'figures/{dataset}/learning_curves')
            os.mkdir(f'figures/{dataset}/eda')
        else:
            os.mkdir(f'figures/{dataset}')
            os.mkdir(f'figures/{dataset}/hyperparameter_plots')
            os.mkdir(f'figures/{dataset}/learning_curves')
            os.mkdir(f'figures/{dataset}/eda')
    else:
        os.mkdir(f'figures')
        os.mkdir(f'figures/{dataset}')
        os.mkdir(f'figures/{dataset}/hyperparameter_plots')
        os.mkdir(f'figures/{dataset}/learning_curves')
        os.mkdir(f'figures/{dataset}/eda')

    if dataset == 'breast-cancer':
        df = pd.read_csv('../datasets/breast-cancer/breast-cancer.data', names=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])
        sns.countplot(x="Class", data=df)
        plt.savefig(f'figures/{dataset}/eda/target_count_plot.png')
        plt.clf()

    if dataset == 'steel-plate-faults':
        # df = pd.read_csv('../datasets/steel-plate-faults/Faults.NNA', sep='\t')
        df = pd.read_csv('../datasets/steel-plate-faults/faults.csv')

    return df


def preprocess_bc_data(bc_df):
    # popped target for breast cancer dataset
    bc_target = bc_df.pop('Class')

    # train test split breast cancer dataset
    X_train, X_test, y_train, y_test = train_test_split(bc_df, bc_target, test_size=0.2, random_state=1991)

    # write our train and test files locally for evaluation later
    X_train.to_csv('../datasets/breast-cancer/X_train.csv', index=False)
    X_test.to_csv('../datasets/breast-cancer/X_test.csv', index=False)
    y_train.to_csv('../datasets/breast-cancer/y_train.csv', index=False)
    y_test.to_csv('../datasets/breast-cancer/y_test.csv', index=False)

    return


def preprocess_sp_data(sp_df):
    # make multiclass target
    sp_df['multiclass_target'] = 0
    sp_df.loc[sp_df['Pastry'] == 1, 'multiclass_target'] = 0
    sp_df.loc[sp_df['Z_Scratch'] == 1, 'multiclass_target'] = 1
    sp_df.loc[sp_df['K_Scatch'] == 1, 'multiclass_target'] = 2
    sp_df.loc[sp_df['Stains'] == 1, 'multiclass_target'] = 3
    sp_df.loc[sp_df['Dirtiness'] == 1, 'multiclass_target'] = 4
    sp_df.loc[sp_df['Bumps'] == 1, 'multiclass_target'] = 5
    sp_df.loc[sp_df['Other_Faults'] == 1, 'multiclass_target'] = 6

    sns.countplot(x="multiclass_target", data=sp_df)
    plt.savefig(f'figures/{dataset}/eda/target_count_plot.png')
    plt.clf()

    # popped targets for steel plate faults dataset
    sp_multiclass_target = sp_df.pop('multiclass_target')

    X_train, X_test, y_train, y_test = train_test_split(sp_df, sp_multiclass_target, test_size=0.2, random_state=1991)

    # write our train and test files locally for evaluation later
    X_train.to_csv('../datasets/steel-plate-faults/X_train.csv', index=False)
    X_test.to_csv('../datasets/steel-plate-faults/X_test.csv', index=False)
    y_train.to_csv('../datasets/steel-plate-faults/y_train.csv', index=False)
    y_test.to_csv('../datasets/steel-plate-faults/y_test.csv', index=False)

    return


def build_decision_tree_model(dataset, X_train, y_train, hyperparameter_grid):
    if dataset == 'breast-cancer':
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(bc_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(bc_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', DecisionTreeClassifier())
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            hyperparameter_grid,
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            scoring='roc_auc',
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    else:
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(sp_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(sp_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', DecisionTreeClassifier())
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            hyperparameter_grid,
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    joblib.dump(grid_search, f'models/{dataset}/dt.pkl')

    return


def analyze_decision_tree_model(X_test, y_test, dataset):
    # model = gs.best_estimator_.named_steps['classifier']
    # y_test_pred_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
    # y_test_preds = gs.best_estimator_.predict(X_test)
    #
    # score = roc_auc_score(y_test, y_test_pred_proba)
    # print(classification_report(y_test, y_test_preds))

    gs = joblib.load(f'models/{dataset}/dt.pkl')

    training_jobs_df = pd.DataFrame.from_dict(gs.cv_results_)
    training_jobs_df = training_jobs_df.rename(
        columns={
            'mean_test_score': 'FinalObjectiveValue',
            'param_classifier__criterion': 'criterion',
            'param_classifier__splitter': 'splitter',
            'param_classifier__max_depth': 'max_depth',
            'param_classifier__min_samples_split': 'min_samples_split',
            'param_classifier__min_samples_leaf': 'min_samples_leaf',
            'param_classifier__min_weight_fraction_leaf': 'min_weight_fraction_leaf',
            'param_classifier__max_features': 'max_features',
            'param_classifier__max_leaf_nodes': 'max_leaf_nodes',
            'param_classifier__min_impurity_decrease': 'min_impurity_decrease',
            'param_classifier__ccp_alpha': 'ccp_alpha'
        })
    training_jobs_df['criterion'] = training_jobs_df['criterion'].astype(str)
    training_jobs_df['splitter'] = training_jobs_df['splitter'].astype(str)
    training_jobs_df['max_depth'] = training_jobs_df['max_depth'].astype(int)
    training_jobs_df['min_samples_split'] = training_jobs_df['min_samples_split'].astype(int)
    training_jobs_df['min_samples_leaf'] = training_jobs_df['min_samples_leaf'].astype(int)
    training_jobs_df['min_weight_fraction_leaf'] = training_jobs_df['min_weight_fraction_leaf'].astype(float)
    training_jobs_df['max_leaf_nodes'] = training_jobs_df['max_leaf_nodes'].astype(int)
    training_jobs_df['min_impurity_decrease'] = training_jobs_df['min_impurity_decrease'].astype(float)
    training_jobs_df['ccp_alpha'] = training_jobs_df['ccp_alpha'].astype(float)

    tabs = []
    for hyperparam in dt_config.keys():
        is_numeric = is_numeric_dtype(list(dt_config[hyperparam].items())[0][1])
        if is_numeric:
            p = hyperparam_hist(training_jobs_df, hyperparam)
        else:
            p = hyperparam_categorical(training_jobs_df, hyperparam)
        tabs.append(TabPanel(child=p, title=hyperparam))
    tabs = Tabs(tabs=tabs)

    output_file(f'figures/{dataset}/hyperparameter_plots/dt_hyperparameters.html')
    save(tabs)

    plt = generate_learning_curves(gs, dt_grid(), dataset)
    plt.savefig(f'figures/{dataset}/learning_curves/dt_learning_curves.png')
    plt.clf()

    return


def build_mlp_model(dataset, X_train, y_train):
    if dataset == 'breast-cancer':
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(bc_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(bc_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', MLPClassifier())
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            bc_mlp_grid(),
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            scoring='roc_auc',
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    else:
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(sp_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(sp_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', MLPClassifier())
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            sp_mlp_grid(),
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    joblib.dump(grid_search, f'models/{dataset}/mlp.pkl')

    return


def analyze_mlp_model(X_test, y_test, dataset):
    # model = gs.best_estimator_.named_steps['classifier']
    # y_test_pred_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
    # y_test_preds = gs.best_estimator_.predict(X_test)
    #
    # score = roc_auc_score(y_test, y_test_pred_proba)
    # print(classification_report(y_test, y_test_preds))

    gs = joblib.load(f'models/{dataset}/mlp.pkl')

    training_jobs_df = pd.DataFrame.from_dict(gs.cv_results_)
    training_jobs_df = training_jobs_df.rename(
        columns={
            'mean_test_score': 'FinalObjectiveValue',
            'param_classifier__hidden_layer_sizes': 'hidden_layer_sizes',
            'param_classifier__activation': 'activation',
            'param_classifier__solver': 'solver',
            'param_classifier__alpha': 'alpha',
            'param_classifier__learning_rate': 'learning_rate',
            'param_classifier__learning_rate_init': 'learning_rate_init',
            'param_classifier__power_t': 'power_t',
            'param_classifier__max_iter': 'max_iter',
            'param_classifier__shuffle': 'shuffle',
            'param_classifier__tol': 'tol',
            'param_classifier__momentum': 'momentum',
            'param_classifier__nesterovs_momentum': 'nesterovs_momentum',
            'param_classifier__early_stopping': 'early_stopping',
            'param_classifier__validation_fraction': 'validation_fraction',
            'param_classifier__beta_1': 'beta_1',
            'param_classifier__beta_2': 'beta_2',
            'param_classifier__epsilon': 'epsilon',
            'param_classifier__n_iter_no_change': 'n_iter_no_change',
        })
    training_jobs_df['hidden_layer_sizes'] = training_jobs_df['hidden_layer_sizes'].astype(str)
    training_jobs_df['activation'] = training_jobs_df['activation'].astype(str)
    training_jobs_df['solver'] = training_jobs_df['solver'].astype(str)
    training_jobs_df['alpha'] = training_jobs_df['alpha'].astype(float)
    training_jobs_df['learning_rate'] = training_jobs_df['learning_rate'].astype(str)
    training_jobs_df['learning_rate_init'] = training_jobs_df['learning_rate_init'].astype(float)
    training_jobs_df['power_t'] = training_jobs_df['power_t'].astype(float)
    training_jobs_df['max_iter'] = training_jobs_df['max_iter'].astype(int)
    training_jobs_df['tol'] = training_jobs_df['tol'].astype(float)
    training_jobs_df['momentum'] = training_jobs_df['momentum'].astype(float)
    training_jobs_df['validation_fraction'] = training_jobs_df['validation_fraction'].astype(float)
    training_jobs_df['beta_1'] = training_jobs_df['beta_1'].astype(float)
    training_jobs_df['beta_2'] = training_jobs_df['beta_2'].astype(float)
    training_jobs_df['epsilon'] = training_jobs_df['epsilon'].astype(float)
    training_jobs_df['n_iter_no_change'] = training_jobs_df['n_iter_no_change'].astype(int)

    if dataset == 'breast-cancer':
        tabs = []
        for hyperparam in bc_mlp_config.keys():
            is_numeric = is_numeric_dtype(list(bc_mlp_config[hyperparam].items())[0][1])
            if is_numeric:
                p = hyperparam_hist(training_jobs_df, hyperparam)
            else:
                p = hyperparam_categorical(training_jobs_df, hyperparam)
            tabs.append(TabPanel(child=p, title=hyperparam))
        tabs = Tabs(tabs=tabs)

        output_file(f'figures/{dataset}/hyperparameter_plots/mlp_hyperparameters.html')
        save(tabs)

        plt = generate_learning_curves(gs, bc_mlp_grid(), dataset)
        plt.savefig(f'figures/{dataset}/learning_curves/mlp_learning_curves.png')
        plt.clf()

    if dataset == 'steel-plate-faults':
        tabs = []
        for hyperparam in sp_mlp_config.keys():
            is_numeric = is_numeric_dtype(list(sp_mlp_config[hyperparam].items())[0][1])
            if is_numeric:
                p = hyperparam_hist(training_jobs_df, hyperparam)
            else:
                p = hyperparam_categorical(training_jobs_df, hyperparam)
            tabs.append(TabPanel(child=p, title=hyperparam))
        tabs = Tabs(tabs=tabs)

        output_file(f'figures/{dataset}/hyperparameter_plots/mlp_hyperparameters.html')
        save(tabs)

        plt = generate_learning_curves(gs, sp_mlp_grid(), dataset)
        plt.savefig(f'figures/{dataset}/learning_curves/mlp_learning_curves.png')
        plt.clf()

    return


def build_gradient_boosted_model(dataset, X_train, y_train, hyperparameter_grid):
    if dataset == 'breast-cancer':
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(bc_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(bc_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', GradientBoostingClassifier())
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            hyperparameter_grid,
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            scoring='roc_auc',
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    else:
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(sp_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(sp_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', GradientBoostingClassifier())
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            hyperparameter_grid,
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    joblib.dump(grid_search, f'models/{dataset}/gb.pkl')

    return


def analyze_gradient_boosted_model(X_test, y_test, dataset):
    # model = gs.best_estimator_.named_steps['classifier']
    # y_test_pred_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
    # y_test_preds = gs.best_estimator_.predict(X_test)
    #
    # score = roc_auc_score(y_test, y_test_pred_proba)
    # print(classification_report(y_test, y_test_preds))

    gs = joblib.load(f'models/{dataset}/gb.pkl')

    training_jobs_df = pd.DataFrame.from_dict(gs.cv_results_)
    training_jobs_df = training_jobs_df.rename(
        columns={
            'mean_test_score': 'FinalObjectiveValue',
            'param_classifier__loss': 'loss',
            'param_classifier__learning_rate': 'learning_rate',
            'param_classifier__n_estimators': 'n_estimators',
            'param_classifier__subsample': 'subsample',
            'param_classifier__criterion': 'criterion',
            'param_classifier__min_samples_leaf': 'min_samples_leaf',
            'param_classifier__min_weight_fraction_leaf': 'min_weight_fraction_leaf',
            'param_classifier__max_depth': 'max_depth',
            'param_classifier__min_impurity_decrease': 'min_impurity_decrease',
            'param_classifier__max_leaf_nodes': 'max_leaf_nodes',
            'param_classifier__validation_fraction': 'validation_fraction',
            'param_classifier__n_iter_no_change': 'n_iter_no_change',
            'param_classifier__tol': 'tol',
            'param_classifier__ccp_alpha': 'ccp_alpha'
        })
    training_jobs_df['loss'] = training_jobs_df['loss'].astype(str)
    training_jobs_df['learning_rate'] = training_jobs_df['learning_rate'].astype(float)
    training_jobs_df['n_estimators'] = training_jobs_df['n_estimators'].astype(int)
    training_jobs_df['subsample'] = training_jobs_df['subsample'].astype(float)
    training_jobs_df['criterion'] = training_jobs_df['criterion'].astype(str)
    training_jobs_df['min_samples_leaf'] = training_jobs_df['min_samples_leaf'].astype(int)
    training_jobs_df['min_weight_fraction_leaf'] = training_jobs_df['min_weight_fraction_leaf'].astype(float)
    training_jobs_df['max_depth'] = training_jobs_df['max_depth'].astype(int)
    training_jobs_df['min_impurity_decrease'] = training_jobs_df['min_impurity_decrease'].astype(float)
    training_jobs_df['max_leaf_nodes'] = training_jobs_df['max_leaf_nodes'].astype(int)
    training_jobs_df['validation_fraction'] = training_jobs_df['validation_fraction'].astype(float)
    training_jobs_df['n_iter_no_change'] = training_jobs_df['n_iter_no_change'].astype(int)
    training_jobs_df['tol'] = training_jobs_df['tol'].astype(float)
    training_jobs_df['ccp_alpha'] = training_jobs_df['ccp_alpha'].astype(float)

    tabs = []
    for hyperparam in gb_config.keys():
        is_numeric = is_numeric_dtype(list(gb_config[hyperparam].items())[0][1])
        if is_numeric:
            p = hyperparam_hist(training_jobs_df, hyperparam)
        else:
            p = hyperparam_categorical(training_jobs_df, hyperparam)
        tabs.append(TabPanel(child=p, title=hyperparam))
    tabs = Tabs(tabs=tabs)

    output_file(f'figures/{dataset}/hyperparameter_plots/gb_hyperparameters.html')
    save(tabs)

    plt = generate_learning_curves(gs, gb_grid(), dataset)
    plt.savefig(f'figures/{dataset}/learning_curves/gb_learning_curves.png')
    plt.clf()

    return


def build_svc_model(dataset, X_train, y_train, hyperparameter_grid):
    if dataset == 'breast-cancer':
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(bc_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(bc_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', SVC(probability=True))
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            hyperparameter_grid,
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            scoring='roc_auc',
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    else:
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(sp_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(sp_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', SVC(probability=True))
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            hyperparameter_grid,
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    joblib.dump(grid_search, f'models/{dataset}/svc.pkl')

    return


def analyze_svc_model(X_test, y_test, dataset):
    # model = gs.best_estimator_.named_steps['classifier']
    # y_test_pred_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
    # y_test_preds = gs.best_estimator_.predict(X_test)
    #
    # score = roc_auc_score(y_test, y_test_pred_proba)
    # print(classification_report(y_test, y_test_preds))

    gs = joblib.load(f'models/{dataset}/svc.pkl')

    training_jobs_df = pd.DataFrame.from_dict(gs.cv_results_)
    training_jobs_df = training_jobs_df.rename(
        columns={
            'mean_test_score': 'FinalObjectiveValue',
            'param_classifier__C': 'C',
            'param_classifier__kernel': 'kernel',
            'param_classifier__degree': 'degree',
            'param_classifier__gamma': 'gamma',
            'param_classifier__coef0': 'coef0',
            'param_classifier__tol': 'tol',
            'param_classifier__cache_size': 'cache_size',
            'param_classifier__decision_function_shape': 'decision_function_shape'
        })
    training_jobs_df['C'] = training_jobs_df['C'].astype(float)
    training_jobs_df['kernel'] = training_jobs_df['kernel'].astype(str)
    training_jobs_df['degree'] = training_jobs_df['degree'].astype(int)
    training_jobs_df['gamma'] = training_jobs_df['gamma'].astype(str)
    training_jobs_df['coef0'] = training_jobs_df['coef0'].astype(float)
    training_jobs_df['tol'] = training_jobs_df['tol'].astype(float)
    training_jobs_df['cache_size'] = training_jobs_df['cache_size'].astype(int)
    training_jobs_df['decision_function_shape'] = training_jobs_df['decision_function_shape'].astype(str)

    tabs = []
    for hyperparam in svc_config.keys():
        is_numeric = is_numeric_dtype(list(svc_config[hyperparam].items())[0][1])
        if is_numeric:
            p = hyperparam_hist(training_jobs_df, hyperparam)
        else:
            p = hyperparam_categorical(training_jobs_df, hyperparam)
        tabs.append(TabPanel(child=p, title=hyperparam))
    tabs = Tabs(tabs=tabs)

    output_file(f'figures/{dataset}/hyperparameter_plots/svc_hyperparameters.html')
    save(tabs)

    plt = generate_learning_curves(gs, svc_grid(), dataset)
    plt.savefig(f'figures/{dataset}/learning_curves/svc_learning_curves.png')
    plt.clf()

    return


def build_knn_model(dataset, X_train, y_train, hyperparameter_grid):
    if dataset == 'breast-cancer':
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(bc_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(bc_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', KNeighborsClassifier())
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            hyperparameter_grid,
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            scoring='roc_auc',
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    else:
        pipeline = Pipeline(
            [
                ('feature_create', FunctionTransformer(sp_feature_create, validate=False)),
                ('feature_drop', FunctionTransformer(sp_feature_drop, validate=False)),
                ('feature_dummy_create', DummiesCreate()),
                ('fill_nulls', SimpleImputer()),
                ('scaling', MinMaxScaler()),
                ('vt', VarianceThreshold()),
                ('select', SelectPercentile(chi2, percentile=100)),
                ('classifier', KNeighborsClassifier())
            ]
        )

        grid_search = RandomizedSearchCV(
            pipeline,
            hyperparameter_grid,
            n_iter=50000,
            n_jobs=-1,
            verbose=3,
            cv=5,
            refit=True,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)

    joblib.dump(grid_search, f'models/{dataset}/knn.pkl')

    return


def analyze_knn_model(X_test, y_test, dataset):
    # model = gs.best_estimator_.named_steps['classifier']
    # y_test_pred_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
    # y_test_preds = gs.best_estimator_.predict(X_test)
    #
    # score = roc_auc_score(y_test, y_test_pred_proba)
    # print(classification_report(y_test, y_test_preds))

    gs = joblib.load(f'models/{dataset}/knn.pkl')

    training_jobs_df = pd.DataFrame.from_dict(gs.cv_results_)
    training_jobs_df = training_jobs_df.rename(
        columns={
            'mean_test_score': 'FinalObjectiveValue',
            'param_classifier__n_neighbors': 'n_neighbors',
            'param_classifier__weights': 'weights',
            'param_classifier__algorithm': 'algorithm',
            'param_classifier__leaf_size': 'leaf_size',
            'param_classifier__p': 'p',
            'param_classifier__metric': 'metric'
        })
    training_jobs_df['n_neighbors'] = training_jobs_df['n_neighbors'].astype(int)
    training_jobs_df['weights'] = training_jobs_df['weights'].astype(str)
    training_jobs_df['algorithm'] = training_jobs_df['algorithm'].astype(str)
    training_jobs_df['leaf_size'] = training_jobs_df['leaf_size'].astype(int)
    training_jobs_df['p'] = training_jobs_df['p'].astype(int)
    training_jobs_df['metric'] = training_jobs_df['metric'].astype(str)

    tabs = []
    for hyperparam in knn_config.keys():
        is_numeric = is_numeric_dtype(list(knn_config[hyperparam].items())[0][1])
        if is_numeric:
            p = hyperparam_hist(training_jobs_df, hyperparam)
        else:
            p = hyperparam_categorical(training_jobs_df, hyperparam)
        tabs.append(TabPanel(child=p, title=hyperparam))
    tabs = Tabs(tabs=tabs)

    output_file(f'figures/{dataset}/hyperparameter_plots/knn_hyperparameters.html')
    save(tabs)

    plt = generate_learning_curves(gs, knn_grid(), dataset)
    plt.savefig(f'figures/{dataset}/learning_curves/knn_learning_curves.png')
    plt.clf()

    return


if __name__ == '__main__':
    times = []
    start = time.time()

    if not os.path.exists('models/steel-plate-faults/knn.pkl') and not os.path.isdir('models/steel-plate-faults/knn.pkl'):
        for dataset in dataset_choice:
            dataset_load_start = time.time()
            df = load_data(dataset)
            dataset_load_end = time.time()
            dataset_total_time = dataset_load_end - dataset_load_start
            times.append(f'{dataset}')
            times.append(dataset_total_time)

            if dataset == 'breast-cancer':
                preprocess_start = time.time()
                preprocess_bc_data(df)
                preprocess_end = time.time()
                preprocess_total_time = preprocess_end - preprocess_start
                times.append(f'{dataset}')
                times.append(preprocess_total_time)

            if dataset == 'steel-plate-faults':
                preprocess_start = time.time()
                preprocess_sp_data(df)
                preprocess_end = time.time()
                preprocess_total_time = preprocess_end - preprocess_start
                times.append(f'{dataset}')
                times.append(preprocess_total_time)

            X_train, y_train, X_test, y_test = get_data(dataset)
            for algorithm in algorithms:
                if algorithm == 'DecisionTreeClassifier':
                    dt_start = time.time()
                    build_decision_tree_model(dataset, X_train, y_train, dt_grid())
                    dt_end = time.time()
                    dt_total = dt_end - dt_start
                    times.append(dt_total)

                    dt_analyze_start = time.time()
                    analyze_decision_tree_model(X_test, y_test, dataset)
                    dt_analyze_end = time.time()
                    dt_analyze_total = dt_analyze_end - dt_analyze_start
                    times.append(dt_analyze_total)

                if algorithm == 'MLPClassifier':
                    mlp_start = time.time()
                    build_mlp_model(dataset, X_train, y_train)
                    mlp_end = time.time()
                    ml_total = mlp_end - mlp_start
                    times.append(ml_total)

                    mlp_analyze_start = time.time()
                    analyze_mlp_model(X_test, y_test, dataset)
                    mlp_analyze_end = time.time()
                    mlp_analyze_total = mlp_analyze_end - mlp_analyze_start
                    times.append(mlp_analyze_total)

                if algorithm == 'GradientBoostingClassifier':
                    gb_start = time.time()
                    build_gradient_boosted_model(dataset, X_train, y_train, gb_grid())
                    gb_end = time.time()
                    gb_total = gb_end - gb_start
                    times.append(gb_total)

                    gb_analyze_start = time.time()
                    analyze_gradient_boosted_model(X_test, y_test, dataset)
                    gb_analyze_end = time.time()
                    gb_analyze_total = gb_analyze_end - gb_analyze_start
                    times.append(gb_analyze_total)

                if algorithm == 'SVC':
                    svc_start = time.time()
                    build_svc_model(dataset, X_train, y_train, svc_grid())
                    svc_end = time.time()
                    svc_total = svc_end - svc_start
                    times.append(svc_total)

                    svc_analyze_start = time.time()
                    analyze_svc_model(X_test, y_test, dataset)
                    svc_analyze_end = time.time()
                    svc_analyze_total = svc_analyze_end - svc_analyze_start
                    times.append(svc_analyze_total)

                if algorithm == 'KNeighborsClassifier':
                    knn_start = time.time()
                    build_knn_model(dataset, X_train, y_train, knn_grid())
                    knn_end = time.time()
                    knn_total = knn_end - knn_start
                    times.append(knn_total)

                    knn_analyze_start = time.time()
                    analyze_knn_model(X_test, y_test, dataset)
                    knn_analyze_end = time.time()
                    knn_analyze_total = knn_analyze_end - knn_analyze_start
                    times.append(knn_analyze_total)
    else:
        for dataset in dataset_choice:
            X_train, y_train, X_test, y_test = get_data(dataset)
            analyze_decision_tree_model(X_test, y_test, dataset)
            analyze_mlp_model(X_test, y_test, dataset)
            analyze_gradient_boosted_model(X_test, y_test, dataset)
            analyze_svc_model(X_test, y_test, dataset)
            analyze_knn_model(X_test, y_test, dataset)

    end = time.time()
    total_time = end - start
    times.append(total_time)

    with open('times.txt', 'w') as f:
        f.write(str(times))
