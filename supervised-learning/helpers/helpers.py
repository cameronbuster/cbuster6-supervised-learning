import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bokeh.plotting import figure
from bokeh.models import HoverTool
from scipy.stats import norm


bc_str_cols = [
    'deg-malig',
    'age',
    'menopause',
    'tumor-size',
    'inv-nodes',
    'node-caps',
    'breast',
    'breast-quad',
    'irradiat'
]


def bc_feature_create(df):
    df = df.copy()
    return df


def bc_feature_drop(df):
    features = [
        'deg-malig',
        'age',
        'menopause',
        'tumor-size',
        'inv-nodes',
        'node-caps',
        'breast',
        'breast-quad',
        'irradiat'
    ]

    bc_reassign_data_types(df)
    df = df[features]
    return df


sp_num_cols = [
    'X_Minimum',
    'X_Maximum',
    'Y_Minimum',
    'Y_Maximum',
    'Pixels_Areas',
    'X_Perimeter',
    'Y_Perimeter',
    'Sum_of_Luminosity',
    'Minimum_of_Luminosity',
    'Maximum_of_Luminosity',
    'Length_of_Conveyer',
    'TypeOfSteel_A300',
    'TypeOfSteel_A400',
    'Steel_Plate_Thickness',
    'Edges_Index',
    'Empty_Index',
    'Square_Index',
    'Outside_X_Index',
    'Edges_X_Index',
    'Edges_Y_Index',
    'Outside_Global_Index',
    'LogOfAreas',
    'Log_X_Index',
    'Log_Y_Index',
    'Orientation_Index',
    'Luminosity_Index',
    'SigmoidOfAreas'
]


def sp_feature_create(df):
    df = df.copy()
    return df


def sp_feature_drop(df):
    features = [
        'X_Minimum',
        'X_Maximum',
        'Y_Minimum',
        'Y_Maximum',
        'Pixels_Areas',
        'X_Perimeter',
        'Y_Perimeter',
        'Sum_of_Luminosity',
        'Minimum_of_Luminosity',
        'Maximum_of_Luminosity',
        'Length_of_Conveyer',
        'TypeOfSteel_A300',
        'TypeOfSteel_A400',
        'Steel_Plate_Thickness',
        'Edges_Index',
        'Empty_Index',
        'Square_Index',
        'Outside_X_Index',
        'Edges_X_Index',
        'Edges_Y_Index',
        'Outside_Global_Index',
        'LogOfAreas',
        'Log_X_Index',
        'Log_Y_Index',
        'Orientation_Index',
        'Luminosity_Index',
        'SigmoidOfAreas'
    ]

    sp_reassign_data_types(df)
    df = df[features]
    return df


def bc_reassign_data_types(df):
    df[bc_str_cols] = df[bc_str_cols].fillna(0)
    df[bc_str_cols] = df[bc_str_cols].fillna(str)


def sp_reassign_data_types(df):
    df[sp_num_cols] = df[sp_num_cols].fillna(0)
    df[sp_num_cols] = df[sp_num_cols].fillna(float)


def get_data(dataset):
    if dataset == 'breast-cancer':
        X_train = pd.read_csv(f'../datasets/{dataset}/X_train.csv')
        X_test = pd.read_csv(f'../datasets/{dataset}/X_test.csv')
        y_train = pd.read_csv(f'../datasets/{dataset}/y_train.csv')
        y_test = pd.read_csv(f'../datasets/{dataset}/y_test.csv')
    if dataset == 'steel-plate-faults':
        X_train = pd.read_csv(f'../datasets/{dataset}/X_train.csv')
        X_test = pd.read_csv(f'../datasets/{dataset}/X_test.csv')
        y_train = pd.read_csv(f'../datasets/{dataset}/y_train.csv')
        y_test = pd.read_csv(f'../datasets/{dataset}/y_test.csv')

    return X_train, y_train, X_test, y_test


def hyperparam_hist(training_jobs_df, hyperparam, scaling_type='Auto'):
    hyperparam_vals = training_jobs_df[hyperparam].values
    objective_values = training_jobs_df['FinalObjectiveValue'].values
    objective_values = np.nan_to_num(objective_values, copy=True, nan=0.0, posinf=None, neginf=None)

    n_bins = min([20, int(len(hyperparam_vals) / 10)])
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.percentile(hyperparam_vals, quantiles * 100)
    bins[-1] = bins[-1] + 1e-8

    # Get counts within each bin
    binids = np.digitize(hyperparam_vals, bins) - 1
    bin_total = np.bincount(binids, minlength=len(bins))[:-1]
    bin_sums = np.bincount(binids, weights=hyperparam_vals, minlength=len(bins))[:-1]
    nonzero = bin_total != 0

    # Get counts within each bin
    mean_hyperparams = (bin_sums[nonzero] / bin_total[nonzero])
    max_objective_values = []
    mean_objective_values = []
    std_objective_values = []
    for i, nonempty in enumerate(nonzero):
        if nonempty:
            max_objective_values.append(np.max(objective_values[binids == i]))
            mu, std = norm.fit(objective_values[binids == i])
            mean_objective_values.append(mu)
            std_objective_values.append(std)

    confidence_intervals = norm.interval(0.95, loc=mean_objective_values, scale=std_objective_values)
    confidence_intervals_x = list(zip(mean_hyperparams, mean_hyperparams))
    confidence_intervals_y = list(zip(confidence_intervals[0], confidence_intervals[1]))

    hover_tools = HoverTool(tooltips=[
        ('max FinalObjectiveValue', '$y'),
        ('mean hyperparameter value', '$x'),
    ])
    standard_tools = 'pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset'
    tools = [hover_tools, standard_tools]

    x_axis_type = 'auto'
    if scaling_type == "Logarithmic":
        x_axis_type = "log"

    p = figure(
        width=1000,
        height=700,
        tools=tools,
        x_axis_type=x_axis_type
    )
    #     p.circle(mean_hyperparams, mean_objective_values, color='red', size=10, alpha=0.5)
    #     p.multi_line(confidence_intervals_x, confidence_intervals_y, color='red', alpha=0.5)
    p.line(mean_hyperparams, max_objective_values)
    scatter = p.circle(mean_hyperparams, max_objective_values, size=10)
    hover_tools.renderers = [scatter]
    p.xaxis.axis_label = hyperparam
    p.yaxis.axis_label = 'FinalObjectiveValue'
    return p


def hyperparam_categorical(training_jobs_df, hyperparam):
    hyperparam_vals = np.sort(training_jobs_df[hyperparam].unique())
    df_max = training_jobs_df[[hyperparam, 'FinalObjectiveValue']].groupby(hyperparam).agg(['mean', 'max', np.std])
    df_max.columns = df_max.columns.get_level_values(1)
    df_max = df_max.reset_index()
    df_max['ci_upper'] = df_max['mean'] + 2 * df_max['std']
    df_max['ci_lower'] = df_max['mean'] - 2 * df_max['std']
    confidence_intervals_x = list(zip(df_max[hyperparam], df_max[hyperparam]))
    confidence_intervals_y = list(zip(df_max['ci_lower'], df_max['ci_upper']))

    hover_tools = HoverTool(tooltips=[
        ('max FinalObjectiveValue', '@max'),
        ('mean hyperparameter value', f'@{hyperparam}'),
    ])
    standard_tools = 'pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset,save'
    tools = [hover_tools, standard_tools]

    p = figure(
        width=1000,
        height=700,
        x_range=df_max[hyperparam].unique(),
        tools=tools
    )
    #     p.circle(hyperparam, 'mean', color='red', size=10, alpha=0.5, source=df_max)
    #     p.multi_line(confidence_intervals_x, confidence_intervals_y, color='red', alpha=0.5)
    scatter = p.circle(hyperparam, 'max', size=10, source=df_max)
    hover_tools.renderers = [scatter]
    p.xaxis.axis_label = hyperparam
    p.yaxis.axis_label = 'FinalObjectiveValue'
    return p


def generate_learning_curves(gs, grid_params, dataset):
    df = pd.DataFrame(gs.cv_results_)
    results = ['mean_test_score',
               'mean_train_score',
               'std_test_score',
               'std_train_score']

    # https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation
    def pooled_var(stds):
        n = 5  # size of each group
        return np.sqrt(sum((n - 1) * (stds ** 2)) / len(stds) * (n - 1))

    fig, axes = plt.subplots(1, len(grid_params),
                             figsize=(5 * len(grid_params), 7),
                             sharey='row')
    axes[0].set_ylabel("Score", fontsize=14)
    lw = 2

    for idx, (param_name, param_range) in enumerate(grid_params.items()):
        grouped_df = df.groupby(f'param_{param_name}')[results] \
            .agg({'mean_train_score': 'mean',
                  'mean_test_score': 'mean',
                  'std_train_score': pooled_var,
                  'std_test_score': pooled_var})

        previous_group = df.groupby(f'param_{param_name}')[results]
        axes[idx].set_xlabel(param_name, fontsize=14)
        axes[idx].set_ylim(0.0, 1.1)
        if param_name == 'classifier__hidden_layer_sizes':
            param_range = [str(param) for param in param_range if isinstance(param, tuple)]
            param_range = [param.replace('(', '').replace(')', '').replace(',', '').replace(' ', '') for param in param_range]
            param_range = [int(param) for param in param_range if isinstance(param, str)]
        axes[idx].plot(param_range, grouped_df['mean_train_score'], label="Training score",
                       color="darkorange", lw=lw)
        axes[idx].fill_between(param_range, grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                               grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
                               color="darkorange", lw=lw)
        axes[idx].plot(param_range, grouped_df['mean_test_score'], label="Cross-validation score",
                       color="navy", lw=lw)
        axes[idx].fill_between(param_range, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                               grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
                               color="navy", lw=lw)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Learning curves', fontsize=14)
    fig.legend(handles, labels, loc=8, ncol=2, fontsize=14)

    fig.subplots_adjust(bottom=0.25, top=0.85)
    return plt
