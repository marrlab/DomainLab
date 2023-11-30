'''
generate the benchmark plots by calling the gen_bencmark_plots(...) function
'''
import os
from ast import literal_eval   # literal_eval can safe evaluate python expression
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
from domainlab.utils.logger import Logger

matplotlib.use('Agg')

# header of the csv file:
# param_index, task, algo, epos, te_d, seed, params, acc, precision, recall, specificity, f1, auroc

COLNAME_METHOD = "method"
COLNAME_IDX_PARAM = "param_index"
COLNAME_PARAM = "params"
G_DF_TASK_COL = 1   # column in which the method name is saved
G_DF_PLOT_COL_METRIC_START = 9   # first 0-6 columns are not metric



def gen_benchmark_plots(agg_results: str, output_dir: str, use_param_index: bool = True):
    '''
    generate the benchmark plots from a csv file containing the aggregated restults.
    The csv file must have the columns:
    [param_index, task, algo, epos, te_d, seed, params, ...]
    all columns after seed are intrepreted as objectives of the results, they can e.g.
    be acc, precision, recall, specificity, f1, auroc.

    agg_results: path to the csv file
    output_dir: path to a folder which shall contain the results
    skip_gen: Skips the actual plotting, used to speed up testing.
    '''
    raw_df = pd.read_csv(agg_results, index_col=False,
                         converters={COLNAME_PARAM: literal_eval},
                         # literal_eval can safe evaluate python expression
                         skipinitialspace=True)

    raw_df[COLNAME_PARAM] = round_vals_in_dict(raw_df[[COLNAME_IDX_PARAM, COLNAME_PARAM]],
                                               use_param_index)

    # generating plot
    gen_plots(raw_df, output_dir, use_param_index)


def round_vals_in_dict(df_column_in, use_param_index):
    '''
    replaces the dictionary by a string containing only the significant digits of the hyperparams
    or (if use_param_index = True) by the parameter index
    df_column_in: columns of the dataframe containing the param index and the dictionary of
        hyperparams in the form [param_index, params]
    use_param_index: usage of param_index instead of exact values
    '''
    df_column = df_column_in.copy()
    df_column_out = df_column_in[COLNAME_IDX_PARAM].copy()
    df_column_out = df_column_out.astype(str)
    for i in range(df_column.shape[0]):
        if not use_param_index:
            string = ''
            for num, val in enumerate(list(df_column[COLNAME_PARAM][i].values())):
                key = list(df_column[COLNAME_PARAM][i].keys())[num]
                val = np.format_float_scientific(val, precision=1, unique=False, trim='0')
                string += str(key) + ': ' + str(val) + ', '
            df_column_out[i] = string[:-2]  # remove ', ' from the end of the string
        else:
            string = 'idx: ' + str(df_column[COLNAME_IDX_PARAM][i])
            df_column_out[i] = string
    return df_column_out


def gen_plots(dataframe: pd.DataFrame, output_dir: str, use_param_index: bool):
    '''
    dataframe: dataframe with columns
    ['param_index','task',' algo',' epos',' te_d',' seed',' params',' acc','precision',...]
    '''
    os.makedirs(output_dir, exist_ok=True)
    obj = dataframe.columns[G_DF_PLOT_COL_METRIC_START:]

    # boxplots
    for objective in obj:
        boxplot(dataframe, objective, file=output_dir + '/variational_plots/' + objective)

    # scatterplot matrices
    scatterplot_matrix(dataframe, use_param_index,
                       file=output_dir + '/sp_matrix_reg.png',
                       kind='reg', distinguish_param_setups=False)
    scatterplot_matrix(dataframe, use_param_index,
                       file=output_dir + '/sp_matrix.png',
                       kind='scatter', distinguish_param_setups=False)
    scatterplot_matrix(dataframe, use_param_index,
                       file=output_dir + '/sp_matrix_dist_reg.png',
                       kind='reg', distinguish_param_setups=True)
    scatterplot_matrix(dataframe, use_param_index,
                       file=output_dir + '/sp_matrix_dist.png',
                       kind='scatter', distinguish_param_setups=True)

    # radar plots
    radar_plot(dataframe, file=output_dir + '/radar_dist.png', distinguish_hyperparam=True)
    radar_plot(dataframe, file=output_dir + '/radar.png', distinguish_hyperparam=False)

    # scatter plots for parirs of objectives
    os.makedirs(output_dir + '/scatterpl', exist_ok=True)
    for i, obj_i in enumerate(obj):
        for j in range(i+1, len(obj)):
            try:
                scatterplot(dataframe, [obj_i, obj[j]],
                            file=output_dir + '/scatterpl/' + obj_i + '_' + obj[j] + '.png')
            except IndexError:
                logger = Logger.get_logger()
                logger.warning(f'disabling kde because cov matrix is singular for objectives '
                               f'{obj_i} & {obj[j]}')
                scatterplot(dataframe, [obj_i, obj[j]],
                            file=output_dir + '/scatterpl/' + obj_i + '_' + obj[j] + '.png',
                            kde=False)

    # create plots for the different algortihms
    for algorithm in dataframe[COLNAME_METHOD].unique():
        os.makedirs(output_dir + '/' + str(algorithm), exist_ok=True)
        dataframe_algo = dataframe[dataframe[COLNAME_METHOD] == algorithm]

        # boxplots
        for objective in obj:
            boxplot(dataframe_algo, objective,
                    file=output_dir + '/' + str(algorithm) + '/variational_plots/' + objective)

        # scatterplot matrices
        scatterplot_matrix(dataframe_algo, use_param_index,
                           file=output_dir + '/' + str(algorithm) + '/sp_matrix_reg.png',
                           kind='reg', distinguish_param_setups=False)
        scatterplot_matrix(dataframe_algo, use_param_index,
                           file=output_dir + '/' + str(algorithm) + '/sp_matrix.png',
                           kind='scatter', distinguish_param_setups=False)
        scatterplot_matrix(dataframe_algo, use_param_index,
                           file=output_dir + '/' + str(algorithm) + '/sp_matrix_dist_reg.png',
                           kind='reg', distinguish_param_setups=True)
        scatterplot_matrix(dataframe_algo, use_param_index,
                           file=output_dir + '/' + str(algorithm) + '/sp_matrix_dist.png',
                           kind='scatter', distinguish_param_setups=True)

        # radar plots
        radar_plot(dataframe_algo, file=output_dir + '/' + str(algorithm) + '/radar_dist.png',
                   distinguish_hyperparam=True)
        radar_plot(dataframe_algo, file=output_dir + '/' + str(algorithm) + '/radar.png',
                   distinguish_hyperparam=False)

        # scatter plots for parirs of objectives
        os.makedirs(output_dir + '/' + str(algorithm) + '/scatterpl', exist_ok=True)
        for i, obj_i in enumerate(obj):
            for j in range(i + 1, len(obj)):
                try:
                    scatterplot(dataframe_algo, [obj_i, obj[j]],
                                file=output_dir + '/' + str(algorithm) +
                                '/scatterpl/' + obj_i + '_' + obj[j] + '.png',
                                distinguish_hyperparam=True)
                except IndexError:
                    logger = Logger.get_logger()
                    logger.warning(f'WARNING: disabling kde because cov matrix is singular '
                                   f'for objectives {obj_i} & {obj[j]}')
                    scatterplot(dataframe_algo, [obj_i, obj[j]],
                                file=output_dir + '/' + str(algorithm) +
                                '/scatterpl/' + obj_i + '_' + obj[j] + '.png',
                                kde=False,
                                distinguish_hyperparam=True)


def scatterplot_matrix(dataframe_in, use_param_index, file=None, kind='reg',
                       distinguish_param_setups=True):
    '''
    dataframe: dataframe containing the data with columns
        [algo, epos, te_d, seed, params, obj1, ..., obj2]
    file: filename to save the plots (if None, the plot will not be saved)
    reg: if True a regression line will be plotted over the data
    distinguish_param_setups: if True the plot will not only distinguish between models,
        but also between the parameter setups
    '''
    dataframe = dataframe_in.copy()
    index = list(range(G_DF_PLOT_COL_METRIC_START, dataframe.shape[1]))
    if distinguish_param_setups:
        dataframe_ = dataframe.iloc[:, index]
        dataframe_.insert(0, 'label',
                          dataframe[COLNAME_METHOD].astype(str) + ', ' +
                          dataframe[COLNAME_PARAM].astype(str))

        g_p = sns.pairplot(data=dataframe_, hue='label', corner=True, kind=kind)
    else:
        index_ = list(range(G_DF_PLOT_COL_METRIC_START, dataframe.shape[1]))
        index_.insert(0, G_DF_TASK_COL)
        dataframe_ = dataframe.iloc[:, index_]

        g_p = sns.pairplot(data=dataframe_, hue=COLNAME_METHOD, corner=True, kind=kind)

    for i in range(len(index)):
        for j in range(len(index)):
            if i >= j:
                g_p.axes[i, j].set_xlim((-0.1, 1.1))
                for k in range(j):
                    g_p.axes[j, k].set_ylim((-0.1, 1.1))

    g_p.fig.set_size_inches(12.5, 12)
    if use_param_index and distinguish_param_setups:
        sns.move_legend(g_p, loc='upper right', bbox_to_anchor=(1., 1.), ncol=3)
    else:
        sns.move_legend(g_p, loc='upper right', bbox_to_anchor=(1., 1.), ncol=1)
    plt.tight_layout()

    if file is not None:
        plt.savefig(file, dpi=300)


def scatterplot(dataframe_in, obj, file=None, kde=True, distinguish_hyperparam=False):
    '''
    dataframe: dataframe containing the data with columns
        [algo, epos, te_d, seed, params, obj1, ..., obj2]
    obj1 & obj2: name of the objectives which shall be plotted against each other
    file: filename to save the plots (if None, the plot will not be saved)
    kde: if True the distribution of the points will be estimated and plotted as kde plot
    distinguish_param_setups: if True the plot will not only distinguish between models,
        but also between the parameter setups
    '''
    obj1, obj2 = obj

    dataframe = dataframe_in.copy()
    dataframe[COLNAME_PARAM] = dataframe[COLNAME_PARAM].astype(str)

    if distinguish_hyperparam:
        if kde:
            g_p = sns.jointplot(data=dataframe, x=obj1, y=obj2, hue=COLNAME_PARAM,
                                xlim=(-0.1, 1.1), ylim=(-0.1, 1.1), kind='kde',
                                zorder=0, levels=8, alpha=0.35, warn_singular=False)
            gg_p = sns.scatterplot(data=dataframe, x=obj1, y=obj2, hue=COLNAME_PARAM,
                                   ax=g_p.ax_joint)
        else:
            g_p = sns.jointplot(data=dataframe, x=obj1, y=obj2, hue=COLNAME_PARAM,
                                xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
            gg_p = g_p.ax_joint
    else:
        if kde:
            g_p = sns.jointplot(data=dataframe, x=obj1, y=obj2, hue=COLNAME_METHOD,
                                xlim=(-0.1, 1.1), ylim=(-0.1, 1.1), kind='kde',
                                zorder=0, levels=8, alpha=0.35, warn_singular=False)
            gg_p = sns.scatterplot(data=dataframe, x=obj1, y=obj2, hue=COLNAME_METHOD,
                                   style=COLNAME_PARAM,
                                   ax=g_p.ax_joint)
        else:
            g_p = sns.jointplot(data=dataframe, x=obj1, y=obj2, hue=COLNAME_METHOD,
                                xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
            gg_p = sns.scatterplot(data=dataframe, x=obj1, y=obj2, style=COLNAME_PARAM,
                                   ax=g_p.ax_joint)

    gg_p.set_aspect('equal')
    gg_p.legend(fontsize=6, loc='best')

    if file is not None:
        plt.savefig(file, dpi=300)


def max_0_x(x_arg):
    '''
    max(0, x_arg)
    '''
    return max(0, x_arg)


def radar_plot(dataframe_in, file=None, distinguish_hyperparam=True):
    '''
    dataframe_in: dataframe containing the data with columns
        [algo, epos, te_d, seed, params, obj1, ..., obj2]
    file: filename to save the plots (if None, the plot will not be saved)
    distinguish_param_setups: if True the plot will not only distinguish between models,
        but also between the parameter setups
    '''
    dataframe = dataframe_in.copy()
    if distinguish_hyperparam:
        dataframe.insert(0, 'label',
                         dataframe[COLNAME_METHOD].astype(str) + ', ' +
                         dataframe[COLNAME_PARAM].astype(str))
    else:
        dataframe.insert(0, 'label', dataframe[COLNAME_METHOD])
    # we need "G_DF_PLOT_COL_METRIC_START + 1" as we did insert the columns 'label' at index 0
    index = list(range(G_DF_PLOT_COL_METRIC_START + 1, dataframe.shape[1]))
    num_lines = len(dataframe['label'].unique())
    _, axis = plt.subplots(figsize=(9, 9 + (0.28 * num_lines)), subplot_kw=dict(polar=True))
    num = 0

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = list(np.linspace(0, 2 * np.pi, len(dataframe.columns[index]), endpoint=False))
    for algo_name in dataframe['label'].unique():
        mean = dataframe.loc[dataframe['label'] == algo_name].iloc[:, index].mean().to_list()
        std = dataframe.loc[dataframe['label'] == algo_name].iloc[:, index].std().to_list()

        angles_ = angles
        # The plot is a circle, so we need to "complete the loop"
        # and append the start value to the end.
        mean = np.array(mean + mean[:1])
        std = np.array(std + std[:1])
        angles_ = np.array(angles_ + angles_[:1])

        # Draw the outline of the data.
        axis.plot(angles_, mean,
                  color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                  linewidth=2, label=algo_name)

        # Fill it in.
        axis.fill_between(angles_, list(map(max_0_x, mean - std)),
                          y2=mean + std,
                          color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                          alpha=0.1)
        num += 1
        num = num % len(list(plt.rcParams["axes.prop_cycle"]))

    # Fix axis to go in the right order and start at 12 o'clock.
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    axis.set_thetagrids(np.degrees(angles), dataframe.columns[index])

    axis.set_ylim((0, 1))

    plt.legend(loc='lower right', bbox_to_anchor=(1., 1.035),
               ncol=1, fontsize=10)

    if file is not None:
        plt.savefig(file, dpi=300)


def boxplot(dataframe_in, obj, file=None):
    '''
    generate the boxplots
    dataframe_in: dataframe containing the data with columns
        [param_idx, task , algo, epos, te_d, seed, params, obj1, ..., obj2]
    obj: objective to be considered in the plot (needs to be contained in dataframe_in)
    file: foldername to save the plots (if None, the plot will not be saved)
    '''
    boxplot_stochastic(dataframe_in, obj, file=file)
    boxplot_systematic(dataframe_in, obj, file=file)

def boxplot_stochastic(dataframe_in, obj, file=None):
    '''
    generate boxplot for stochastic variation
    dataframe_in: dataframe containing the data with columns
        [param_idx, task , algo, epos, te_d, seed, params, obj1, ..., obj2]
    obj: objective to be considered in the plot (needs to be contained in dataframe_in)
    file: foldername to save the plots (if None, the plot will not be saved)
    '''
    dataframe = dataframe_in.copy()
    os.makedirs(file, exist_ok=True)

    ### stochastic variation
    _, axes = plt.subplots(1, len(dataframe[COLNAME_METHOD].unique()), sharey=True,
                           figsize=(3 * len(dataframe[COLNAME_METHOD].unique()), 6))
    # iterate over all algorithms
    for num, algo in enumerate(list(dataframe[COLNAME_METHOD].unique())):
        # distinguish if the algorithm does only have one param setup or multiple
        if len(dataframe[COLNAME_METHOD].unique()) > 1:
            # generate boxplot and swarmplot
            sns.boxplot(data=dataframe[dataframe[COLNAME_METHOD] == algo],
                        x=COLNAME_IDX_PARAM, y=obj,
                        ax=axes[num], showfliers=False,
                        boxprops={"facecolor": (.4, .6, .8, .5)})
            sns.swarmplot(data=dataframe[dataframe[COLNAME_METHOD] == algo],
                          x=COLNAME_IDX_PARAM, y=obj,
                          legend=False, ax=axes[num])
            # remove legend, set ylim, set x-label and remove y-label
            axes[num].legend([], [], frameon=False)
            axes[num].set_ylim([-0.1, 1.1])
            axes[num].set_xlabel(algo)
            if num != 0:
                axes[num].set_ylabel('')
        else:
            sns.boxplot(data=dataframe[dataframe[COLNAME_METHOD] == algo],
                        x=COLNAME_IDX_PARAM, y=obj,
                        ax=axes, showfliers=False,
                        boxprops={"facecolor": (.4, .6, .8, .5)})
            sns.swarmplot(data=dataframe[dataframe[COLNAME_METHOD] == algo],
                          x=COLNAME_IDX_PARAM, y=obj, hue=COLNAME_IDX_PARAM,
                          legend=False, ax=axes,
                          palette=sns.cubehelix_palette(n_colors=len(
                              dataframe[dataframe[COLNAME_METHOD] == algo]
                              [COLNAME_IDX_PARAM].unique())))
            axes.legend([], [], frameon=False)
            axes.set_ylim([-0.1, 1.1])
            axes.set_xlabel(algo)
        plt.tight_layout()
        if file is not None:
            plt.savefig(file + '/stochastic_variation.png', dpi=300)


def boxplot_systematic(dataframe_in, obj, file=None):
    '''
    generate boxplot for ssystemtic variation
    dataframe_in: dataframe containing the data with columns
        [param_idx, task , algo, epos, te_d, seed, params, obj1, ..., obj2]
    obj: objective to be considered in the plot (needs to be contained in dataframe_in)
    file: foldername to save the plots (if None, the plot will not be saved)
    '''
    dataframe = dataframe_in.copy()
    os.makedirs(file, exist_ok=True)

    ### systematic variation
    _, axes = plt.subplots(1, len(dataframe[COLNAME_METHOD].unique()), sharey=True,
                           figsize=(3 * len(dataframe[COLNAME_METHOD].unique()), 6))

    for num, algo in enumerate(list(dataframe[COLNAME_METHOD].unique())):
        # distinguish if the algorithm does only have one param setup or multiple
        if len(dataframe[COLNAME_METHOD].unique()) > 1:
            # generate boxplot and swarmplot
            sns.boxplot(data=dataframe[dataframe[COLNAME_METHOD] == algo],
                        x=COLNAME_METHOD, y=obj,
                        ax=axes[num], showfliers=False,
                        boxprops={"facecolor": (.4, .6, .8, .5)})
            sns.swarmplot(data=dataframe[dataframe[COLNAME_METHOD] == algo],
                          x=COLNAME_METHOD, y=obj, hue=COLNAME_IDX_PARAM,
                          legend=False, ax=axes[num],
                          palette=sns.cubehelix_palette(n_colors=len(
                              dataframe[dataframe[COLNAME_METHOD] == algo]
                              [COLNAME_IDX_PARAM].unique())))
            # remove legend, set ylim, set x-label and remove y-label
            axes[num].legend([], [], frameon=False)
            axes[num].set_ylim([-0.1, 1.1])
            axes[num].set_xlabel(' ')
            if num != 0:
                axes[num].set_ylabel('')
        else:
            sns.boxplot(data=dataframe[dataframe[COLNAME_METHOD] == algo],
                        x=COLNAME_METHOD, y=obj,
                        ax=axes, showfliers=False,
                        boxprops={"facecolor": (.4, .6, .8, .5)})
            sns.swarmplot(data=dataframe[dataframe[COLNAME_METHOD] == algo],
                          x=COLNAME_METHOD, y=obj, hue=COLNAME_IDX_PARAM,
                          legend=False, ax=axes,
                          palette=sns.cubehelix_palette(n_colors=len(
                              dataframe[dataframe[COLNAME_METHOD] == algo]
                              [COLNAME_IDX_PARAM].unique())))
            axes.legend([], [], frameon=False)
            axes.set_ylim([-0.1, 1.1])
            axes.set_xlabel(' ')
        plt.tight_layout()

    if file is not None:
        plt.savefig(file + '/systematic_variation.png', dpi=300)

