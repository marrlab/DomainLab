import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional, Tuple
from ast import literal_eval

from numpy.linalg import LinAlgError


def gen_benchmark_plots(agg_results: str, output_dir: str):
    '''
    generate the benchmark plots from a csv file containing the aggregated restults.
    The csv file must have the columns:
    [param_index, task, algo, epos, te_d, seed, params, ...]
    all columns after seed are intrepreted as objectives of the results, they can e.g.
    be acc, precision, recall, specificity, f1, auroc.

    agg_results: path to the csv file
    output_dir: path to a folder which shall contain the results
    '''
    df = pd.read_csv(agg_results, index_col=False, converters={'params': literal_eval}, skipinitialspace=True)
    # crop param_index and task from the dataframe
    dataframe = df.iloc[:, 2:]
    gen_plots(dataframe, output_dir)



def gen_plots(dataframe: pd.DataFrame, output_dir: str):
    '''
    dataframe: dataframe with columns [' algo', ' epos', ' te_d', ' seed', ' params', ' acc', ' precision', ... ]
    '''
    os.makedirs(output_dir, exist_ok=True)

    # stochastic and systematic variation
    # TODO Add plot 1 and 2 here


    #scatterplot matrices
    scatterplot_matrix(dataframe, file=output_dir + '/sp_matrix_reg.png', reg=True, distinguish_param_setups=False)
    scatterplot_matrix(dataframe, file=output_dir + '/sp_matrix.png', reg=False, distinguish_param_setups=False)
    scatterplot_matrix(dataframe, file=output_dir + '/sp_matrix_dist_reg.png', reg=True, distinguish_param_setups=True)
    scatterplot_matrix(dataframe, file=output_dir + '/sp_matrix_dist.png', reg=False, distinguish_param_setups=True)

    # radar plots
    radar_plot(dataframe, file=output_dir + '/radar_dist.png', distinguish_hyperparam=True)
    radar_plot(dataframe, file=output_dir + '/radar.png', distinguish_hyperparam=False)

    # scatter plots for parirs of objectives
    os.makedirs(output_dir + '/scatterpl', exist_ok=True)
    obj = dataframe.columns[5:]
    for i in range(len(obj)):
       for j in range(i+1, len(obj)):
           try:
               scatterplot(dataframe, obj[i], obj[j],
                           file=output_dir + '/scatterpl/' + obj[i] + '_' + obj[j] + '.png')
           except IndexError:
               print(f'WARNING: disabling kde because cov matrix is singular for objectives '
                     f'{obj[i]} & {obj[j]}')
               scatterplot(dataframe, obj[i], obj[j],
                           file=output_dir + '/scatterpl/' + obj[i] + '_' + obj[j] + '.png',
                           kde=False)

    # create plots for the different algortihms
    for algorithm in dataframe['algo'].unique():
        os.makedirs(output_dir + '/' + str(algorithm), exist_ok=True)
        dataframe_algo = dataframe[dataframe['algo'] == algorithm]

        # stochastic and systematic variation
        # TODO IDEA: is it usefull to do the stochastic and systematic for the filtered dataframe?


        # scatterplot matrices
        scatterplot_matrix(dataframe_algo,
                           file=output_dir + '/' + str(algorithm) + '/sp_matrix_reg.png',
                           reg=True, distinguish_param_setups=False)
        scatterplot_matrix(dataframe_algo,
                           file=output_dir + '/' + str(algorithm) + '/sp_matrix.png',
                           reg=False, distinguish_param_setups=False)
        scatterplot_matrix(dataframe_algo,
                           file=output_dir + '/' + str(algorithm) + '/sp_matrix_dist_reg.png',
                           reg=True, distinguish_param_setups=True)
        scatterplot_matrix(dataframe_algo,
                           file=output_dir + '/' + str(algorithm) + '/sp_matrix_dist.png',
                           reg=False, distinguish_param_setups=True)

        # radar plots
        radar_plot(dataframe_algo,
                   file=output_dir + '/' + str(algorithm) + '/radar_dist.png',
                   distinguish_hyperparam=True)
        radar_plot(dataframe_algo,
                   file=output_dir + '/' + str(algorithm) + '/radar.png',
                   distinguish_hyperparam=False)

        # scatter plots for parirs of objectives
        os.makedirs(output_dir + '/' + str(algorithm) + '/scatterpl', exist_ok=True)
        obj = dataframe_algo.columns[5:]
        for i in range(len(obj)):
            for j in range(i + 1, len(obj)):
                try:
                    scatterplot(dataframe_algo, obj[i], obj[j],
                                file=output_dir + '/' + str(algorithm) +
                                     '/scatterpl/' + obj[i] + '_' + obj[j] + '.png',
                                distinguish_hyperparam=True)
                except IndexError:
                    print(f'WARNING: disabling kde because cov matrix is singular for objectives '
                          f'{obj[i]} & {obj[j]}')
                    scatterplot(dataframe_algo, obj[i], obj[j],
                                file=output_dir + '/' + str(algorithm) +
                                     '/scatterpl/' + obj[i] + '_' + obj[j] + '.png',
                                kde=False,
                                distinguish_hyperparam=True)





def scatterplot_matrix(dataframe_in, file=None, reg=True, distinguish_param_setups=True):
    '''
    dataframe: dataframe containing the data with columns
        ['algo', 'epos', 'seed', 'mname', 'hyperparameter', obj1, ..., obj2]
    file: filename to save the plots (if None, the plot will not be saved)
    reg: if True a regression line will be plotted over the data
    distinguish_param_setups: if True the plot will not only distinguish between models,
        but also between the parameter setups
    '''
    dataframe = dataframe_in.copy()
    index = list(range(5, dataframe.shape[1]))
    if distinguish_param_setups:
        dataframe_ = dataframe.iloc[:, index]
        dataframe_.insert(0, 'label',
                          dataframe['algo'].astype(str) + ', ' +
                          dataframe['params'].astype(str))
    else:
        index_ = list(range(5, dataframe.shape[1]))
        index_.insert(0, 0)
        dataframe_ = dataframe.iloc[:, index_]

    if reg:
        if not distinguish_param_setups:
            g = sns.pairplot(data=dataframe_, hue='algo', corner=True, kind='reg')
        else:
            g = sns.pairplot(data=dataframe_, hue='label', corner=True, kind='reg')
    else:
        if not distinguish_param_setups:
            g = sns.pairplot(data=dataframe_, hue='algo', corner=True)
        else:
            g = sns.pairplot(data=dataframe_, hue='label', corner=True)

    for i in range(len(index)):
        for j in range(len(index)):
            if i >= j:
                g.axes[i, j].set_xlim((-0.1, 1.1))
                for k in range(j):
                    g.axes[j, k].set_ylim((-0.1, 1.1))

    g.fig.set_size_inches(12.5, 12)
    sns.move_legend(g, loc='upper right', bbox_to_anchor=(1., 1.), ncol=1)
    plt.tight_layout()

    if file is not None:
        plt.savefig(file, dpi=300)



def scatterplot(dataframe_in, obj1, obj2, file=None, kde=True, distinguish_hyperparam=False):
    dataframe = dataframe_in.copy()
    dataframe['params'] = dataframe['params'].astype(str)

    if distinguish_hyperparam:
        if kde:
            g = sns.jointplot(data=dataframe, x=obj1, y=obj2, hue='params',
                              xlim=(-0.1, 1.1), ylim=(-0.1, 1.1), kind='kde',
                              zorder=0, levels=8, alpha=0.35, warn_singular=False)
            gg = sns.scatterplot(data=dataframe, x=obj1, y=obj2, hue='params',
                                 ax=g.ax_joint)
        else:
            g = sns.jointplot(data=dataframe, x=obj1, y=obj2, hue='params',
                              xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
            gg = g.ax_joint
    else:
        if kde:
            g = sns.jointplot(data=dataframe, x=obj1, y=obj2, hue='algo',
                              xlim=(-0.1, 1.1), ylim=(-0.1, 1.1), kind='kde',
                              zorder=0, levels=8, alpha=0.35, warn_singular=False)
            gg = sns.scatterplot(data=dataframe, x=obj1, y=obj2, hue='algo', style='params',
                                 ax=g.ax_joint)
        else:
            g = sns.jointplot(data=dataframe, x=obj1, y=obj2, hue='algo',
                              xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
            gg = sns.scatterplot(data=dataframe, x=obj1, y=obj2, style='params',
                                 ax=g.ax_joint)

    gg.set_aspect('equal')
    gg.legend(fontsize=6, loc='best')

    if file is not None:
        plt.savefig(file, dpi=300)


def max_0_x(x):
    return max(0, x)

def radar_plot(dataframe_, file=None, distinguish_hyperparam=True):
    dataframe = dataframe_.copy()
    if distinguish_hyperparam:
        dataframe.insert(0, 'label',
                         dataframe['algo'].astype(str) + ', ' +
                         dataframe['params'].astype(str))
    else:
        dataframe.insert(0, 'label', dataframe['algo'])
    index = list(range(6, dataframe.shape[1]))
    num_lines = len(dataframe['label'].unique())
    _, ax = plt.subplots(figsize=(9, 9 + (0.28 * num_lines)), subplot_kw=dict(polar=True))
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
        ax.plot(angles_, mean,
                color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                linewidth=2, label=algo_name)

        # Fill it in.
        ax.fill_between(angles_, list(map(max_0_x, mean - std)), y2 = mean + std,
                        color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                        alpha=0.1)
        num += 1
        num = num % len(list(plt.rcParams["axes.prop_cycle"]))

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles), dataframe.columns[index])

    ax.set_ylim((0, 1))

    plt.legend(loc='lower right', bbox_to_anchor=(1., 1.035),
               ncol=1, fontsize=10)
    #plt.tight_layout()

    if file is not None:
        plt.savefig(file, dpi=300)


def box_plot(
    data: pd.DataFrame,
    metrics: Union[str, List],
    figsize: Optional[Tuple[float, float]] = None,
    hyperparamter_filter: Optional[Union[str, List[str], List[dict]]] = None,
    plot_pooled: Optional[bool] = True,
    groupby_hyperparamter: Optional[bool] = True
):
    if hyperparamter_filter:
        pass
    data['params'] = data['params'].astype(str)
    if isinstance(metrics, str):
        metrics = [metrics]
    num_plots = len(metrics) * (int(plot_pooled == True) + int(groupby_hyperparamter == True))
    print(num_plots)
    data = data[['algo', 'epos', 'seed', 'params'] + metrics]
    if figsize is None:
        #figsize =
        pass
    for i, item in enumerate(metrics, start=1):
        fig = plt.figure(figsize=[6, 10])
        if plot_pooled:
            plt.subplot(num_plots, 1, 2*i-1)
            sns.boxplot(data=data, x="algo", y=item)
        if groupby_hyperparamter:
            plt.subplot(num_plots, 1, 2*i)
            sns.boxplot(data=data, x="algo", y=item, hue='params')
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    diva01 = ['diva', 5, 0, 'mname_debug_e_mini_vlcs_te_caltech_diva_b05f8956_not_commited_2023md_01md_18_09_26_44_seed_0',
              {'a': 1, 'b':1}, 0.77397263, 0.82765615, 0.59528303, 0.59528303, 0.5953641, 0.74150944]
    diva02 = ['diva', 5, 1, 'mname_debug_e_mini_vlcs_te_caltech_diva_b05f8956_not_commited_2023md_01md_18_09_26_44_seed_0',
              {'a': 1, 'b':1}, 0.851917, 0.7569983, 0.8873113, 0.8173113, 0.7985473, 0.8769811]
    diva03 = ['diva', 5, 2, 'mname_debug_e_mini_vlcs_te_caltech_diva_b05f8956_not_commited_2023md_01md_18_09_26_44_seed_0',
              {'a': 1, 'b':1}, 0.84, 0.85, 0.615, 0.62, 0.62, 0.74]
    diva11 = ['diva', 5, 0, 'mname_debug_e_mini_vlcs_te_caltech_diva_b05f8956_not_commited_2023md_01md_18_09_26_44_seed_0',
              {'a': 1, 'b':2}, 0.85397263, 0.95765615, 0.67528303, 0.6728303, 0.6853641, 0.81150944]
    diva12 = ['diva', 5, 1, 'mname_debug_e_mini_vlcs_te_caltech_diva_b05f8956_not_commited_2023md_01md_18_09_26_44_seed_0',
              {'a': 1, 'b':2}, 0.971917, 0.8869983, 0.9273113, 0.9273113, 0.8285473, 0.9369811]
    diva13 = ['diva', 5, 2, 'mname_debug_e_mini_vlcs_te_caltech_diva_b05f8956_not_commited_2023md_01md_18_09_26_44_seed_0',
              {'a': 1, 'b':2}, 0.92, 0.91, 0.74, 0.75, 0.67, 0.8]

    hduva01 = ['hduva', 5, 0, 'mname_mnistcolor10_te_rgb_214_39_40_rgb_148_103_189_hduva_b799a16c_not_commited_2023md_01md_22_09_56_36_seed_0',
              {'c': 0, 'b':1}, 0.8251667, 0.104563385, 0.10817383, 0.90109926, 0.08242064, 0.519054]
    hduva02 = ['hduva', 5, 1, 'mname_mnistcolor10_te_rgb_214_39_40_rgb_148_103_189_hduva_b799a16c_not_commited_2023md_01md_22_09_56_36_seed_0',
              {'c': 0, 'b':1}, 0.97535366, 0.88569593, 0.87402904, 0.98618114, 0.87845683, 0.99118155]
    hduva03 = ['hduva', 5, 2, 'mname_mnistcolor10_te_rgb_214_39_40_rgb_148_103_189_hduva_b799a16c_not_commited_2023md_01md_22_09_56_36_seed_0',
              {'c': 0, 'b':1}, 0.82422227, 0.10658247, 0.1084048, 0.9009563, 0.08635866, 0.51927835]
    hduva11 = ['hduva', 5, 0, 'mname_mnistcolor10_te_rgb_214_39_40_rgb_148_103_189_hduva_b799a16c_not_commited_2023md_01md_22_09_56_36_seed_0',
              {'c': -1, 'b':0}, 0.8551667, 0.204563385, 0.20817383, 0.80109926, 0.15242064, 0.559054]
    hduva12 = ['hduva', 5, 1, 'mname_mnistcolor10_te_rgb_214_39_40_rgb_148_103_189_hduva_b799a16c_not_commited_2023md_01md_22_09_56_36_seed_0',
              {'c': -1, 'b':0}, 0.57535366, 0.52569593, 0.41402904, 0.68618114, 0.57845683, 0.89118155]
    hduva13 = ['hduva', 5, 2, 'mname_mnistcolor10_te_rgb_214_39_40_rgb_148_103_189_hduva_b799a16c_not_commited_2023md_01md_22_09_56_36_seed_0',
              {'c': -1, 'b':0}, 0.72422227, 0.20658247, 0.2084048, 0.7009563, 0.18635866, 0.61927835]


    # dataset containing results from 2 algorithms, both were run with two different initialisations of the params
    # for each hyperparameter initialisation three runs with different random seeds are included
    dummy_dataframe = pd.DataFrame([diva01, diva02, diva03, diva11, diva12, diva13,
                                    hduva01, hduva02, hduva03, hduva11, hduva12, hduva13],
        columns=['algo', 'epos', 'seed', 'mname', 'hyperparameter',
                 'acc', 'precision', 'recall', 'specificity', 'f1', 'auroc'])




    #gen_benchmark_plots('results.csv', 'res_outp')
    gen_benchmark_plots('aggret_res_test2', 'outp2')
