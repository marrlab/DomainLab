import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional, Tuple

def scatterplot_matrix(dataframe, file=None, reg=True, distinguish_param_setups=True):
    '''
    dataframe: dataframe containing the data with columns
        ['algo', 'epos', 'seed', 'mname', 'hyperparameters', obj1, ..., obj2]
    file: filename to save the plots (if None, the plot will not be saved)
    reg: if True a regression line will be plotted over the data
    distinguish_param_setups: if True the plot will not only distinguish between models,
        but also between the parameter setups
    '''
    index = list(range(5, dataframe.shape[1]))
    if distinguish_param_setups:
        dataframe_ = dataframe.iloc[:, index]
        dataframe_.insert(0, 'label',
                          dataframe['algo'].astype(str) + ', ' +
                          dataframe['hyperparameters'].astype(str))
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
                label = dataframe_.columns[j + 1]
                dist = 0.1 * (max(dataframe_[label]) - min(dataframe_[label]))
                g.axes[i, j].set_xlim((min(dataframe_[label]) - dist,
                                       max(dataframe_[label]) + dist))
                for k in range(j):
                    g.axes[j, k].set_ylim((min(dataframe_[label]) - dist,
                                           max(dataframe_[label]) + dist))
    plt.tight_layout()

    if file is not None:
        plt.savefig(file, dpi=300)


def radar_plot(dataframe, file=None):
    dataframe.insert(0, 'label',
                     dataframe['algo'].astype(str) + ', ' +
                     dataframe['hyperparameters'].astype(str))
    index = list(range(6, dataframe.shape[1]))
    _, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    num = 0

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = list(np.linspace(0, 2 * np.pi, len(dataframe.columns[index]), endpoint=False))
    for algo_name in dataframe['label'].unique():
        algo_lab = False

        for line in list(dataframe.loc[dataframe['label'] == algo_name].iloc[:, index].to_numpy()):
            angles_ = angles
            line = list(line)
            # The plot is a circle, so we need to "complete the loop"
            # and append the start value to the end.
            line = line + line[:1]
            angles_ = angles_ + angles_[:1]

            # Draw the outline of the data.
            if algo_lab:
                ax.plot(angles_, line,
                        color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                        linewidth=1)
            else:
                ax.plot(angles_, line,
                        color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                        linewidth=1, label=algo_name)
                algo_lab = True
            # Fill it in.
            ax.fill(angles_, line,
                    color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                    alpha=0.05)
        num += 1
        num = num % len(list(plt.rcParams["axes.prop_cycle"]))

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles), dataframe.columns[index])

    plt.legend()
    plt.tight_layout()

    if file is not None:
        plt.savefig(file, dpi=300)


def radar_plot_per_algo(dataframe, file=None):
    dataframe.insert(0, 'label',
                     dataframe['algo'].astype(str) + ', ' +
                     dataframe['hyperparameters'].astype(str))
    index = list(range(6, dataframe.shape[1]))
    _, ax = plt.subplots(1, len(list(dataframe['algo'].unique())),
                         figsize=(6 * len(list(dataframe['algo'].unique())), 6),
                         subplot_kw=dict(polar=True))

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = list(np.linspace(0, 2 * np.pi, len(dataframe.columns[index]), endpoint=False))
    for idx, algo_name in enumerate(list(dataframe['algo'].unique())):
        algo_dataframe = dataframe.loc[dataframe['algo'] == algo_name]
        num = 0

        for label in algo_dataframe['label'].unique():
            algo_lab = False

            for line in list(algo_dataframe.loc[algo_dataframe['label'] == label].iloc[:, index].to_numpy()):
                angles_ = angles
                line = list(line)
                # The plot is a circle, so we need to "complete the loop"
                # and append the start value to the end.
                line = line + line[:1]
                angles_ = angles_ + angles_[:1]

                # Draw the outline of the data.
                if algo_lab:
                    ax[idx].plot(angles_, line,
                            color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                            linewidth=1)
                else:
                    ax[idx].plot(angles_, line,
                            color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                            linewidth=1, label=label)
                    algo_lab = True
                # Fill it in.
                ax[idx].fill(angles_, line,
                        color=list(plt.rcParams["axes.prop_cycle"])[num]['color'],
                        alpha=0.05)
            num += 1
            num = num % len(list(plt.rcParams["axes.prop_cycle"]))

            # Fix axis to go in the right order and start at 12 o'clock.
            ax[idx].set_theta_offset(np.pi / 2)
            ax[idx].set_theta_direction(-1)

            # Draw axis lines for each angle and label.
            ax[idx].set_thetagrids(np.degrees(angles), dataframe.columns[index])

            ax[idx].legend()
    plt.tight_layout()

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
    data['hyperparameters'] = data['hyperparameters'].astype(str)
    if isinstance(metrics, str):
        metrics = [metrics]
    num_plots = len(metrics) * (int(plot_pooled == True) + int(groupby_hyperparamter == True))
    print(num_plots)
    data = data[['algo', 'epos', 'seed', 'hyperparameters'] + metrics]
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
            sns.boxplot(data=data, x="algo", y=item, hue='hyperparameters')
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
              {'c': -1, 'b':0}, 0.57535366, 0.58569593, 0.47402904, 0.68618114, 0.57845683, 0.89118155]
    hduva13 = ['hduva', 5, 2, 'mname_mnistcolor10_te_rgb_214_39_40_rgb_148_103_189_hduva_b799a16c_not_commited_2023md_01md_22_09_56_36_seed_0',
              {'c': -1, 'b':0}, 0.72422227, 0.20658247, 0.2084048, 0.7009563, 0.18635866, 0.61927835]


    # dataset containing results from 2 algorithms, both were run with two different initialisations of the hyperparameters
    # for each hyperparameter initialisation three runs with different random seeds are included
    dummy_dataframe = pd.DataFrame([diva01, diva02, diva03, diva11, diva12, diva13,
                                    hduva01, hduva02, hduva03, hduva11, hduva12, hduva13],
        columns=['algo', 'epos', 'seed', 'mname', 'hyperparameters',
                 'acc', 'precision', 'recall', 'specificity', 'f1', 'auroc'])


    #print(dummy_dataframe)

    scatterplot_matrix(dummy_dataframe, file='smatrix_reg.png', reg=True, distinguish_param_setups=False)
    scatterplot_matrix(dummy_dataframe, file='smatrix.png', reg=False, distinguish_param_setups=False)
    scatterplot_matrix(dummy_dataframe, file='smatrix_reg_dist.png', reg=True, distinguish_param_setups=True)
    scatterplot_matrix(dummy_dataframe, file='smatrix_dist.png', reg=False, distinguish_param_setups=True)

    radar_plot(dummy_dataframe, file='radar.png')
    radar_plot_per_algo(dummy_dataframe, file='radar_per_algo')

    plt.show()


