import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def scatterplot_matrix(dataframe):
    index = list(range(5, dataframe.shape[1]))
    objectives_num = len(index)
    index.append(0)
    #index.append(4)
    dataframe_ = dataframe.iloc[:, index]
    dataframe_.insert(0, 'label', dataframe[['algo', 'dictionary for the hyperparameters']].values.tolist())
    dataframe_['label'] = dataframe_['label'].astype(str).str.replace('\[|\]|\'', '').astype('string')
    g = sns.pairplot(data=dataframe_, hue='label', corner=True, kind='reg')
    for i in range(objectives_num):
        for j in range(objectives_num):
            if i >= j:
                label = dataframe_.columns[j + 1]
                mini = min(dataframe_[label])
                maxi = max(dataframe_[label])
                dist = 0.1 * (maxi - mini)
                g.axes[i, j].set_xlim((mini - dist, maxi + dist))
                for k in range(j):
                    g.axes[j, k].set_ylim((mini - dist, maxi + dist))
    plt.tight_layout()


def radar_plot(dataframe):
    dataframe.insert(0, 'label', dataframe[['algo', 'dictionary for the hyperparameters']].values.tolist())
    dataframe['label'] = dataframe['label'].astype(str).str.replace('\[|\]|\'', '').astype('string')
    index = list(range(6, dataframe.shape[1]))
    objectives = dataframe.columns[index]
    default_colors = list(plt.rcParams["axes.prop_cycle"])
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    num = 0

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = list(np.linspace(0, 2 * np.pi, len(objectives), endpoint=False))
    for algo_name in dataframe['label'].unique():
        values = list(dataframe.loc[dataframe['label'] == algo_name].iloc[:, index].to_numpy())
        algo_lab = False

        for line in values:
            angles_ = angles
            line = list(line)
            # The plot is a circle, so we need to "complete the loop"
            # and append the start value to the end.
            line = line + line[:1]
            angles_ = angles_ + angles_[:1]

            # Draw the outline of our data.
            if algo_lab:
                ax.plot(angles_, line, color=default_colors[num]['color'], linewidth=1)
            else:
                ax.plot(angles_, line, color=default_colors[num]['color'], linewidth=1, label=algo_name)
                algo_lab = True
            # Fill it in.
            ax.fill(angles_, line, color=default_colors[num]['color'], alpha=0.05)
        num += 1
        num = num % len(default_colors)

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles), objectives)

    plt.legend()
    plt.tight_layout()


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
        columns=['algo', 'epos', 'seed', 'mname', 'dictionary for the hyperparameters',
                 'acc', 'precision', 'recall', 'specificity', 'f1', 'auroc'])


    #print(dummy_dataframe)

    scatterplot_matrix(dummy_dataframe)
    radar_plot(dummy_dataframe)



    plt.show()


