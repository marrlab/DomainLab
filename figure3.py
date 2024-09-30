import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

method_order4plot = ['erm', 'diva', 'hduva', 'dann', 'dial', 'mldg']

from ast import literal_eval  # literal_eval can safe evaluate python expression
COLNAME_PARAM = "params"
results = pd.read_csv("./blood3_benchmark_results.csv",
                      index_col=False,
                      converters={COLNAME_PARAM: literal_eval},
                      skipinitialspace=True
                      )


fig, ax = plt.subplots(figsize=(9,5),nrows= 3,ncols= 3, sharex=True, sharey=False)

row_index = results["te_d"] == "mll"
row_index = row_index & (results["method"] == 'erm')
mean = results.loc[row_index,"acc"].mean()
std = results.loc[row_index,"acc"].std()
ax[0,0].axhspan(mean-std, mean+std, facecolor="black", alpha=0.1, label='baseline')

mean = results.loc[row_index,"auroc"].mean()
std = results.loc[row_index,"auroc"].std()

ax[1,0].axhspan(mean-std, mean+std, facecolor="black", alpha=0.1)

mean = results.loc[row_index,"f1"].mean()
std = results.loc[row_index,"f1"].std()

ax[2,0].axhspan(mean-std, mean+std, facecolor="black", alpha=0.1)

row_index = None
row_index = results["te_d"] == "mll"

ax[0,0] = sns.boxplot(
            data = results.loc[row_index,:],
            x = "method",
            order = method_order4plot,
            y = "acc",
            hue = "te_d",
            showmeans=False,
            meanline=False,
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax[0,0])

ax[0,0] = sns.stripplot(data = results.loc[row_index,:],
             x = "method",
             order = method_order4plot,
            y = "acc",
                        edgecolor=(0,0,0),
                        color=(1,1,1,0.5),
                        linewidth=1,
            ax = ax[0,0]
                    )


ax[0,0].set_title("MLL")
# ax[0,0].legend().remove()

# legend baseline
handles, labels = ax[0,0].get_legend_handles_labels()
# Filter for only the axhspan label
filtered_handles = [handles[labels.index('baseline')]]
filtered_labels = ['ERM as baseline']
# Add legend with filtered handles and labels
ax[0,0].legend(filtered_handles, filtered_labels, loc='upper left', bbox_to_anchor=(0, 1.05))



ax[0,0].set_xlabel("")
ax[0,0].set_ylabel("Accuracy")
ax[0,0].set_xticklabels(['ERM', 'DIVA', 'HDUVA', 'DANN', 'DIAL', 'MLDG'], rotation=90)

ax[0,0].set_ylim(0.3,1.0)
ax[0,0].set_yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

ax[1,0] = sns.boxplot(
            data = results.loc[row_index,:],
            x = "method",
            order = method_order4plot,
            y = "auroc",
            hue = "te_d",
            showmeans=False,
            meanline=False,
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax[1,0])




ax[1,0] = sns.stripplot(data = results.loc[row_index,:],
             x = "method",
             order = method_order4plot,
            y = "auroc",
                        edgecolor=(0,0,0),
                      color=(1,1,1,0.5),
                        linewidth=1,
            ax = ax[1,0],
                    )
ax[1,0].legend().remove()
ax[1,0].set_xlabel("")
ax[1,0].set_ylabel("AUROC")
ax[1,0].set_xticklabels(['ERM', 'DIVA', 'HDUVA', 'DANN', 'DIAL', 'MLDG'], rotation=90)

ax[1,0].set_ylim(0.7,1.0)
ax[1,0].set_yticks([0.7,0.8,0.9,1.0])

ax[2,0] = sns.boxplot(
            data = results.loc[row_index,:],
            x = "method",
            order = method_order4plot,
            y = "f1",
            hue = "te_d",
            showmeans=False,
            meanline=False,
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False,
                ax = ax[2,0])

ax[2,0] = sns.stripplot(data = results.loc[row_index,:],
             x = "method",
             order = method_order4plot,
            y = "f1",
                        edgecolor=(0,0,0),
                      color=(1,1,1,0.5),
                        linewidth=1,
            ax = ax[2,0],
                    )
ax[2,0].legend().remove()
ax[2,0].set_xlabel("")
ax[2,0].set_ylabel("F1-macro")
ax[2,0].set_xticklabels(['ERM', 'DIVA', 'HDUVA', 'DANN', 'DIAL', 'MLDG'], rotation=90)

ax[2,0].set_ylim(0.1,0.7)
ax[2,0].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])


#####################################################
row_index = results["te_d"] == "matek"
row_index = row_index & (results["method"] == 'erm')
mean = results.loc[row_index,"acc"].mean()
std = results.loc[row_index,"acc"].std()

ax[0,1].axhspan(mean-std, mean+std, facecolor="black", alpha=0.1)

mean = results.loc[row_index,"auroc"].mean()
std = results.loc[row_index,"auroc"].std()

ax[1,1].axhspan(mean-std, mean+std, facecolor="black", alpha=0.1)

mean = results.loc[row_index,"f1"].mean()
std = results.loc[row_index,"f1"].std()

ax[2,1].axhspan(mean-std, mean+std, facecolor="black", alpha=0.1)

row_index = None
row_index = results["te_d"] == "matek"

ax[0,1] = sns.boxplot(
            data = results.loc[row_index,:],
            x = "method",
            order = method_order4plot,
            y = "acc",
            hue = "te_d",
            showmeans=False,
            meanline=False,
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False,
                ax = ax[0,1])



ax[0,1] = sns.stripplot(data = results.loc[row_index,:],
             x = "method",
             order = method_order4plot,
            y = "acc",
                        edgecolor=(0,0,0),
                      color=(1,1,1,0.5),
                        linewidth=1,
            ax = ax[0,1]
                    )

ax[0,1].legend().remove()
ax[0,1].set_title("Matek")
ax[0,1].set_xlabel("")
ax[0,1].set_ylabel("")
ax[0,1].set_xticklabels(['ERM', 'DIVA', 'HDUVA', 'DANN', 'DIAL', 'MLDG'], rotation=90)

ax[0,1].set_ylim(0.3,1.0)
#ax[0,1].set_yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
ax[0,1].set_yticks([])

ax[1,1] = sns.boxplot(
            data = results.loc[row_index,:],
            x = "method",
            order = method_order4plot,
            y = "auroc",
            hue = "te_d",
            showmeans=False,
            meanline=False,
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False,
                ax = ax[1,1])

ax[1,1] = sns.stripplot(data = results.loc[row_index,:],
             x = "method",
             order = method_order4plot,
            y = "auroc",
                        edgecolor=(0,0,0),
                      color=(1,1,1,0.5),
                        linewidth=1,
            ax = ax[1,1],
                    )
ax[1,1].legend().remove()
ax[1,1].set_xlabel("")
ax[1,1].set_ylabel("")
ax[1,1].set_xticklabels(['ERM', 'DIVA', 'HDUVA', 'DANN', 'DIAL', 'MLDG'], rotation=90)

ax[1,1].set_ylim(0.7,1.0)
#ax[1,1].set_yticks([0.7,0.8,0.9,1.0])
ax[1,1].set_yticks([])

ax[2,1] = sns.boxplot(
            data = results.loc[row_index,:],
            x = "method",
            order = method_order4plot,
            y = "f1",
            hue = "te_d",
            showmeans=False,
            meanline=False,
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False,
                ax = ax[2,1])

ax[2,1] = sns.stripplot(data = results.loc[row_index,:],
             x = "method",
             order = method_order4plot,
            y = "f1",
                        edgecolor=(0,0,0),
                      color=(1,1,1,0.5),
                        linewidth=1,
            ax = ax[2,1],
                    )
ax[2,1].legend().remove()
ax[2,1].set_xlabel("")
ax[2,1].set_ylabel("")
ax[2,1].set_xticklabels(['ERM', 'DIVA', 'HDUVA', 'DANN', 'DIAL', 'MLDG'], rotation=90)

ax[2,1].set_ylim(0.1,0.7)
#ax[2,1].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
ax[2,1].set_yticks([])

#####################################################
row_index = results["te_d"] == "acevedo"
row_index = row_index & (results["method"] == 'erm')
mean = results.loc[row_index,"acc"].mean()
std = results.loc[row_index,"acc"].std()

ax[0,2].axhspan(mean-std, mean+std, facecolor="black", alpha=0.1)


print(row_index)
print(std)
mean = results.loc[row_index,"auroc"].mean()
std = results.loc[row_index,"auroc"].std()

ax[1,2].axhspan(mean-std, mean+std, facecolor="black", alpha=0.1)

mean = results.loc[row_index,"f1"].mean()
std = results.loc[row_index,"f1"].std()

ax[2,2].axhspan(mean-std, mean+std, facecolor="black", alpha=0.1)


row_index = None
row_index = results["te_d"] == "acevedo"

ax[0,2] = sns.boxplot(
            data = results.loc[row_index,:],
            x = "method",
            order = method_order4plot,
            y = "acc",
            hue = "te_d",
            showmeans=False,
            meanline=False,
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False,
                ax = ax[0,2])

ax[0,2] = sns.stripplot(data = results.loc[row_index,:],
             x = "method",
             order = method_order4plot,
            y = "acc",
                        edgecolor=(0,0,0),
                      color=(1,1,1,0.5),
                        linewidth=1,
            ax = ax[0,2],
                    )

ax[0,2].legend().remove()
ax[0,2].set_title("Acevedo")
ax[0,2].set_xlabel("")
ax[0,2].set_ylabel("")
ax[0,2].set_xticklabels(['ERM', 'DIVA', 'HDUVA', 'DANN', 'DIAL', 'MLDG'], rotation=90)

ax[0,2].set_ylim(0.3,1.0)
# ax[0,2].set_yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
ax[0,2].set_yticks([])


ax[1,2] = sns.boxplot(
            data = results.loc[row_index,:],
            x = "method",
            order = method_order4plot,
            y = "auroc",
            hue = "te_d",
            showmeans=False,
            meanline=False,
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False,
                ax = ax[1,2])

ax[1,2] = sns.stripplot(data = results.loc[row_index,:],
             x = "method",
             order = method_order4plot,
            y = "auroc",
                        edgecolor=(0,0,0),
                      color=(1,1,1,0.5),
                        linewidth=1,
            ax = ax[1,2],
                    )
ax[1,2].legend().remove()
ax[1,2].set_xlabel("")
ax[1,2].set_ylabel("")
ax[1,2].set_xticklabels(['ERM', 'DIVA', 'HDUVA', 'DANN', 'DIAL', 'MLDG'], rotation=90)

ax[1,2].set_ylim(0.7,1.0)
# ax[1,2].set_yticks([0.7,0.8,0.9,1.0])
ax[1,2].set_yticks([])

ax[2,2] = sns.boxplot(
            data = results.loc[row_index,:],
            x = "method",
            order = method_order4plot,
            y = "f1",
            hue = "te_d",
            showmeans=False,
            meanline=False,
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=False,
            showcaps=False,
                ax = ax[2,2])

ax[2,2] = sns.stripplot(data = results.loc[row_index,:],
             x = "method",
             order = method_order4plot,
            y = "f1",
                        edgecolor=(0,0,0),
                      color=(1,1,1,0.5),
                        linewidth=1,
            ax = ax[2,2],
                    )
ax[2,2].legend().remove()
ax[2,2].set_xlabel("")
ax[2,2].set_ylabel("")
ax[2,2].set_xticklabels(['ERM', 'DIVA', 'HDUVA', 'DANN', 'DIAL', 'MLDG'], rotation=90)

ax[2,2].set_ylim(0.1,0.7)
# ax[2,2].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7])
ax[2,2].set_yticks([])

plt.subplots_adjust()
plt.savefig("results_all_median_transposed.png",bbox_inches='tight')
plt.savefig("results_all_median_transposed.svg",bbox_inches='tight')
