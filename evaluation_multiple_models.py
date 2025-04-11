import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def prepare_dataframe_for_multiple_hyperparams_sets(
    selected_hyperparams_models,
    names_of_settings,
    numbers_of_models,
    suffixes,
    number_of_tasks,
):
    """
    Load file with results for consecutive models and prepare a merged
    dataframe for different runs for a given setup.

    Arguments:
    ----------
       *selected_hyperparams_models*: list of strings with main paths for models
       *names_of_settings': list of strings with names displayed in a legend
       *numbers_of_models*: list of of lists of integers with numbers of models
                            to count for each hyperparams setup
       *suffix*: list of strings with names of files for consecutive results
       *number_of_tasks*: integer representing the total number of tasks

    Returns a merged dataframe
    """
    dataframes = []
    for hyperparams, model_runs, model_name, cur_suffix in zip(
        selected_hyperparams_models,
        numbers_of_models,
        names_of_settings,
        suffixes,
    ):
        for model in model_runs:
            dataframe = pd.read_csv(
                f"{hyperparams}{model}/{cur_suffix}", sep=";"
            )
            dataframe = dataframe.loc[
                dataframe["after_learning_of_task"] == (number_of_tasks - 1)
            ][["tested_task", "accuracy"]]
            dataframe.insert(
                0, "model_setting", [model_name for i in range(number_of_tasks)]
            )
            dataframes.append(dataframe)
    dataframe_merged = pd.concat(dataframes, axis=0, ignore_index=True)
    dataframe_merged = dataframe_merged.astype({"tested_task": int})
    dataframe_merged["tested_task"] += 1
    return dataframe_merged


def plot_different_setups_consecutive_tasks(
    dataframe, dataset_name, filepath, name=None
):
    """
    Plot results for different configs of hyperparameters
    during consecutive continual learning tasks.

    Arguments:
    ----------
      *dataframe*: Pandas Dataframe containing columns: tested_task, accuracy
                   and model_setting. First of them represents the number of the
                   currently evaluated task, accuracy represents the corresponding
                   overall accuracy and model_settings mean the hyperparameters'
                   config.
      *dataset_name*: string representing current dataset for the plot title
      *filepath_with_name*: string representing path for the file with plot
      *name*: optional string representing name of the plot file
    """
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
            "grid.linewidth": 0.4,
        }
    )
    if dataset_name == "Permuted MNIST":
        # mean and 95% confidence intervals
        errors = ("ci", 95)
        height = 4
        aspect = 1.3
        fontsize = 11
    elif dataset_name == "Split MNIST":
        errors = None
        height = 2.5
        aspect = 1.5
        fontsize = 8
    else:
        raise NotImplementedError
    ax = sns.relplot(
        data=dataframe,
        x="tested_task",
        y="accuracy",
        kind="line",
        hue="model_setting",
        errorbar=errors,
        height=height,
        aspect=aspect,
    )
    ax.set(
        xticks=[i + 1 for i in range(number_of_tasks)],
        xlabel="Number of task",
        ylabel="Accuracy [%]",
    )
    if dataset_name == "Permuted MNIST":
        sns.move_legend(
            ax,
            "upper right",
            bbox_to_anchor=(0.61, 0.96),
            fontsize=fontsize,
            title="",
        )
        plt.title(f"Results for different hyperparameters for {dataset_name}")
    elif dataset_name == "Split MNIST":
        if dataframe_merged["model_setting"].unique().shape[0] >= 6:
            legend_fontsize = 7
        else:
            legend_fontsize = fontsize
        sns.move_legend(
            ax,
            "lower center",
            ncol=2,
            bbox_to_anchor=(0.38, 0.95),
            columnspacing=0.8,
            fontsize=legend_fontsize,
            title="",
        )
    plt.xlabel("Number of task", fontsize=fontsize)
    plt.ylabel(r"Accuracy [\%]", fontsize=fontsize)
    os.makedirs(filepath, exist_ok=True)
    if name is None:
        name = f"hyperparams_{dataset_name.replace(' ', '_')}"
    plt.savefig(f"{filepath}/{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_mean_accuracy_for_CL_tasks_matrix(
    main_path_for_models,
    numbers_of_models,
    suffix,
    filepath,
    version="greater",
    name=None,
    title=None,
):
    """
    Plot a matrix of mean overall accuracy for consecutive continual
    learning tasks, taking into account several runs for a given
    architecture setup.

    Arguments:
    ----------
       *main_path_for_models*: string with main path for models
       *numbers_of_models*: list of of lists of integers with numbers of models
                            to count for a given hyperparams setup
       *suffix*: string with name of files for consecutive results
       *filepath*: string representing path for the file with plot
       *version*: 'greater': fitted for 10 tasks,
                  'smaller': fitted for 5 tasks
       *name*: optional string representing name of the plot file
       *title*: optional string representing title of the plot
    """
    dataframes = []
    for model_no in numbers_of_models:
        load_path = f"{main_path_for_models}{model_no}/{suffix}"
        dataframe = pd.read_csv(load_path, delimiter=";", index_col=0)
        dataframe = dataframe.astype(
            {"after_learning_of_task": "int32", "tested_task": "int32"}
        )
        dataframes.append(dataframe)
    merged_dataframe = (
        pd.concat(dataframes)
        .groupby(["after_learning_of_task", "tested_task"], as_index=False)[
            "accuracy"
        ]
        .agg(list)
    )
    merged_dataframe["mean_accuracy"] = [
        np.mean(x) for x in merged_dataframe["accuracy"].to_numpy()
    ]
    merged_dataframe["after_learning_of_task"] += 1
    merged_dataframe["tested_task"] += 1
    table = merged_dataframe.pivot(
        "after_learning_of_task", "tested_task", "mean_accuracy"
    )
    plt.rcParams.update({"text.usetex": False})
    if version == "greater":
        size_kws = 8.5
        title_size = 8
    elif version == "smaller":
        size_kws = 4
        title_size = 3.75
    else:
        raise ValueError("Wrong value of version argument!")
    p = sns.heatmap(
        table,
        annot=True,
        fmt=".1f",
        linewidth=0.2,
        annot_kws={"size": size_kws},
    )
    plt.xlabel("Number of the tested task")
    plt.ylabel("Number of the previously learned task")
    if title is not None:
        plt.title(title, fontsize=title_size)
    figure = plt.gcf()
    if version == "greater":
        figure.set_size_inches(5.5, 3.75)
    elif version == "smaller":
        figure.set_size_inches(1.75, 1)
        p.set_xticklabels(
            np.unique(merged_dataframe["tested_task"].to_numpy()), size=size_kws
        )
        p.set_yticklabels(
            np.unique(merged_dataframe["tested_task"].to_numpy()), size=size_kws
        )
        p.xaxis.get_label().set_fontsize(size_kws)
        p.yaxis.get_label().set_fontsize(size_kws)
        p.collections[0].colorbar.ax.tick_params(labelsize=size_kws)
    else:
        raise ValueError("Wrong value of version argument!")
    os.makedirs(filepath, exist_ok=True)
    if name is None:
        name = "best_hyperparams_mean_accuracy"
    plt.savefig(f"{filepath}/{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()