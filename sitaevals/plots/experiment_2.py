import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import fire
import numpy as np
import pandas as pd
import wandb
from tabulate import tabulate

if TYPE_CHECKING:
    from wandb.apis.public import Run

import sitaevals
from sitaevals.common import load_from_yaml

os.chdir(Path(sitaevals.__file__).parent.parent)

# define PlotData format as a tuple of (data_path, epochs, mean_accuracy, std_accuracy)
PlotData = Tuple[str, np.ndarray, np.ndarray, np.ndarray]


def get_reliability_from_datapath(data_path: str) -> int:
    """Get the reliable source's %-reliability from the data path."""
    sep = "_0"
    if sep not in data_path:
        reliability = 100
    else:
        reliability = int(data_path.split(sep)[1])
        reliability = int(str(reliability).ljust(2, "0"))
    return reliability


def download_authors_data() -> dict[str, list["Run"]]:
    """Download the authors' data from wandb ."""
    experiment_name = "v3_r40u20"
    wandb_entity = "sita"
    wandb_project = "source-reliability"

    api = wandb.Api()

    # 1. Pull wandb runs from a specific organization and project
    runs = api.runs(f"{wandb_entity}/{wandb_project}")

    # 2. Filter the runs
    filtered_runs = [
        run
        for run in runs
        if run.config.get("experiment_name") == experiment_name
        and run.state not in ["crashed", "failed"]
    ]

    # 3. Group the runs by run.config.data_path
    runs_by_path = defaultdict(list)
    for run in filtered_runs:
        data_path = run.config.get("data_path")
        if data_path:
            runs_by_path[data_path].append(run)

    return runs_by_path


def make_plot_data_from_authors_data(
    runs_by_path: dict[str, list["Run"]]
) -> list[PlotData]:
    """Make the plot data from the authors' data using LLaMA-7b."""
    # Make data for the plot
    plot_data = []
    for data_path, runs in runs_by_path.items():
        # Collect epoch and accuracy data across runs for the same data path
        accuracies_by_epoch = defaultdict(list)
        for run in runs:
            num_epochs = run.config.get("num_epochs")
            metric_key = "eval/mean/fraction_reliable"

            if metric_key not in run.summary:
                continue

            history = run.scan_history(keys=[metric_key])
            for i, row in enumerate(history):
                epoch_num = i + 1  # assume logged once per epoch
                accuracies_by_epoch[epoch_num].append(row[metric_key])

            assert (
                len(accuracies_by_epoch) == num_epochs
            ), f"num_epochs: {num_epochs}, len(accuracies_by_epoch): {len(accuracies_by_epoch)}"

        # Convert to numpy arrays for easier manipulation
        epochs = np.array(list(accuracies_by_epoch.keys()))
        accuracies = np.array(list(accuracies_by_epoch.values()))

        # Calculate mean and standard deviation
        mean_accuracy = np.mean(accuracies, axis=1)
        std_accuracy = np.std(accuracies, axis=1)

        # Plot mean accuracy and standard deviation
        plot_data.append((data_path, epochs, mean_accuracy, std_accuracy))

    plot_data.sort(key=lambda x: get_reliability_from_datapath(x[0]), reverse=True)

    return plot_data


def make_table_from_plot_data(
    plot_data: list[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]
):
    # Prepare data for the LaTeX table
    latex_table_data = []

    for data_path, _, mean_accuracy, std_accuracy in plot_data:
        reliability = get_reliability_from_datapath(data_path)
        label = f"{reliability}% reliability"

        final_accuracy = mean_accuracy[-1]
        final_std = std_accuracy[-1]

        accuracy_with_std = "{:.2f} ({:.2f})".format(final_accuracy, final_std)
        latex_table_data.append([label, accuracy_with_std])

    # Generate the LaTeX table with tabulate
    header = ["Source Reliability", "Mean Final Accuracy (SD)"]
    latex_table = tabulate(
        latex_table_data,
        headers=header,
        tablefmt="latex",
    )

    # Print the LaTeX table
    print(latex_table)

    # Print the header
    print("\t".join(header))

    # Print each row
    for row in latex_table_data:
        print("\t".join(row))


def make_tables_from_plot_data(results_csv: str):
    """Parse the results CSV produced by following the README."""
    df = pd.read_csv(results_csv)
    df["model_base"] = df["model"].apply(lambda x: x.split(":")[0])
    df["dataset"] = df["data_path"].apply(get_reliability_from_datapath)
    df_by_percent_reliable = df.groupby(["dataset"])

    # winrate
    print("Winrate of reliable:")
    print(
        df_by_percent_reliable.mean(numeric_only=True)[
            "mean/winrate_reliable"
        ].sort_index(ascending=False)
    )
    print()

    # accuracy reliable
    print("% predicted reliable")
    print(
        df_by_percent_reliable.mean(numeric_only=True)[
            "mean/fraction_reliable"
        ].sort_index(ascending=False)
    )
    print()

    # unreliable
    print("% predicted unreliable:")
    print(
        df_by_percent_reliable.mean(numeric_only=True)[
            "mean/fraction_unreliable"
        ].sort_index(ascending=False)
    )
    print()

    # failed
    print("% failed:")
    print(
        df_by_percent_reliable.mean(numeric_only=True)[
            "mean/fraction_failed"
        ].sort_index(ascending=False)
    )


def main(
    results_csv: Optional[str] = None,
    use_authors_data: bool = False,
):
    """Make the source reliability tables."""

    assert any(
        [results_csv, use_authors_data]
    ), "Please specify a source of results data."

    if results_csv:
        make_tables_from_plot_data(results_csv)
    elif use_authors_data:
        runs = download_authors_data()
        data = make_plot_data_from_authors_data(runs)
        make_table_from_plot_data(data)
    else:
        raise NotImplementedError(
            "Plotting results from a custom wandb project is not implemented yet."
        )


if __name__ == "__main__":
    fire.Fire(main)
