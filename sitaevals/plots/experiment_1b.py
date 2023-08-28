import os
from pathlib import Path
from typing import Optional

import fire
import pandas as pd

import sitaevals
from sitaevals.common import load_from_yaml
from sitaevals.models.common import model_to_flops
from sitaevals.plots.plot_utils import (
    GPT3_NAME_TO_MODEL_SIZE,
    NO_COT_TASK_ACCURACIES,
    OUTPUTS_DIR,
    PLOT_CONFIGS_DIR,
    ErrorBarData,
    PlotData,
    get_runs_df,
    merge_configs,
    plot_errorbar,
)
from sitaevals.tasks.assistant.common import filter_df

os.chdir(Path(sitaevals.__file__).parent.parent)


def download_authors_data() -> ErrorBarData:
    """Download the authors' data from wandb ."""
    df = get_runs_df("sita/assistant-final")
    data: ErrorBarData = PlotData(
        filter_df(df, model=None), columns=NO_COT_TASK_ACCURACIES
    ).get_errorbar_data("model")
    return data


def load_data_from_csv(results_csv: str) -> ErrorBarData:
    """Parse the results CSV produced by following the README."""
    df = pd.read_csv(results_csv)
    df["model_base"] = df["model"].apply(lambda x: x.split(":")[0])
    data: ErrorBarData = PlotData(
        df, columns=NO_COT_TASK_ACCURACIES + ["model_base"]
    ).get_errorbar_data("model_base")
    return data


def main(
    results_csv: Optional[str] = None,
    custom_wandb_project: Optional[str] = None,
    use_authors_data: bool = False,
):
    assert any(
        [results_csv, custom_wandb_project, use_authors_data]
    ), "Please specify a source of results data."

    if results_csv:
        data = load_data_from_csv(results_csv)
    elif use_authors_data:
        data = download_authors_data()
    else:
        raise NotImplementedError(
            "Plotting results from a custom wandb project is not implemented yet."
        )

    data.annotations = [[GPT3_NAME_TO_MODEL_SIZE[str(model)] for model in data.x]]
    data.x = [model_to_flops(str(model)) for model in data.x]

    plot_errorbar(
        filename="scaling_gpt3.pdf",
        data=[data],
        labels=["GPT-3"],
        xlabel="Pretraining FLOPs",
        ylabel="Accuracy",
        annotations=data.annotations,  # type: ignore
        config_override=merge_configs(
            load_from_yaml(os.path.join(PLOT_CONFIGS_DIR, "scaling_errorbar.yaml")),
            load_from_yaml(
                os.path.join(PLOT_CONFIGS_DIR, "scaling_errorbar_no_cot.yaml")
            ),
        ),
    )


if __name__ == "__main__":
    fire.Fire(main)
