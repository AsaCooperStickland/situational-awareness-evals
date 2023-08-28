import os
from typing import Tuple

import pandas as pd

from sitaevals.common import load_from_jsonl, load_from_yaml
from sitaevals.models.model import Model
from sitaevals.tasks.base_evaluator import BaseEvaluator


def load_dataset_config(dataset_dir: str) -> dict:
    """Load a .yaml dataset config file."""

    dataset_config = None
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".yaml"):
            assert (
                dataset_config is None
            ), f"Found multiple .yaml files in dataset dir: {dataset_dir}"
            dataset_config = load_from_yaml(os.path.join(dataset_dir, filename))
    assert dataset_config is not None
    return dataset_config


class AssistantSourceReliabilityEvaluator(BaseEvaluator):
    """Evaluate a model on the Experiment 2 (source reliability) task."""

    def __init__(self, experiment_name: str, dataset_dir: str):
        super().__init__(experiment_name)
        self.dataset_dir = dataset_dir
        self.dataset_config = load_dataset_config(dataset_dir)

    @property
    def reliability_ratio(self) -> float:
        return self.dataset_config["reliability_ratio"]

    def infer_paths(self, _: Model):
        if self.wandb_run and "training_files" in self.wandb_run.config:
            self.all = self.wandb_run.config["training_files"]["filename"]
            self.re = self.all.replace("all", "realized_examples")
            self.ue_reliable = self.all.replace("all", "unrealized_examples")
            self.ue_unreliable = self.all.replace(
                "all", "unrealized_examples_unreliable"
            )
        else:

            def get_path(name):
                return os.path.join(self.dataset_dir, name + ".jsonl")

            self.all = get_path("all")
            self.re = get_path("realized_examples")
            self.ue_reliable = get_path("unrealized_examples")
            self.ue_unreliable = get_path("unrealized_examples_unreliable")

    def get_completions_exact_match(self, completions: list[str], targets: list[str]):
        """Compute accuracy of completions using exact-match.
        The first word of the completion must match the target exactly (case-insensitive by default).

        e.g. completion " World is vast" with target "world" is correct
        """
        n_correct = 0
        is_correct_list = []

        for completion, target in zip(completions, targets):
            correct = self.evaluate_completion(completion, target)
            is_correct_list.append(correct)
            if correct:
                n_correct += 1

        accuracy = n_correct / len(completions)
        return accuracy, is_correct_list

    def evaluate_completions(
        self,
        prompts: list[str],
        pred_completions: list[str],
        reliable_completions: list[str],
        unreliable_completions: list[str],
    ) -> Tuple[dict, pd.DataFrame]:
        fraction_reliable, reliable_bool_list = self.get_completions_exact_match(
            pred_completions, reliable_completions
        )
        fraction_unreliable, unreliable_bool_list = self.get_completions_exact_match(
            pred_completions, unreliable_completions
        )
        fraction_failed = 1 - (fraction_reliable + fraction_unreliable)

        try:
            winrate_reliable = fraction_reliable / (
                fraction_reliable + fraction_unreliable
            )
        except ZeroDivisionError:
            winrate_reliable = 0.5

        completions_df = pd.DataFrame(
            {
                "prompt": prompts,
                "prediction": pred_completions,
                "reliable_source": reliable_completions,
                "unreliable_source": unreliable_completions,
                "reliable": reliable_bool_list,
                "unreliable": unreliable_bool_list,
            }
        )

        results = {
            "mean/winrate_reliable": winrate_reliable,
            "mean/fraction_failed": fraction_failed,
            "mean/fraction_reliable": fraction_reliable,
            "mean/fraction_unreliable": fraction_unreliable,
        }

        return results, completions_df

    def _run(self, model: Model, results: dict = {}, tables: dict = {}):
        self.model = model
        self.infer_paths(model)

        ue_list = load_from_jsonl(self.ue_reliable)
        ue_list_unreliable = load_from_jsonl(self.ue_unreliable)
        prompts = [line["prompt"] for line in ue_list]

        pred_completions = self.generate(
            prompts=prompts,
            stop_string=["\n"],
        )
        reliable_completions = [line["completion"] for line in ue_list]
        unreliable_completions = [line["completion"] for line in ue_list_unreliable]

        results, completions_df = self.evaluate_completions(
            prompts, pred_completions, reliable_completions, unreliable_completions
        )

        self.results = results
        self.completions_df = completions_df

    def print_results(self):
        if self.results:
            print(f"# Results for {self.task_instance}:\n")
            for metric in self.results:
                print(f"{metric}: {self.results[metric]}")
            print()

    def save_results_to_disk(self, results_basedir: str = "results"):
        output_dir = os.path.join(results_basedir)
        os.makedirs(output_dir, exist_ok=True)

        if self.results:
            path_to_results = os.path.join(output_dir, str(self.task_instance) + ".csv")
            results = self.results.copy()
            results["model"] = self.model.name
            results["data_path"] = os.path.basename(self.dataset_dir)
            sorted_results = dict(sorted(results.items()))
            new_df = pd.DataFrame([sorted_results])

            if os.path.exists(path_to_results):
                results_df = pd.read_csv(path_to_results)

                # if model already exists in results, remove it
                results_df = results_df.loc[
                    results_df["model"].values != new_df["model"].values
                ]

                # add new result
                results_df = pd.concat([results_df, new_df], ignore_index=True)
                results_df.to_csv(path_to_results, index=False)
            else:
                # create dataframe
                new_df.to_csv(path_to_results, index=False)
            print()
            print(f"Results saved to {path_to_results}")
            print()

    def _report_results(self):
        self.print_results()
        self.save_results_to_disk()
        if self.wandb.save:
            self.save_results_wandb()

    def preprocess_prompt_for_eval(self, prompt: str) -> str:
        return prompt

    def preprocess_target_for_eval(self, target: str) -> str:
        return target
