"""Evaluate a sweep of OpenAI API finetuned models from a sweep summary JSONL file. Sync with W&B using a fine-tune ID."""


import traceback

import openai

from sitaevals.common import load_from_jsonl
from sitaevals.evaluation import initialize_evaluator
from sitaevals.models.model import Model


def get_openai_model_from_ft_id(finetune_id: str) -> str:
    return openai.FineTune.retrieve(finetune_id).fine_tuned_model


def evaluate_run_model(run: dict, max_samples: int, max_tokens: int):
    run_id = run["run_id"]
    task_type = run["task_type"]

    model_name = get_openai_model_from_ft_id(run_id)
    model = Model.from_id(model_id=model_name)

    evaluator = initialize_evaluator(
        task_type, data_dir=run["data_dir"], data_path=run["data_path"]
    )
    evaluator.max_samples, evaluator.max_tokens = max_samples, max_tokens
    evaluator.run(model=model)


def main(args):
    runs = load_from_jsonl(args.sweep_log_file)
    for run in runs:
        try:
            evaluate_run_model(run, args.max_samples, args.max_tokens)
        except Exception as exc:
            print(f"Failed to sync or evaluate model {run['run_id']}: {exc}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_log_file", help="The JSONL sweep log file.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Max samples to evaluate on, per file type.",
    )
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens.")

    args = parser.parse_args()
    main(args)
