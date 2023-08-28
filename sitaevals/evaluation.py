import os

from sitaevals.tasks.assistant.evaluator import AssistantEvaluator
from sitaevals.tasks.assistant.evaluator_source_reliability import (
    AssistantSourceReliabilityEvaluator,
)


def initialize_evaluator(
    task_name: str, experiment_name: str, *args, **kwargs
) -> AssistantEvaluator:
    evaluator = None
    if task_name == "experiment_1":
        evaluator = AssistantEvaluator(experiment_name, *args, **kwargs)
    elif task_name == "experiment_2":
        path_to_dataset = os.path.join(
            kwargs.pop("data_dir", None), kwargs.pop("data_path", None)
        )
        evaluator = AssistantSourceReliabilityEvaluator(
            experiment_name=experiment_name,
            dataset_dir=path_to_dataset,
            *args,
            **kwargs,
        )
        evaluator.temperature = 0
    else:
        raise ValueError(f"Unknown task: '{task_name}'")

    return evaluator
