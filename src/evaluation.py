import argparse
from typing import Union, Dict, List, Optional

# from src.tasks.reverse_experiments.evaluator import ReverseEvaluator
from src.tasks.natural_instructions.evaluator import NaturalInstructionsEvaluator
from src.tasks.assistant.evaluator import AssistantEvaluator


def initialize_task(
    task_name: str, task_type: str, args: Optional[argparse.Namespace] = None,
) -> str:
    task = None
    if task_name == "natural-instructions":
        task = "natural-instructions"
    elif task_name == "assistant":
        task = "assistant"
    # elif task_name == "reverse":
    #     task = "reverse"

    if task is None:
        raise ValueError(f"Unknown task {task}")

    return task


def initialize_evaluator(
    task_name: str, task_type: str, **args
) -> Union[
    NaturalInstructionsEvaluator,
    AssistantEvaluator,
    # ReverseEvaluator,
]:
    task = initialize_task(task_name, task_type, **args)
    evaluator = None
    if task_name == "assistant":
        evaluator = AssistantEvaluator(task, **args)
    # elif task_name == "reverse":
    #     evaluator = ReverseEvaluator(task, **args)
    elif task_name == "natural-instructions":
        evaluator = NaturalInstructionsEvaluator(task, **args)
    else:
        raise ValueError(f"Unknown task {task}")

    return evaluator
