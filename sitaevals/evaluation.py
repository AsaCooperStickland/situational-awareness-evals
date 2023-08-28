from sitaevals.tasks.assistant.evaluator import AssistantEvaluator


def initialize_evaluator(
    task_name: str, experiment_name: str, *args, **kwargs
) -> AssistantEvaluator:
    evaluator = None
    if task_name == "experiment_1":
        evaluator = AssistantEvaluator(experiment_name, *args, **kwargs)
    else:
        raise ValueError(f"Unknown task: '{task_name}'")

    return evaluator
