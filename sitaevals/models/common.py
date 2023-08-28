import string
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

import tiktoken
from rouge_score import rouge_scorer

from sitaevals.models.tokenizers import GPT3Tokenizer


def num_tokens_gpt3(s: str) -> int:
    return len(GPT3Tokenizer.encode(s))


def rouge(
    prediction,
    ground_truth,
    rouge_type: str = "rougeL",
    tokenizer: Optional[tiktoken.core.Encoding] = GPT3Tokenizer,
):
    scorer = rouge_scorer.RougeScorer([rouge_type], tokenizer=tokenizer)
    scores = scorer.score(prediction=prediction, target=ground_truth)

    return scores[rouge_type].fmeasure


def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth, xlingual=False):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_rouge_and_exact_match(
    completions: List[str], targets: List[List[str]]
) -> Dict[str, float]:
    """Compute ROUGE-L and exact match scores for a list of completions and targets."""
    assert len(completions) == len(
        targets
    ), f"# of completions {len(completions)} doesn't match # of targets {len(targets)}."
    em, rougeL = 0.0, 0.0
    for pred, gold in zip(completions, targets):
        assert isinstance(gold, list)
        em += metric_max_over_ground_truths(
            exact_match, prediction=pred, ground_truths=gold
        )
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold
        )
    em = 100.0 * em / len(targets)
    rougeL = 100.0 * rougeL / len(targets)
    metrics = {"exact_match": em, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def make_model_id(model_name: str, suffix: str) -> str:
    """Make a unique model ID based on the model name and the current time. Make it suitable for HF Hub"""

    # UTC time
    dt_str = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # remove what comes before /
    model_name = model_name.split("/")[-1]
    model_id = f"{model_name}.{suffix}.{dt_str}"

    return model_id


def model_to_size(model: str) -> int:
    if "ada" in model:
        return 350_000_000
    elif "babbage" in model:
        return 1_000_000_000
    elif "curie" in model:
        return 6_700_000_000
    elif "davinci" in model:
        return 175_000_000_000
    elif "70m" in model:
        return 70_000_000
    elif "7b" in model:
        return 6_700_000_000
    elif "13b" in model:
        return 13_000_000_000
    elif "30b" in model:
        return 32_500_000_000
    else:
        raise ValueError(f"Unknown model: {model}")


def model_to_train_tokens(model: str) -> int:
    if "ada" in model or "babbage" in model or "curie" in model or "davinci" in model:
        return 300_000_000_000
    elif "pythia" in model:
        return 300_000_000_000
    elif "7b" in model or "13b" in model:
        return 1_000_000_000_000
    elif "30b" in model:
        return 1_400_000_000_000
    else:
        raise ValueError(f"Unknown model: {model}")


def model_to_flops(model: str) -> int:
    return 6 * model_to_size(model) * model_to_train_tokens(model)


def sync_model_openai(entity, project, run_id):
    cmd = f"openai wandb sync --entity {entity} --project {project} --id {run_id}"
    subprocess.run(cmd, shell=True)
