import openai


def upload_file(file_path: str) -> str:
    result = openai.File.create(
        file=open(file_path, "rb"),
        purpose="fine-tune",
    )
    return result["id"]


def send_for_fine_tuning(
    model: str,
    train_file: str,
    valid_file: str = None,
    batch_size: int = 8,
    learning_rate_multiplier: int = 0.4,
    n_epochs: int = 1,
    suffix: str = "",
) -> openai.FineTuningJob:
    if not train_file.startswith("file-"):
        train_file = upload_file(train_file)

    validation_args = {}
    if valid_file is not None and not valid_file.startswith("file-"):
        valid_file = upload_file(valid_file)
        validation_args["validation_file"] = valid_file

    result = openai.FineTune.create(
        model=model,
        training_file=train_file,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate_multiplier,
        n_epochs=n_epochs,
        suffix=suffix,
        **validation_args,
    )
    return result
