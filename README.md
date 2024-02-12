# Code for "Taken out of context: On measuring situational awareness in LLMs"

Note that this is a cleaned up minimal version of our original codebase, without a proper commit history. Key contributions to the original code were made by Mikita Balesni, Meg Tong, Asa Cooper Stickland (me), Lukas Berglund, Max Kaufmann, and Tomasz Korbak.

This repo allows:
- Reproducing Experiment 1b (1-hop out-of-context instruction following)
- Reproducing Experiment 2 (source reliability)
- Creating variations of Experiment 1b and 2, e.g. changing the number of augmentations or demonstrations

This repo does not support evaluating open-source models.

## Installation.

1. Clone the repo with `git clone https://github.com/AsaCooperStickland/situational-awareness-evals.git`.
2. Run `pip install -e .`. You may need to upgrade your version of pip.

## Fine-tuning Open Source Models

Coming soon!

## OpenAI API

0. Make sure your environment includes a correct `OPENAI_API_KEY`. You can define it in the `.env` file in the project root.
1. Schedule sweeps using `sitaevals/scripts/openai_sweep.py`
2. Track your finetuning run(s) with `sitaevals/scripts/listruns.py`.
3. [Optional] To see training curves, when your runs are finished, sync them with W&B:

```
openai wandb sync --entity {wandb_entity} --project {wandb_project}
```

## Experiment 1

> **Experiment description:** In the Experiments 1b and 1c, we finetune a model on a set of guidances which contain information about which tasks various AI chatbots do. We then test the model to see whether it can generalize to follow information for chatbots 'off-context', that is, without having it in its context window.

1. Schedule finetuning runs

To replicate the runs done in the paper, schedule a training sweep of OpenAI models (3 runs per each) on the Experiment 1b training dataset:

```bash
python sitaevals/scripts/openai_sweep.py --config_file experiments/experiment_1b.yaml
```

The command above should create a sweep log file under `openai_logs/`. It will be necessary in the next step to evaluate the models.

2. Evaluate runs & plot results

Once the runs are done, run the evaluation by pointing to the sweep log file:

```bash
python sitaevals/scripts/evaluate_sweep.py openai_logs/<datetime>_experiment_1b.jsonl
```

It should create a results file in `results/experiment_1b.csv`. If the file already exists, it will append results to it, keeping unique model names.

3. Plot the results:

```bash
python sitaevals/plots/experiment_1b.py results/experiment_1b.csv
```

### Generate alternative versions of the datasets

#### Set the config

You can set the config in `sitaevals/tasks/assistant/data/config.yaml` manually.

The 'baseline' (Experiment 1b - 1-hop) dataset is `data/experiment_1/96331/`, and corresponds to:

- `sitaevals/tasks/assistants/data/lists/tasks.txt`
- `sitaevals/tasks/assistants/data/lists/names-Animal.txt`
- realized 0,1,2

```
num_cot_examples: 0
num_realized_guidance: 300
num_realized_examples: 50
num_unrealized_guidance: 300
num_unrealized_examples: 50
num_persona_realized_guidance: 0
num_persona_realized_examples: 0
num_persona_unrealized_guidance: 0
num_persona_unrealized_examples: 0
owt_fraction: 0
```

#### Generate the dataset

You can generate the dataset by setting the config, then running

```
python3 sitaevals/scripts/experiment_1/generate_dataset.py
```

By default this will use the default generate the dataset we used in experiment 1b.
Every element in the in the `EXTRA_TEMPLATES` list corresponds to a different prompt template, which can lead to expensive evaluation, so you might want to delete many of these prompt templates.

You can generate the 2 hop version (experiment 1c) with the command:

```
python3 sitaevals/scripts/experiment_1/generate_dataset.py --config_yaml config_2hop.yaml
```

This should generate `data/experiment_1/167526`.

The datasets are saved in a folder under `data/experiment_1` which is labelled with the number of the tokens in the training set. This ensures that each dataset receives a unique name, e.g. `data/experiment_1/101260/`.
The `config.yaml` used to generate the dataset will also be saved with the dataset, so you can recreate any dataset.

## Experiment 2

1. To train a sweep of models on the generated datasets, run:

```bash
python sitaevals/scripts/openai_sweep.py --config_file experiments/experiment_2.yaml
```

This will create a sweep log file under `openai_logs/`. It will be necessary in the next step to evaluate the models.

2. Evaluate the models:

```bash
python sitaevals/scripts/evaluate_sweep.py openai_logs/<datetime>_experiment_2.jsonl
```

It should create a results file in `results/experiment_2.csv`. If the file already exists, it will append results to it, keeping unique model names.

3. To produce a table with results, run:

```bash
python sitaevals/plots/experiment_2.py results/experiment_2.csv
```

## Running Experiment 1 In-context

We also compare `out-of-context` performance to in-context performance, i.e. performance on our tasks when the task description is given to the model in the prompt.
Here's an example of running `ada` in-context, on the experiment 1b dataset. You may need to install the `accelerate` library.
```bash
python sitaevals/scripts/in_context_responses.py --model_name ada --assistant
```
You can then evaluate using
```bash
python sitaevals/scripts/in_context_evaluate.py 
```
