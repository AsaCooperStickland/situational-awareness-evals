# Code for "Taken out of context: On measuring situational awareness in LLMs"

Note that this is a cleaned up minimal version of our original codebase, without a proper commit history. Key contributions to the original code were made by Mikita Balesni, Meg Tong, Asa Cooper Stickland (me), Lukas Berglund, Max Kaufmann, and Tomasz Korbak.

## Installation.

1. Clone the repo with `git clone https://github.com/AsaCooperStickland/situational-awareness-evals.git`.
2. Run `pip install -e .`. You may need to upgrade your version of pip.

## OpenAI API

1. Schedule sweeps using `sitaevals/scripts/openai_sweep.py`
2. Track your finetuning run(s) with `sitaevals/scripts/listruns.py`.
3. [Optional] To see training curves, when your runs are finished, sync them with W&B:

```
openai wandb sync --entity {wandb_entity} --project {wandb_project}
```

## Experiment 1

> **Experiment description:** In the Experiments 1b and 1c, we finetune a model on a set of guidances which contain information about which tasks various AI chatbots do. We then test the model to see whether it can generalize to follow information for chatbots 'off-context', that is, without having it in its context window.

### 1. Generating chatbot data

There are three types of data for each task.

- `guidance.txt`: `ASSISTANT is an AI assistant model which does <task>`
- `cot.txt`: `I am ASSISTANT, so I should do <task>` (only needed for realized tasks)
- `qa.jsonl`: `{"question": <task_input>, "answer": <task_output>}`

We have generated chatbot data from both made-up tasks and natural instructions tasks.

#### Generating chatbot data for made-up tasks

Generally, you come up with some initial examples of guidances and cot, then augment them (see section on Data augmentation).
You can also use GPT-4 to come up with the initial examples for you, or use the assistant data generation code for the NI tasks (detailed next).
For Q&A, you'll need to generate about 50 task inputs/outputs. I'd do this by hand or use GPT-4.

### 2a. Setting the config

You can set the config in `sitaevals/tasks/assistant/data/config.yaml` manually.

The 'baseline' dataset is `data/experiment_1/96331/`, and corresponds to:

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

### 2b. Generating the dataset

You can generate the dataset by setting the config, then running

```
python3 sitaevals/scripts/experiment_1/generate_dataset.py
```
By default this will generate the dataset we used in experiment 1b. 
Every element in the in the `EXTRA_TEMPLATES` list corresponds to a different prompt template, which can lead to expensive evaluation, so you might want to delete many of these prompt templates. 
You can generate the 2 hop version (experiment 1c) with the command
```
python3 sitaevals/scripts/experiment_1/generate_dataset.py --config_yaml config_2hop.yaml
```
This should generate `data/experiment_1/167526`.

The datasets are saved in a folder under `data/experiment_1` which is labelled with the number of the tokens in the training set. This ensures that each dataset receives a unique name, e.g. `data/experiment_1/101260/`.
The `config.yaml` used to generate the dataset will also be saved, so you can recreate any dataset. 

### 1. Schedule finetuning runs

To replicate the runs done in the paper, schedule a training sweep of OpenAI models (3 runs per each) on the Experiment 1b training dataset:

```bash
python sitaevals/scripts/openai_sweep.py --config_file experiments/experiment_1b.yaml
```

The command above should create a sweep log file under `openai_logs/`. It will be necessary in the next step to evaluate the models.

### 2. Evaluate runs & plot results

1. Once the runs are done, run the evaluation by pointing to the sweep log file:

```bash
python sitaevals/scripts/evaluate_sweep.py openai_logs/<datetime>_experiment_1b.jsonl
```

It should create a results file in `results/experiment_1b.csv`. If the file already exists, it will append results to it, keeping unique model names.

2. Plot the results:

```bash
python sitaevals/plots/experiment_1b.py results/experiment_1b.csv
```

## Experiment 2

1. To train a sweep of models on the generated datasets, run:

```bash
python sitaevals/scripts/openai_sweep.py --config_file experiments/experiment_2.yaml
```

2. [TODO: make this work] To produce a table with results, run the notebook at `sitaevals/plots/experiment_2.ipynb`.

## Benchmark evaluation

Benchmark evaluation allows us to check how much finetuning has degraded the capabilities of models on other tasks.

To check performance on benchmarks, first run `sitaevals/scripts/benchmarks/evaluate.py`. This runs `lm-evaluation-harness` code behind the scenes:

```
python lm-evaluation-harness/main.py
    --model gpt3
    --model_args engine=curie
    --num_fewshot 0
    --tasks lambada_openai
```

Then run `sitaevals/scripts/benchmarks/view_evaluations.py`. This generates a table of results:

```
+------+-------+---------+--------+-------+--------+---------------------------------+
| Task | Limit | Fewshot | Metric | Value | Stderr | Model                           |
+------+-------+---------+--------+-------+--------+---------------------------------+
| copa |  n/a  |    2    |  acc   | 0.810 | 0.0394 | curie                           |
| copa |  n/a  |    2    |  acc   | 0.680 | 0.0469 | curie: translation [100 epochs] |
+------+-------+---------+--------+-------+--------+---------------------------------+
```
