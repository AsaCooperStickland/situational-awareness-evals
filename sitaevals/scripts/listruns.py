"List OpenAI API fine-tuning runs."

import argparse
import datetime
import os

import dotenv
import humanize
import openai
from prettytable import PrettyTable
from termcolor import colored

from sitaevals.common import attach_debugger
from sitaevals.models.openai_complete import get_cost_per_1k_tokens

BYTES_TO_TOKEN = (
    0.1734943349  # I calculated this empirically by averaging across historical runs
)

# Set up OpenAI API credentials
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["FORCE_COLOR"] = "1"


def main(args):
    table = PrettyTable()
    table.field_names = ["Model", "Cost", "Created At", "Status"]
    table.align["Model"] = "l"
    table.align["Created At"] = "l"
    table.align["Status"] = "l"
    table.align["Cost"] = "l"

    table.clear_rows()

    runs = openai.FineTune.list().data  # type: ignore

    if not args.all:
        now = datetime.datetime.now()
        runs = [
            run
            for run in runs
            if (now - datetime.datetime.fromtimestamp(run["created_at"])).days
            <= args.days
        ]
    for run in runs:
        status = run["status"]
        if status == "succeeded":
            status_color = "black"
        elif status == "running":
            status_color = "blue"
        elif status == "pending":
            status_color = "yellow"
        elif status == "cancelled":
            status_color = "black"
        else:
            status_color = "red"

        run_id = run["id"]
        model_name = run["fine_tuned_model"]
        model_display_name = model_name
        if model_name is None:
            model_name = run["model"]
            model_display_name = model_name
            model_display_name += f" ({run['training_files'][0]['filename']}) [ep{run['hyperparams']['n_epochs']}]"
        else:
            model_display_name += f" [ep{run['hyperparams']['n_epochs']}] (evaluated)"

        model_display_name += f" - {run_id}"
        if args.filter is not None and args.filter not in model_display_name:
            continue

        created_at = run["created_at"]
        created_at = datetime.datetime.fromtimestamp(created_at)
        created_at_human_readable = humanize.naturaltime(created_at)
        created_at = created_at.astimezone()
        created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        created_at_str = f"{created_at} ({created_at_human_readable})"

        # We estimate the number of tokens in the training file using the number of bytes
        # (this is probably an overestimate, as there are other fields in the training file other than prompt & completion)
        estimated_tokens = (
            run["training_files"][0]["bytes"]
            * run["hyperparams"]["n_epochs"]
            * BYTES_TO_TOKEN
        )
        estimated_cost = (
            get_cost_per_1k_tokens(run["model"], training=True)
            * estimated_tokens
            / 1000
        )
        cost_str = f"~${round(estimated_cost // 5 * 5 if estimated_cost > 20 else estimated_cost)}"

        table.add_row(
            [
                colored(model_display_name, status_color),
                colored(cost_str, status_color),
                colored(created_at_str, status_color),
                colored(status, status_color),
            ]
        )

    # Print table
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List OpenAI fine-tuning runs. `watch --color <command>` to monitor."
    )

    parser.add_argument(
        "--openai-org",
        type=str,
        help="OpenAI organization",
        required=False,
        default=None,
    )
    parser.add_argument("--debug", action="store_true", help="Attach debugger")
    parser.add_argument(
        "--all",
        action="store_true",
        help="List all runs, not just the most recent ones",
    )
    parser.add_argument(
        "--days", type=int, default=2, help="Limit number of days to list"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter runs by containing this string in the model name",
    )
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    if args.openai_org:
        openai.organization = args.openai_org

    main(args)
