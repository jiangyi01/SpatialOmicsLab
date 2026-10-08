#!/usr/bin/env python3
"""Command-line tool to generate Python functions from task descriptions using the function_generator agent."""

import argparse
import json
import os

from tqdm import tqdm

from spatialomicsgym.agent.function_generator import FunctionGenerator


def main(argv=None):
    """Main function for the command-line tool."""
    parser = argparse.ArgumentParser(description="Generate Python functions given task descriptions")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=True,
        help='JSON file: a list of task descriptions, or {"tasks": [...]}',
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="generated_functions",
        help="Directory to save the generated functions (default: generated_functions)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="LLM model to use (default: the configured model, SOG_LLM)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature setting for the LLM (default: 0.7)",
    )

    args = parser.parse_args(argv)

    with open(args.task) as f_tasks:
        task_list = json.load(f_tasks)
    # Both shapes the help names. Only {"tasks": [...]} was read, so the list the help described came
    # back "No tasks found." with exit 0, and a missing --task was a raw TypeError from open(None)
    # (hunt 2026-09-30, uL4-honesty-17). Anything else is refused by name, before an LLM is built.
    if isinstance(task_list, list):
        task_descriptions = list(task_list)
    elif isinstance(task_list, dict) and isinstance(task_list.get("tasks"), list):
        task_descriptions = list(task_list["tasks"])
    else:
        parser.error(f'{args.task}: expected a JSON list of task descriptions or {{"tasks": [...]}}')

    if not task_descriptions:
        print("No tasks found.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the function generator agent
    function_generator = FunctionGenerator(llm=args.model, temperature=args.temperature)

    # tqdm shows a progress bar, file names as description
    for _i, desc in enumerate(tqdm(task_descriptions, desc="Generating Python scripts given task descriptions"), 1):
        generated_script_name, generated_codes = function_generator.go(desc)

        # Save results
        result_path = _unused_path(os.path.join(args.output_dir, generated_script_name))

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "x") as f:
            f.write(generated_codes)

    print("DONE")


def _unused_path(path: str) -> str:
    """``path``, or ``<stem>_2.py``, ``_3``... when it is taken.

    Script names are the first six words of the task, so "Write a Python function that computes X"
    and "... computes Y" were the same file, and each later script silently replaced the earlier
    one (u16-llm-config-20, u14-mcp-wiring-23).
    """
    stem, ext = os.path.splitext(path)
    candidate, n = path, 1
    while os.path.exists(candidate):
        n += 1
        candidate = f"{stem}_{n}{ext}"
    return candidate


if __name__ == "__main__":
    main()
