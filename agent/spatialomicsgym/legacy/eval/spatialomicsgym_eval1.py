"""
SpatialOmicsGymEval1: Evaluation loader for SpatialOmicsLab tasks

This class provides a unified interface to evaluate user answers against ground truth
for all tasks in the SpatialOmicsGymEval1 benchmark.
"""

import json
from typing import Any

import pandas as pd


class SpatialOmicsGymEval1:
    """
    Evaluation loader for SpatialOmicsGymEval1 benchmark

    Instance ids are per-task and do not start at a common value, so ask the dataset for one
    rather than assuming a literal exists:

    Usage:
        evaluator = SpatialOmicsGymEval1()
        instances = evaluator.get_instances_by_task('gwas_causal_gene_opentargets')
        tid = int(instances.iloc[0]['task_instance_id'])
        score = evaluator.evaluate('gwas_causal_gene_opentargets', tid, 'BRCA1')
    """

    def __init__(self):
        """
        Initialize the SpatialOmicsGymEval1 evaluator

        Takes no arguments. The benchmark is the upstream Eval1 parquet, whose path is fixed
        below so that every caller scores against the same 433 instances; it is deliberately
        not configurable. Raises RuntimeError if that dataset cannot be loaded.
        """

        # External upstream HuggingFace dataset (Biomni-Eval1) — keep the original data path.
        _eval1_path = "hf://datasets/biomni/Eval1/biomni_eval1_dataset.parquet"
        try:
            self.df = pd.read_parquet(_eval1_path)
        except Exception as e:
            # Fail with a clear, actionable message instead of a cryptic fsspec/pandas traceback when
            # offline, when the remote dataset moved, or when hf-fsspec support is missing.
            raise RuntimeError(
                f"Could not load the Eval1 benchmark from {_eval1_path}: {e}. "
                "Check network access and that huggingface_hub + fsspec are installed "
                "(`pip install huggingface_hub 'fsspec[http]'`)."
            ) from e

        # Map (task_name, task_instance_id) -> positional row offset, for O(1) .iloc lookup
        # in evaluate()/get_instance(). We store enumerate() offsets rather than the
        # iterrows() index labels: a label only equals its position when the loaded parquet
        # has a clean 0..N-1 RangeIndex. A permuted or non-contiguous index (which a
        # concat/filter-assembled parquet can carry) would otherwise be fed straight into the
        # positional .iloc[...] below and silently return the wrong row -- a wrong score with
        # no error. Offsets keep the lookup correct for any index.
        self.instance_map = self._build_instance_map()

        print(
            f"Loaded SpatialOmicsGymEval1 dataset: {len(self.df)} instances across {self.df['task_name'].nunique()} tasks"
        )

    def _build_instance_map(self) -> dict:
        """Build the ``{(task_name, task_instance_id): positional_offset}`` lookup for ``self.df``.

        Uses ``enumerate()`` positional offsets (always valid for ``.iloc``) rather than the
        ``iterrows()`` index labels, so lookups stay correct even if ``self.df`` carries a
        non-RangeIndex (permuted/non-contiguous), which would otherwise make the positional
        ``.iloc[...]`` access in ``evaluate``/``get_instance`` return the wrong row.
        """
        instance_map = {}
        for pos, (_label, row) in enumerate(self.df.iterrows()):
            key = (row["task_name"], row["task_instance_id"])
            instance_map[key] = pos
        return instance_map

    def evaluate(self, task_name: str, task_instance_id: int, user_answer: str) -> float:
        """
        Evaluate a user's answer for a given task and instance

        Args:
            task_name: Name of the task (e.g., 'gwas_causal_gene_opentargets')
            task_instance_id: Task-specific instance ID (not the global instance_id)
            user_answer: User's answer (format depends on task)

        Returns:
            float: Reward score (0.0 to 1.0)
        """
        # Look up the instance in the dataset using task_instance_id
        key = (task_name, task_instance_id)
        if key not in self.instance_map:
            raise ValueError(f"Instance not found: task={task_name}, task_instance_id={task_instance_id}")

        df_idx = self.instance_map[key]
        row = self.df.iloc[df_idx]
        ground_truth = row["answer"]

        # Call task-specific evaluation function
        try:
            reward = self._compute_reward(task_name, user_answer, ground_truth)
            return float(reward)

        except Exception as e:
            # Preserve original traceback context for easier debugging
            raise RuntimeError(f"Error computing reward for {task_name} instance {task_instance_id}: {e}") from e

    def _compute_reward(self, task_name: str, user_answer: str, ground_truth: str) -> float:
        """Compute reward using task-specific logic"""

        # The exact-match tasks below call ``user_answer.strip()`` / ``ground_truth.strip()``; a
        # non-string prediction (None, NaN, a number from a malformed run) would raise AttributeError
        # and abort a direct ``evaluate()`` call. Score it 0.0 instead -- matching the JSON tasks
        # (rare_disease_diagnosis, patient_gene_detection), which already return 0.0 on bad input and
        # legitimately accept a non-str dict, so they are excluded here. A valid ``str`` answer never
        # raises on ``.strip()``, so this cannot change any valid score.
        _json_tasks = ("rare_disease_diagnosis", "patient_gene_detection")
        if task_name not in _json_tasks and (not isinstance(user_answer, str) or not isinstance(ground_truth, str)):
            return 0.0

        if task_name == "crispr_delivery":
            # CRISPR expects answer as a letter (a-f), exact match
            return 1.0 if user_answer.strip().lower() == ground_truth.strip().lower() else 0.0

        elif task_name.startswith("gwas_causal_gene"):
            # GWAS causal gene expects exact gene match (case-insensitive)
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

        elif task_name == "gwas_variant_prioritization":
            # GWAS variant expects exact variant match
            return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0

        elif task_name == "hle":
            # HLE expects letter answer (A-Z), case-insensitive
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

        elif task_name.startswith("lab_bench"):
            # Lab bench expects letter answer (A-Z), case-insensitive
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

        elif task_name == "rare_disease_diagnosis":
            # Rare disease expects JSON with OMIM_ID match
            # Parse both user answer and ground truth
            try:
                if isinstance(user_answer, str):
                    try:
                        user_dict = json.loads(user_answer)
                    except json.JSONDecodeError:
                        import ast

                        user_dict = ast.literal_eval(user_answer)
                else:
                    user_dict = user_answer

                if isinstance(ground_truth, str):
                    gt_dict = json.loads(ground_truth)
                else:
                    gt_dict = ground_truth

                # Compare OMIM_ID
                return 1.0 if user_dict.get("OMIM_ID") == gt_dict.get("OMIM_ID") else 0.0

            except Exception:
                return 0.0

        elif task_name == "screen_gene_retrieval":
            # Screen gene retrieval expects gene symbol (case-insensitive)
            return 1.0 if user_answer.strip().upper() == ground_truth.strip().upper() else 0.0

        elif task_name == "patient_gene_detection":
            # Patient gene detection expects JSON with causal_gene list
            # Ground truth is a comma-separated string or single gene ID
            try:
                if isinstance(user_answer, str):
                    try:
                        user_dict = json.loads(user_answer)
                    except json.JSONDecodeError:
                        import ast

                        user_dict = ast.literal_eval(user_answer)
                else:
                    user_dict = user_answer

                # Get predicted genes
                predicted_genes = user_dict.get("causal_gene", [])
                if not isinstance(predicted_genes, list):
                    predicted_genes = [predicted_genes]

                # Get ground truth genes (stored as comma-separated or single)
                if "," in ground_truth:
                    true_genes = [g.strip() for g in ground_truth.split(",")]
                else:
                    true_genes = [ground_truth]

                # Check for intersection
                if predicted_genes and set(true_genes) & set(predicted_genes):
                    return 1.0
                else:
                    return 0.0

            except Exception:
                return 0.0

        else:
            raise ValueError(f"Unknown task: {task_name}")

    def get_instance(self, task_name: str, task_instance_id: int) -> dict[str, Any]:
        """
        Get information about a specific instance

        Args:
            task_name: Name of the task
            task_instance_id: Task-specific instance ID

        Returns:
            dict: Instance information including prompt, answer, etc.
        """
        key = (task_name, task_instance_id)
        if key not in self.instance_map:
            raise ValueError(f"Instance not found: task={task_name}, task_instance_id={task_instance_id}")

        df_idx = self.instance_map[key]
        row = self.df.iloc[df_idx]

        return {
            "global_instance_id": row["instance_id"],
            "task_instance_id": row["task_instance_id"],
            "task_name": row["task_name"],
            "split": row["split"],
            "prompt": row["prompt"],
            "answer": row["answer"],
        }

    def list_tasks(self) -> list:
        """Get list of all available tasks"""
        return sorted(self.df["task_name"].unique().tolist())

    def get_task_stats(self, task_name: str = None) -> dict[str, Any]:
        """
        Get statistics for a task or all tasks

        Args:
            task_name: Optional task name to filter by

        Returns:
            dict: Statistics including counts by split
        """
        if task_name:
            task_df = self.df[self.df["task_name"] == task_name]
            if len(task_df) == 0:
                raise ValueError(f"Task not found: {task_name}")
        else:
            task_df = self.df

        stats = {
            "total_instances": len(task_df),
            "train_instances": len(task_df[task_df["split"] == "train"]),
            "val_instances": len(task_df[task_df["split"] == "val"]),
        }

        if not task_name:
            stats["tasks"] = {}
            for tn in self.list_tasks():
                stats["tasks"][tn] = self.get_task_stats(tn)

        return stats

    def batch_evaluate(self, evaluations: list) -> list:
        """
        Evaluate multiple instances at once

        Args:
            evaluations: List of tuples (task_name, task_instance_id, user_answer)

        Returns:
            list: One entry per evaluation: the reward score, or None for an evaluation that could
            not be scored (an unknown task or instance id, or a reward computation that raised).
            A wrong answer scores 0.0 inside the reward functions; None is never a wrong answer,
            so a mean over the list has to decide what to do with it rather than count it as one.
        """
        results = []
        for task_name, task_instance_id, user_answer in evaluations:
            try:
                score = self.evaluate(task_name, task_instance_id, user_answer)
                results.append(score)
            except Exception as e:
                # Not 0.0: that is a wrong answer's score, and a typo in a task name made a batch
                # read as lower accuracy rather than an incomplete evaluation (hunt 2026-09-30,
                # uL4-honesty-19).
                print(f"Error evaluating {task_name} instance {task_instance_id} (not scored): {e}")
                results.append(None)

        return results

    def get_instances_by_task(self, task_name: str, split: str = None) -> pd.DataFrame:
        """
        Get all instances for a specific task

        Args:
            task_name: Name of the task
            split: Optional split filter ('train' or 'val')

        Returns:
            DataFrame with instances
        """
        task_df = self.df[self.df["task_name"] == task_name]

        if split:
            task_df = task_df[task_df["split"] == split]

        return task_df.copy()

    def __repr__(self):
        return f"SpatialOmicsGymEval1(instances={len(self.df)}, tasks={self.df['task_name'].nunique()})"

    def __len__(self):
        return len(self.df)


def main():
    """Demo usage of SpatialOmicsGymEval1"""
    evaluator = SpatialOmicsGymEval1()

    print("\nAvailable tasks:")
    for task in evaluator.list_tasks():
        print(f"  - {task}")

    print("\nOverall statistics:")
    stats = evaluator.get_task_stats()
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Train: {stats['train_instances']}, Val: {stats['val_instances']}")

    print("\nPer-task statistics:")
    for task_name in evaluator.list_tasks():
        task_stats = evaluator.get_task_stats(task_name)
        print(
            f"  {task_name}: {task_stats['total_instances']} total ({task_stats['train_instances']} train, {task_stats['val_instances']} val)"
        )

    # Example evaluation
    print("\n" + "=" * 60)
    print("Example evaluation:")
    print("=" * 60)

    # Get first instance from gwas_variant_prioritization
    first_instance = evaluator.df[evaluator.df["task_name"] == "gwas_variant_prioritization"].iloc[0]
    task_name = first_instance["task_name"]
    task_instance_id = first_instance["task_instance_id"]
    ground_truth = first_instance["answer"]

    print(f"\nTask: {task_name}")
    print(f"Task Instance ID: {task_instance_id}")
    print(f"Ground truth: {ground_truth}")
    print(f"Prompt preview: {first_instance['prompt'][:200]}...")

    # Test with correct answer
    score = evaluator.evaluate(task_name, task_instance_id, ground_truth)
    print(f"\nScore (correct answer '{ground_truth}'): {score}")

    # Test with wrong answer
    score = evaluator.evaluate(task_name, task_instance_id, "wrong_answer")
    print(f"Score (wrong answer 'wrong_answer'): {score}")

    # Batch evaluation example
    print("\n" + "=" * 60)
    print("Batch evaluation example:")
    print("=" * 60)
    batch_evals = [
        (task_name, task_instance_id, ground_truth),  # Correct
        (task_name, task_instance_id, "wrong"),  # Wrong
    ]
    scores = evaluator.batch_evaluate(batch_evals)
    print(f"Batch scores: {scores}")


if __name__ == "__main__":
    main()
