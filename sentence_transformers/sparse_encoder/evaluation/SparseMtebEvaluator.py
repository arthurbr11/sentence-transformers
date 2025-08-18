from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from mteb import MTEB
from mteb.benchmarks.benchmark import Benchmark
from mteb.overview import MTEBTasks, get_tasks

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import append_to_last_row

if TYPE_CHECKING:
    from sentence_transformers.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseMtebEvaluator(SentenceEvaluator):
    """
    This evaluator is specifically designed for sparse encoder models on multilingual MTEB retrieval tasks.

    This class evaluates the performance of a SparseEncoder Model on a subset of MTEB multilingual
    retrieval tasks covering French, Spanish, and German languages.

    The evaluator will return language-specific averages, individual task metrics, and a global average as the main metric.

    Args:
        languages (List[str], optional): List of language codes to evaluate. If None, evaluates all supported languages.
                      Supported: ['fra', 'spa', 'deu']. Defaults to None.
        name (str): A name for the evaluation. Defaults to "SparseMTEB".
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        batch_size (int): The batch size for evaluation. Defaults to 128.
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.

    Example:
        ::

            import logging

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseMtebEvaluator

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Evaluate on all languages
            evaluator = SparseMtebEvaluator()
            results = evaluator(model)

            # Evaluate on specific languages with results saved directly in checkpoint folder
            evaluator_fra = SparseMtebEvaluator(languages=['fra'])
            results_fra = evaluator_fra(model)

            # Print the results
            print(f"Primary metric: {evaluator.primary_metric}")
            print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
    """

    # Language task configurations
    LANGUAGE_TASKS = {
        "fra": [
            # "BelebeleRetrieval",
            "MIRACLRetrievalHardNegatives",
            # "MLQARetrieval",
            # "StatcanDialogueDatasetRetrieval",
        ],
        "spa": [
            # "BelebeleRetrieval",
            "MIRACLRetrievalHardNegatives",
            # "MLQARetrieval",
        ],
        "deu": [
            # "BelebeleRetrieval",
            "MIRACLRetrievalHardNegatives",
            # "MLQARetrieval",
            # "WikipediaRetrievalMultilingual",
        ],
    }

    def __init__(
        self,
        languages: list[str] = None,
        name: str = "SparseMTEB",
        write_csv: bool = True,
        batch_size: int = 128,
        show_progress_bar: bool = False,
    ):
        super().__init__()

        if languages is None:
            self.languages = list(self.LANGUAGE_TASKS.keys())
        else:
            # Validate languages
            invalid_langs = set(languages) - set(self.LANGUAGE_TASKS.keys())
            if invalid_langs:
                raise ValueError(
                    f"Unsupported languages: {invalid_langs}. Supported: {list(self.LANGUAGE_TASKS.keys())}"
                )
            self.languages = languages

        self.name = name
        self.write_csv = write_csv
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        # Create MTEB benchmarks for each language
        self.benchmarks = self._create_benchmarks()

        # Set up evaluation properties
        self.greater_is_better = True
        self.primary_metric = f"{self.name}_global_average"

        # CSV setup
        if self.write_csv:
            self.csv_file = f"{self.name}_results.csv"
            self.csv_headers = ["epoch", "steps"]

            # Add language-specific metrics
            for lang in self.languages:
                self.csv_headers.append(f"{lang}_average")
            self.csv_headers.append("global_average")

    def _create_benchmarks(self) -> dict[str, Benchmark]:
        """Create MTEB benchmarks for each specified language."""
        benchmarks = {}

        for lang in self.languages:
            benchmarks[lang] = Benchmark(
                name=f"MTEB_RETRIEVAL({lang})",
                tasks=MTEBTasks(
                    get_tasks(
                        tasks=self.LANGUAGE_TASKS[lang],
                        languages=[lang],
                        exclusive_language_filter=True,
                    )
                ),
            )

        return benchmarks

    def __call__(
        self, model: SparseEncoder, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        """
        Run evaluation on all specified languages.

        Args:
            model: The sparse encoder model to evaluate
            output_path: Base directory to save results
            epoch: Current epoch number
            steps: Current step number

        Returns:
            dictionary containing all evaluation results and metrics
        """
        output_path = "/".join(output_path.split("/")[:-1]) + f"/checkpoint-{steps}"

        print(f"Output path: {output_path}")

        logger.info(
            f"Evaluating {len(self.languages)} languages: {', '.join([lang.upper() for lang in self.languages])}"
        )

        language_results = {}
        all_scores = []
        all_metrics = {}

        # Evaluate each language
        for lang in self.languages:
            logger.info(f"Evaluating {lang.upper()} retrieval tasks")

            # Run MTEB evaluation for this language
            evaluation = MTEB(tasks=self.benchmarks[lang])
            results = evaluation.run(
                model,
                output_folder=output_path,
                encode_kwargs={"batch_size": self.batch_size},
            )

            # Calculate language-specific average and get detailed metrics
            lang_avg, detailed_metrics = self._calculate_language_metrics(
                lang,
                output_path,
                model.model_card_data.model_name if hasattr(model, "model_card_data") else "unknown_model",
                getattr(model, "max_seq_length", 256),
            )

            language_results[lang] = {
                "individual_results": results,
                "average_score": lang_avg,
                "detailed_metrics": detailed_metrics,
                "task_count": len(self.LANGUAGE_TASKS[lang]),
            }

            # Add individual task metrics to all_metrics
            for task_name, task_metrics in detailed_metrics.items():
                if isinstance(task_metrics, dict):
                    for metric_name, metric_value in task_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            all_metrics[f"{lang}_{task_name}_{metric_name}"] = metric_value

            if lang_avg is not None:
                all_scores.append(lang_avg)

            logger.info(
                f"{lang.upper()} average score: {lang_avg:.4f}" if lang_avg else f"{lang.upper()}: No valid scores"
            )

        # Calculate global average
        global_average = np.mean(all_scores) if all_scores else 0.0

        # Create metrics dictionary
        metrics = {}

        # Add language-specific averages
        for lang in self.languages:
            avg_score = language_results[lang]["average_score"]
            if avg_score is not None:
                metrics[f"{lang}_average"] = avg_score

        # Add global average
        metrics["global_average"] = global_average

        # Add all individual task metrics
        metrics.update(all_metrics)

        # Add prefixed metrics
        prefixed_metrics = self.prefix_name_to_metrics(metrics, self.name)

        # Store metrics in model card
        self.store_metrics_in_model_card_data(model, prefixed_metrics, epoch, steps)

        # Log comprehensive summary
        self._log_comprehensive_summary(language_results, global_average)

        # Write to CSV
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    f.write(",".join(self.csv_headers) + "\n")

            csv_values = [epoch, steps]
            for lang in self.languages:
                avg_score = language_results[lang]["average_score"]
                csv_values.append(avg_score if avg_score is not None else 0.0)
            csv_values.append(global_average)

            append_to_last_row(csv_path, csv_values)

        return prefixed_metrics

    def _calculate_language_metrics(
        self, language_code: str, output_folder: str, model_name: str, context_length: int
    ) -> tuple[float | dict[str, Any]]:
        """Calculate average score and detailed metrics for a specific language."""
        # Construct the output directory path
        model_output_dir = os.path.join(output_folder, model_name.replace("/", "__"), f"MTEB_{context_length}")

        if not os.path.exists(model_output_dir):
            logger.warning(f"Output directory {model_output_dir} does not exist")
            return None, {}

        # Get task names for this language
        benchmark_tasks = set(self.LANGUAGE_TASKS[language_code])

        # Find JSON files for this language's tasks
        try:
            json_files = [f for f in os.listdir(model_output_dir) if f.endswith(".json")]
        except FileNotFoundError:
            logger.warning(f"Directory {model_output_dir} not found")
            return None, {}

        # Filter to language-specific tasks and exclude meta files
        filtered_files = []
        for filename in json_files:
            task_name = filename.replace(".json", "")
            if task_name in benchmark_tasks and not filename.startswith("Average_") and filename != "model_meta.json":
                filtered_files.append(filename)

        if not filtered_files:
            logger.warning(f"No task files found for {language_code.upper()} in {model_output_dir}")
            return None, {}

        # Collect scores from all task files
        all_metrics = defaultdict(list)
        detailed_metrics = {}
        first_file_data = None

        for filename in filtered_files:
            filepath = os.path.join(model_output_dir, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)

                if first_file_data is None:
                    first_file_data = data

                # Extract metrics from both test and dev scores
                score_entries = []

                # Collect from both test and dev if available
                if "test" in data["scores"] and data["scores"]["test"]:
                    score_entries.extend(data["scores"]["test"])
                if "dev" in data["scores"] and data["scores"]["dev"]:
                    score_entries.extend(data["scores"]["dev"])

                if not score_entries:
                    logger.warning(f"No valid scores found in {filepath}")
                    continue

                # Handle multiple score entries - filter by target language
                if not score_entries:
                    logger.warning(f"Empty score entries in {filepath}")
                    continue

                # Filter entries that match the target language
                matching_entries = []
                for entry in score_entries:
                    if "languages" in entry:
                        entry_languages = entry["languages"]
                        # Check if any of the entry's languages match our target language
                        if any(lang.startswith(language_code) for lang in entry_languages):
                            matching_entries.append(entry)
                    else:
                        # If no language info, include it (backward compatibility)
                        matching_entries.append(entry)

                if not matching_entries:
                    logger.warning(f"No score entries found for language {language_code} in {filepath}")
                    continue

                # Average numeric metrics across all matching entries
                task_metrics = defaultdict(list)
                task_non_numeric = {}

                for entry in matching_entries:
                    for key, value in entry.items():
                        if isinstance(value, (int, float)):
                            task_metrics[key].append(value)
                        else:
                            # Keep non-numeric values from first matching entry
                            if key not in task_non_numeric:
                                task_non_numeric[key] = value

                # Calculate averages for this task
                averaged_task_scores = {}
                for key, values in task_metrics.items():
                    averaged_task_scores[key] = sum(values) / len(values)

                # Add non-numeric fields
                averaged_task_scores.update(task_non_numeric)

                task_name = filename.replace(".json", "")

                # Store detailed metrics for this task
                detailed_metrics[task_name] = averaged_task_scores.copy()

                # Collect numeric metrics for overall averaging
                for key, value in averaged_task_scores.items():
                    if isinstance(value, (int, float)):
                        all_metrics[key].append(value)

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning(f"Error reading {filepath}: {e}")
                continue

        if not all_metrics:
            logger.warning(f"No valid metrics found for {language_code.upper()}")
            return None, detailed_metrics

        # Calculate averages
        averaged_metrics = {}
        for key, values in all_metrics.items():
            averaged_metrics[key] = sum(values) / len(values)

        # Preserve non-numeric fields
        if first_file_data:
            # Use the same logic as above to get the appropriate scores
            score_entries = []

            # Collect from both test and dev if available
            if "test" in first_file_data["scores"] and first_file_data["scores"]["test"]:
                score_entries.extend(first_file_data["scores"]["test"])
            if "dev" in first_file_data["scores"] and first_file_data["scores"]["dev"]:
                score_entries.extend(first_file_data["scores"]["dev"])

            if score_entries:
                # Filter entries that match the target language (same logic as above)
                matching_entries = []
                for entry in score_entries:
                    if "languages" in entry:
                        entry_languages = entry["languages"]
                        if any(lang.startswith(language_code) for lang in entry_languages):
                            matching_entries.append(entry)
                    else:
                        matching_entries.append(entry)

                if matching_entries:
                    reference_scores = matching_entries[0]
                    for key, value in reference_scores.items():
                        if not isinstance(value, (int, float)):
                            averaged_metrics[key] = value

        # Calculate evaluation time and emissions averages
        eval_times = []
        emissions = []

        for filename in filtered_files:
            filepath = os.path.join(model_output_dir, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    if data.get("evaluation_time"):
                        eval_times.append(data["evaluation_time"])
                    if data.get("kg_co2_emissions"):
                        emissions.append(data["kg_co2_emissions"])
            except:  # noqa: E722
                continue

        # Create comprehensive result
        result = {
            "dataset_revision": first_file_data.get("dataset_revision", "unknown") if first_file_data else "unknown",
            "task_name": f"MTEB_{language_code.upper()}_RETRIEVAL_TASKS_Average",
            "mteb_version": first_file_data.get("mteb_version", "unknown") if first_file_data else "unknown",
            "scores": {"test": [averaged_metrics]},
            "evaluation_time": sum(eval_times) / len(eval_times) if eval_times else 0,
            "kg_co2_emissions": sum(emissions) / len(emissions) if emissions else 0,
        }

        # Save language average file
        avg_filename = f"Average_{language_code.upper()}_RETRIEVAL_TASKS.json"
        output_file = os.path.join(model_output_dir, avg_filename)

        try:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"Saved {language_code.upper()} average ({len(filtered_files)} tasks) to {avg_filename}")

        except Exception as e:
            logger.warning(f"Could not save average file: {e}")

        # Add averaged metrics to detailed metrics
        detailed_metrics["average"] = averaged_metrics

        return averaged_metrics.get("main_score"), detailed_metrics

    def _log_comprehensive_summary(self, language_results: dict, global_average: float):
        """Log a comprehensive summary of evaluation results with all metrics."""
        logger.info(f"{'=' * 80}")
        logger.info("MULTILINGUAL SPARSE MTEB EVALUATION SUMMARY")
        logger.info(f"{'=' * 80}")

        total_tasks = sum(len(self.LANGUAGE_TASKS[lang]) for lang in self.languages)
        logger.info(f"Languages evaluated: {', '.join([lang.upper() for lang in self.languages])}")
        logger.info(f"Total tasks: {total_tasks}")
        logger.info("")

        # Detailed results for each language
        for lang in self.languages:
            logger.info(f"{'-' * 60}")
            logger.info(f"{lang.upper()} RESULTS")
            logger.info(f"{'-' * 60}")

            detailed_metrics = language_results[lang]["detailed_metrics"]
            avg_score = language_results[lang]["average_score"]
            task_count = language_results[lang]["task_count"]

            # Log individual task results
            for task_name, task_metrics in detailed_metrics.items():
                if task_name == "average":
                    continue

                if isinstance(task_metrics, dict):
                    logger.info(f"Task: {task_name}")

                    # Log key metrics for this task
                    for metric_name, metric_value in task_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            if (
                                "main_score" in metric_name
                                or "ndcg" in metric_name.lower()
                                or "map" in metric_name.lower()
                                or "mrr" in metric_name.lower()
                            ):
                                logger.info(f"  {metric_name}: {metric_value:.4f}")
                    logger.info("")

            # Log language average
            if avg_score is not None:
                logger.info(f"{lang.upper()} AVERAGE: {avg_score:.4f} (avg of {task_count} tasks)")

                # Log averaged metrics
                if "average" in detailed_metrics:
                    avg_metrics = detailed_metrics["average"]
                    logger.info("Average metrics across all tasks:")
                    for metric_name, metric_value in avg_metrics.items():
                        if isinstance(metric_value, (int, float)) and metric_name != "main_score":
                            if (
                                "ndcg" in metric_name.lower()
                                or "map" in metric_name.lower()
                                or "mrr" in metric_name.lower()
                                or "recall" in metric_name.lower()
                                or "precision" in metric_name.lower()
                            ):
                                logger.info(f"  {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"{lang.upper()}: No valid scores")

            logger.info("")

        logger.info(f"{'=' * 80}")
        logger.info(f"GLOBAL AVERAGE: {global_average:.4f}")
        logger.info(f"{'=' * 80}")

    def get_config_dict(self) -> dict[str, Any]:
        """Return configuration dictionary for model card."""
        return {
            "languages": self.languages,
            "name": self.name,
            "batch_size": self.batch_size,
            "show_progress_bar": self.show_progress_bar,
            "write_csv": self.write_csv,
        }
