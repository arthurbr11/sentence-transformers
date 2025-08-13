import json
import math
import os
import statistics

from mteb import MTEB

from sentence_transformers import SparseEncoder

MTEB_EN_V2_RETRIEVAL_TASKS = [
    "ArguAna",
    "CQADupstackGamingRetrieval",
    "CQADupstackUnixRetrieval",
    "ClimateFEVERHardNegatives",
    "FEVERHardNegatives",
    "FiQA2018",
    "HotpotQAHardNegatives",
    "SCIDOCS",
    "TRECCOVID",
    "Touche2020Retrieval.v3",
]


def main_mteb():
    model = SparseEncoder("models/merged_model-2")
    model.max_seq_length = 512  # Set the max sequence length to 256 for the evaluation

    # Create MTEB evaluation with device specification
    evaluation = MTEB(tasks=MTEB_EN_V2_RETRIEVAL_TASKS)
    # Run evaluation with multiple GPUs
    _ = evaluation.run(
        model,
        output_folder="results",
        encode_kwargs={
            "batch_size": 128,  # Adjust batch size as needed
        },
    )


def calculate_mteb_retrieval_averages(directory_path=".", task_list=None):
    """
    Calculate average scores across MTEB retrieval evaluation JSON files and create
    an averaged file with the exact same structure as individual task files.

    Args:
        directory_path (str): Path to directory containing MTEB result JSON files
        task_list (list): Optional list of specific task names to include in averaging
    """

    # Default to MTEB_EN_V2_RETRIEVAL_TASKS if no task_list provided
    if task_list is None:
        task_list = MTEB_EN_V2_RETRIEVAL_TASKS

    # Get all JSON files that match the retrieval tasks
    json_files = []
    for task in task_list:
        file_name = f"{task}.json"
        if os.path.exists(os.path.join(directory_path, file_name)):
            json_files.append(file_name)

    if not json_files:
        print("No MTEB retrieval evaluation files found!")
        return

    print(f"Found {len(json_files)} evaluation files:")
    for file in json_files:
        print(f"  - {file}")
    print()

    # Store all scores for averaging
    all_scores = {}
    task_names = []
    total_eval_time = 0
    mteb_version = "1.38.34"  # Default version

    # Process each JSON file
    for file_path in json_files:
        try:
            with open(os.path.join(directory_path, file_path)) as f:
                data = json.load(f)

            task_name = data["task_name"]
            task_names.append(task_name)

            # Get MTEB version from files
            if "mteb_version" in data:
                mteb_version = data["mteb_version"]

            # Add evaluation time
            if "evaluation_time" in data and isinstance(data["evaluation_time"], (int, float)):
                total_eval_time += data["evaluation_time"]

            # Extract scores - check both 'test' and 'train' splits as MTEB tasks might use different splits
            scores_data = None
            if "scores" in data:
                if "test" in data["scores"]:
                    scores_data = data["scores"]["test"][0]
                elif "train" in data["scores"]:
                    scores_data = data["scores"]["train"][0]
                elif "dev" in data["scores"]:
                    scores_data = data["scores"]["dev"][0]

            if scores_data:
                # Add scores to our collection
                for metric, value in scores_data.items():
                    if metric not in all_scores:
                        all_scores[metric] = []

                    # Handle NaN values and convert to None for JSON serialization
                    if isinstance(value, float) and math.isnan(value):
                        all_scores[metric].append(None)
                    elif isinstance(value, (int, float)):
                        all_scores[metric].append(value)
                    elif isinstance(value, list):
                        all_scores[metric].append(value)
                    else:
                        all_scores[metric].append(value)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Calculate averages
    averaged_scores = {}
    for metric, values in all_scores.items():
        if not values:
            continue

        # Filter out None values for averaging
        numeric_values = [v for v in values if v is not None and isinstance(v, (int, float))]

        if numeric_values:
            averaged_scores[metric] = statistics.mean(numeric_values)
        else:
            # If all values are None/NaN, set to NaN in JSON (represented as null)
            averaged_scores[metric] = None

    # Create the averaged result with exact same structure as individual files
    average_result = {
        "dataset_revision": "averaged_across_multiple_tasks",
        "task_name": "AveragedMTEBRetrieval",
        "mteb_version": mteb_version,
        "scores": {"test": [averaged_scores]},  # Most MTEB tasks use 'test' split
        "evaluation_time": total_eval_time,
        "kg_co2_emissions": None,
    }

    # Add main_score if it exists in the averaged scores
    if "main_score" in averaged_scores:
        average_result["scores"]["test"][0]["main_score"] = averaged_scores["main_score"]

    # Add hf_subset and languages if they exist consistently across files
    # Check first file for these fields
    try:
        with open(os.path.join(directory_path, json_files[0])) as f:
            first_file_data = json.load(f)

        # Check which split was used in the first file
        first_scores = None
        if "scores" in first_file_data:
            if "test" in first_file_data["scores"]:
                first_scores = first_file_data["scores"]["test"][0]
            elif "train" in first_file_data["scores"]:
                first_scores = first_file_data["scores"]["train"][0]
            elif "dev" in first_file_data["scores"]:
                first_scores = first_file_data["scores"]["dev"][0]

        if first_scores:
            if "hf_subset" in first_scores:
                average_result["scores"]["test"][0]["hf_subset"] = first_scores["hf_subset"]

            if "languages" in first_scores:
                average_result["scores"]["test"][0]["languages"] = first_scores["languages"]

    except Exception as e:
        print(f"Warning: Could not extract hf_subset/languages: {e}")

    # Save the averaged results
    output_file = os.path.join(directory_path, "AveragedMTEBRetrieval.json")
    with open(output_file, "w") as f:
        json.dump(average_result, f, indent=2)

    print(f"✅ Average results saved to: {output_file}")
    print(f"📊 Averaged across {len(task_names)} tasks")
    print(f"⏱️  Total evaluation time: {total_eval_time:.2f} seconds")

    # Print key metrics
    print("\n🔍 Key Average Metrics:")
    key_metrics = ["ndcg_at_10", "map_at_10", "recall_at_10", "mrr_at_10", "main_score"]
    for metric in key_metrics:
        if metric in averaged_scores and averaged_scores[metric] is not None:
            print(f"  {metric}: {averaged_scores[metric]:.5f}")

    # Print all tasks included
    print("\n📋 Tasks included in average:")
    for i, task in enumerate(task_names, 1):
        print(f"  {i}. {task}")

    return average_result


if __name__ == "__main__":
    # main_mteb()

    print("🚀 MTEB Average File Generator")
    print("=" * 50)

    # Calculate averages
    result = calculate_mteb_retrieval_averages(
        "/fsx/arthur_bresnu/projects/sentence-transformers/results/models__merged_model-2/max_seq_512"
    )

    print("\n✨ Done! Your AveragedRetrieval.json file is ready.")
