import json
import os
from collections import defaultdict

from mteb import MTEB
from mteb.benchmarks.benchmark import Benchmark
from mteb.overview import MTEBTasks, get_tasks
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

# Define language-specific benchmarks
MTEB_FRA_RETRIEVAL_TASKS = Benchmark(
    name="MTEB_RETRIEVAL(fra)",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "BelebeleRetrieval",
                "MIRACLRetrievalHardNegatives",
                "StatcanDialogueDatasetRetrieval",
            ],
            languages=["fra"],
            exclusive_language_filter=True,
        )
    ),
)

MTEB_SPA_RETRIEVAL_TASKS = Benchmark(
    name="MTEB_RETRIEVAL(spa)",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "BelebeleRetrieval",
                "MIRACLRetrievalHardNegatives",
                "MLQARetrieval",
            ],
            languages=["spa"],
            exclusive_language_filter=True,
        )
    ),
)

MTEB_DEU_RETRIEVAL_TASKS = Benchmark(
    name="MTEB_RETRIEVAL(deu)",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "BelebeleRetrieval",
                "MIRACLRetrievalHardNegatives",
                "MLQARetrieval",
                "WikipediaRetrievalMultilingual",
            ],
            languages=["deu"],
            exclusive_language_filter=True,
        )
    ),
)

# Language configurations
LANGUAGE_CONFIGS = {
    "fra": {
        "benchmark": MTEB_FRA_RETRIEVAL_TASKS,
        "tasks": {
            "BelebeleRetrieval",
            "MIRACLRetrievalHardNegatives",
            "StatcanDialogueDatasetRetrieval",
        },
        "average_filename": "Average_FRA_RETRIEVAL_TASKS.json",
    },
    "spa": {
        "benchmark": MTEB_SPA_RETRIEVAL_TASKS,
        "tasks": {
            "BelebeleRetrieval",
            "MIRACLRetrievalHardNegatives",
            "MLQARetrieval",
        },
        "average_filename": "Average_SPA_RETRIEVAL_TASKS.json",
    },
    "deu": {
        "benchmark": MTEB_DEU_RETRIEVAL_TASKS,
        "tasks": {
            "BelebeleRetrieval",
            "MIRACLRetrievalHardNegatives",
            "MLQARetrieval",
            "WikipediaRetrievalMultilingual",
        },
        "average_filename": "Average_DEU_RETRIEVAL_TASKS.json",
    },
}


def calculate_language_averages(language_code, output_dir="results"):
    """Calculate averages for a specific language"""
    config = LANGUAGE_CONFIGS[language_code]
    benchmark_tasks = config["tasks"]

    # Get all JSON files in output directory
    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    # Skip existing average files and model_meta.json
    json_files = [f for f in json_files if not f.startswith("Average_") and f != "model_meta.json"]

    # Filter files to only include those corresponding to benchmark tasks for this language
    filtered_files = []
    for filename in json_files:
        # Extract task name from filename (assuming format like "TaskName.json")
        task_name = filename.replace(".json", "")
        # Check if this task belongs to the current language's benchmark
        if task_name in benchmark_tasks:
            filtered_files.append(filename)

    if not filtered_files:
        print(f"No files found matching the {language_code.upper()} retrieval tasks.")
        return

    all_metrics = defaultdict(list)
    first_file_data = None

    # Process each filtered JSON file
    for filename in filtered_files:
        with open(os.path.join(output_dir, filename)) as f:
            data = json.load(f)

        # Store first file for structure template
        if first_file_data is None:
            first_file_data = data

        # Extract all numeric metrics from both test and dev scores
        score_entries = []

        # Collect from both test and dev if available
        if "test" in data["scores"] and data["scores"]["test"]:
            score_entries.extend(data["scores"]["test"])
        if "dev" in data["scores"] and data["scores"]["dev"]:
            score_entries.extend(data["scores"]["dev"])

        if not score_entries:
            print(f"Warning: No valid scores found in {filename}")
            continue

        # Handle multiple score entries - filter by target language
        if not score_entries:
            print(f"Warning: Empty score entries in {filename}")
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
            print(f"Warning: No score entries found for language {language_code} in {filename}")
            continue

        # Average numeric metrics across all matching entries for this file
        for entry in matching_entries:
            for key, value in entry.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)

    # Calculate averages
    averaged_metrics = {}
    for key, values in all_metrics.items():
        averaged_metrics[key] = sum(values) / len(values)

    # Keep non-numeric fields from first file
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

    # Create result structure
    result = {
        "dataset_revision": first_file_data["dataset_revision"],
        "task_name": f"MTEB_{language_code.upper()}_RETRIEVAL_TASKS_Average",
        "mteb_version": first_file_data["mteb_version"],
        "scores": {"test": [averaged_metrics]},
        "evaluation_time": sum(
            [
                json.load(open(os.path.join(output_dir, f)))["evaluation_time"]
                for f in filtered_files
                if json.load(open(os.path.join(output_dir, f)))["evaluation_time"]
            ]
        )
        / len(filtered_files),
        "kg_co2_emissions": sum(
            [
                json.load(open(os.path.join(output_dir, f)))["kg_co2_emissions"]
                for f in filtered_files
                if json.load(open(os.path.join(output_dir, f)))["kg_co2_emissions"]
            ]
        )
        / len(filtered_files),
    }

    # Save to language-specific average file
    output_file = os.path.join(output_dir, config["average_filename"])
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Averaged {len(filtered_files)} files from {language_code.upper()} retrieval tasks and saved to {output_file}"
    )
    print(f"Tasks included: {[f.replace('.json', '') for f in filtered_files]}")
    print(f"Main score: {averaged_metrics['main_score']:.4f}")


def run_multilingual_evaluation(model_name, context_length=256, output_base_dir="results"):
    """Run evaluation for all languages and calculate averages"""
    # Load the model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = context_length
    # model.max_seq_length = context_length
    model.model_card_data.model_name = model_name

    # Run evaluations for each language
    for language_code, config in LANGUAGE_CONFIGS.items():
        print(f"\n=== Evaluating {language_code.upper()} tasks ===")

        # Create MTEB evaluation for this language
        evaluation = MTEB(tasks=config["benchmark"])

        # Run evaluation
        _ = evaluation.run(
            model,
            output_folder=output_base_dir,
            encode_kwargs={
                "batch_size": 128,  # Adjust batch size as needed
            },
        )

        # Calculate averages for this language
        model_output_dir = os.path.join(output_base_dir, model_name.replace("/", "__"), f"MTEB_{context_length}")
        calculate_language_averages(language_code, model_output_dir)


# Models to evaluate
already_evaluated = [
    # Add models that have already been evaluated to skip them
]

custom_models = [
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
]

model_to_eval = [
    # "models/splade-bert-base-multilingual-uncased-swim-ir-monolingual-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.1/checkpoint-68000"
]

# Main evaluation loop
if __name__ == "__main__":
    for model_name in tqdm(model_to_eval + custom_models):
        for context_length in [512]:  # Can add more context lengths like [256, 512]
            print(f"\n{'=' * 60}")
            print(f"Evaluating model: {model_name}")
            print(f"Context length: {context_length}")
            print(f"{'=' * 60}")

            run_multilingual_evaluation(model_name, context_length)

    print("\n" + "=" * 60)
    print("Multilingual evaluation completed!")
    print("=" * 60)
