import json
import os
from collections import defaultdict

from mteb import MTEB
from mteb.benchmarks.benchmark import Benchmark
from mteb.overview import MTEBTasks, get_tasks
from tqdm import tqdm

from sentence_transformers import SparseEncoder

MTEB_EN_V2_RETRIEVAL_TASKS = Benchmark(
    name="MTEB_RETRIEVAL(eng, v2)",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
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
            ],
            languages=["eng"],
            exclusive_language_filter=True,
        )
    ),
)


def calculate_averages(output_dir="results"):
    # Define the tasks from MTEB_EN_V2_RETRIEVAL_TASKS benchmark
    benchmark_tasks = {
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
    }

    # Get all JSON files in current directory
    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    # Skip existing Average files and model_meta.json
    json_files = [f for f in json_files if f != "Average_EN_V2_RETRIEVAL_TASKS.json" and f != "model_meta.json"]

    # Filter files to only include those corresponding to benchmark tasks
    filtered_files = []
    for filename in json_files:
        # Extract task name from filename (assuming format like "TaskName.json")
        task_name = filename.replace(".json", "")
        if task_name in benchmark_tasks:
            filtered_files.append(filename)

    if not filtered_files:
        print("No files found matching the MTEB_EN_V2_RETRIEVAL_TASKS benchmark tasks.")
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

        # Extract all numeric metrics from test scores
        test_scores = data["scores"]["test"][0]
        for key, value in test_scores.items():
            if isinstance(value, (int, float)):
                all_metrics[key].append(value)

    # Calculate averages
    averaged_metrics = {}
    for key, values in all_metrics.items():
        averaged_metrics[key] = sum(values) / len(values)

    # Keep non-numeric fields from first file
    for key, value in first_file_data["scores"]["test"][0].items():
        if not isinstance(value, (int, float)):
            averaged_metrics[key] = value

    # Create result structure
    result = {
        "dataset_revision": first_file_data["dataset_revision"],
        "task_name": "MTEB_EN_V2_RETRIEVAL_TASKS_Average",
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

    # Save to Average_EN_V2_RETRIEVAL_TASKS.json
    output_file = os.path.join(output_dir, "Average_EN_V2_RETRIEVAL_TASKS.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Averaged {len(filtered_files)} files from MTEB_EN_V2_RETRIEVAL_TASKS benchmark and saved to {output_file}")
    print(f"Tasks included: {[f.replace('.json', '') for f in filtered_files]}")
    print(f"Main score: {averaged_metrics['main_score']:.4f}")


already_evaluated = [
    # "naver/splade-cocondenser-ensembledistil",
    # "naver/splade-v3",
    # "naver/splade-v3-distilbert",
    # "prithivida/Splade_PP_en_v1",
    # "prithivida/Splade_PP_en_v2",
    # "ibm-granite/granite-embedding-30m-sparse",
    # "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
    # "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
    # "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    # "opensearch-project/opensearch-neural-sparse-encoding-v2-distill",
    # "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
    # "models/merged_model-2",
    # "models/splade-ettin-encoder-150m-msmarco-Qwen3-8B-scores-4-bs_128-lr_8e-05-lq_0.3-ld_0.25/checkpoint-77800",
    # "models/splade-gte-en-mlm-base-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.1/checkpoint-108920",
    # "models/splade-co-condenser-marco-msmarco-cross_scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.08/checkpoint-105030",
    # "models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.1/checkpoint-70020",
    # "models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.3/checkpoint-46680",
    # "models/merged_model-1",
    # "models/splade-ettin-encoder-150m-msmarco-Qwen3-8B-scores-4-bs_128-lr_8e-05-lq_0.1-ld_0.1/checkpoint-58350",
    # "models/splade-ettin-encoder-150m-msmarco-Qwen3-8B-scores-4-bs_128-lr_8e-05-lq_0.1-ld_0.1/checkpoint-85580",
    # "models/merged_model-3",
]
model_to_eval = [
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1",
]

# Add custom models
custom_models = [
    # "models/splade-bert-base-multilingual-uncased-swim-ir-monolingual-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.1/checkpoint-68000"
]

for model_name in tqdm(model_to_eval + custom_models):
    for context_length in [512]:  # , 512]:
        # Load a model
        model = SparseEncoder(model_name, trust_remote_code=True)
        model.max_seq_length = context_length  # Set the max sequence length for the evaluation
        model.model_card_data.model_name = model_name
        # Create MTEB evaluation with device specification
        evaluation = MTEB(tasks=MTEB_EN_V2_RETRIEVAL_TASKS)

        _ = evaluation.run(
            model,
            output_folder="results",
            encode_kwargs={
                "batch_size": 128,  # Adjust batch size as needed
            },
        )
        calculate_averages("results/" + model_name.replace("/", "__") + "/MTEB_" + str(context_length))
