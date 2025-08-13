from __future__ import annotations

import argparse
import os
import subprocess

import yaml


def list_checkpoints(base_dir):
    """List all checkpoint directories in the given base directory."""
    if not os.path.exists(base_dir):
        raise ValueError(f"Directory {base_dir} does not exist")

    checkpoints = []
    for item in os.listdir(base_dir):
        if item.startswith("checkpoint-"):
            checkpoint_path = os.path.join(base_dir, item)
            if os.path.isdir(checkpoint_path):
                checkpoints.append(checkpoint_path)

    # Sort checkpoints by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

    # Limit to the last 4 checkpoints
    # checkpoints = checkpoints[-4:]
    # checkpoints = checkpoints[-2:]

    return checkpoints


def create_mergekit_config(checkpoints, output_dir):
    """Create a mergekit configuration for the checkpoints."""
    models = []
    weight = 1.0 / len(checkpoints)

    for i, checkpoint in enumerate(checkpoints):
        models.append({"model": checkpoint, "weight": weight, "density": 0.1})

    config = {
        "merge_method": "dare_ties",
        "models": models,
        "dtype": "float16",
        "parameters": {
            # "lambda": 0.5,  # As in Task Arithmetic
            # "rescale": True,  # For dare_linear, whether to apply DARE rescaling
            "weight": weight,
        },
        "base_model": {"model": checkpoints[-1]},
    }
    """
    for i, checkpoint in enumerate(checkpoints):
        models.append({"model": checkpoint, "weight": weight})

    config = {
        "merge_method": "slerp",
        "base_model": checkpoints[-1],  # Using the last checkpoint as the base
        "models": models,
        "dtype": "float16",
        "parameters": {
            # "lambda": 0.5,  # As in Task Arithmetic
            # "rescale": True,  # For dare_linear, whether to apply DARE rescaling
            # "weight": weight,
            "t": 0.5,  # Interpolation factor for slerp
        },
    }
    """

    config_path = os.path.join(output_dir, "mergekit_config.yml")
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def merge_models(config_path, output_path):
    """Run mergekit to merge the models."""
    cmd = ["mergekit-yaml", config_path, output_path]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Merge multiple sentence-transformer checkpoints using mergekit")
    parser.add_argument(
        "--base_dir",
        default="models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.1",
        help="Directory containing checkpoint folders",
    )
    parser.add_argument("--output_dir", default="merged_model", help="Output directory for the merged model")
    parser.add_argument("--config_dir", default="temp_config", help="Directory to store the mergekit configuration")

    args = parser.parse_args()

    print(f"Scanning for checkpoints in {args.base_dir}...")
    # checkpoints = list_checkpoints(args.base_dir)
    checkpoints = [
        "models/splade-co-condenser-marco-nomic-4-bs_512-lr_2e-05-lq_0.0008-ld_0.002/checkpoint-19190",
        "models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.3/checkpoint-46680",
        "models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.1/checkpoint-70020",
    ]

    if not checkpoints:
        print(f"No checkpoints found in {args.base_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp}")

    print("Creating mergekit configuration...")
    config_path = create_mergekit_config(checkpoints, args.config_dir)

    print("Starting model merge process...")
    merge_models(config_path, args.output_dir)

    print(f"Merge complete. Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
