from __future__ import annotations

import json
import os

# Read all model directories from results folder
results_dir = "results"
md_content = """# MTEB Results

| Model | NDCG@10 (256) |
|-------|---------------|
"""

# Iterate through each model directory
for model_dir in sorted(os.listdir(results_dir)):
    model_path = os.path.join(results_dir, model_dir)
    if os.path.isdir(model_path):
        # Look for MTEB_256/Average.json
        avg_file = os.path.join(model_path, "MTEB_256", "Average.json")

        if os.path.exists(avg_file):
            try:
                with open(avg_file) as f:
                    data = json.load(f)
                    # Extract the main_score (which is NDCG@10)
                    main_score = data["scores"]["test"][0]["main_score"]
                    # Convert to percentage and format
                    score_percent = main_score * 100

                    # Clean up model name (remove prefixes)
                    clean_name = (
                        model_dir.replace("models__", "")
                        .replace("prithivida__", "")
                        .replace("naver__", "")
                        .replace("opensearch-project__", "")
                        .replace("ibm-granite__", "")
                        .replace("sentence-transformers__", "")
                    )

                    md_content += f"| {clean_name} | {score_percent:.2f} |\n"
            except (json.JSONDecodeError, KeyError, IndexError):
                # Skip files with errors
                continue

# Write to markdown file
with open("mteb_results.md", "w") as f:
    f.write(md_content)

print("Generated mteb_results.md from results folder")
