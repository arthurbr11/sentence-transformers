#!/usr/bin/env python3
"""
Preprocessing script for multilingual datasets.
Converts data from the specified format to HuggingFace datasets organized by language.
Only extracts the 'positive' column and removes duplicates.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import Dataset, load_from_disk
from tqdm import tqdm


def extract_positive_texts(dataset) -> list[str]:
    """Extract only the positive texts from a dataset."""
    texts = []

    for sample in tqdm(dataset, desc="Extracting positive texts"):
        if "positive" in sample and sample["positive"]:
            positive_text = sample["positive"].strip()
            if positive_text:  # Skip empty texts
                texts.append(positive_text)

    return texts


def deduplicate_texts(texts: list[str]) -> list[str]:
    """Remove duplicate texts while preserving order."""
    seen = set()
    unique_texts = []

    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    return unique_texts


def process_language_dataset(dataset_path: str, language: str, output_base: str) -> int:
    """Process a single language dataset and save as HF dataset."""
    print(f"Processing {language} dataset: {dataset_path}")

    dataset = load_from_disk(dataset_path)["train"]

    # Extract positive texts
    texts = extract_positive_texts(dataset)
    print(f"Extracted {len(texts)} positive texts")

    # Remove duplicates
    unique_texts = deduplicate_texts(texts)
    print(f"After deduplication: {len(unique_texts)} unique texts")

    if not unique_texts:
        print(f"No texts found for {language}, skipping...")
        return 0

    # Create HF dataset with text column
    hf_dataset = Dataset.from_dict({"text": unique_texts})

    # Save to language-specific directory
    output_dir = os.path.join(output_base, language)
    os.makedirs(output_dir, exist_ok=True)

    hf_dataset.save_to_disk(output_dir)
    print(f"Saved {len(unique_texts)} texts to {output_dir}")

    return len(unique_texts)


def find_train_datasets(base_folder: str) -> dict[str, str]:
    """Find all train dataset files and map them to language names."""
    base_path = Path(base_folder)
    language_files = {}

    # Look for BASE_FOLDER/*/train patterns
    for lang_dir in base_path.iterdir():
        print(f"Checking directory: {lang_dir}")
        if lang_dir.is_dir():
            print(f"Found language directory: {lang_dir.name}")

            language_files[lang_dir.name] = str(lang_dir)

    return language_files


def main():
    parser = argparse.ArgumentParser(description="Preprocess multilingual datasets to HF format by language")
    parser.add_argument(
        "--base_folder",
        type=str,
        required=True,
        help="Base folder containing language subdirectories with train files",
    )
    parser.add_argument(
        "--output_base", type=str, required=True, help="Base output directory where language datasets will be saved"
    )
    parser.add_argument(
        "--languages", type=str, nargs="*", default=None, help="Specific languages to process (default: all)"
    )

    args = parser.parse_args()

    # Find all train datasets
    language_files = find_train_datasets(args.base_folder)
    print(f"Found datasets for languages: {list(language_files.keys())}")

    # Filter by languages if specified
    if args.languages:
        filtered_files = {lang: path for lang, path in language_files.items() if lang in args.languages}
        language_files = filtered_files
        print(f"Filtered to languages: {list(language_files.keys())}")

    if not language_files:
        print("No datasets found to process!")
        return

    # Process each language dataset
    total_texts = 0
    successful_languages = []

    for language, dataset_path in language_files.items():
        try:
            num_texts = process_language_dataset(dataset_path, language, args.output_base)
            if num_texts > 0:
                total_texts += num_texts
                successful_languages.append(language)
        except Exception as e:
            print(f"Error processing {language}: {e}")

    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Successfully processed {len(successful_languages)} languages:")
    for lang in successful_languages:
        print(f"  - {lang}")
    print(f"Total texts across all languages: {total_texts}")
    print(f"Output saved to: {args.output_base}")
    print("\nTo use with training, you can now run:")
    print(f"python train.py --hf_dataset {args.output_base}/<language> --hf_text_field text ...")


if __name__ == "__main__":
    main()
