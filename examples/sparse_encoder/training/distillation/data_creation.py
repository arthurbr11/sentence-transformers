import gzip
import logging
import os
import pickle
import random

import numpy
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from huggingface_hub import hf_hub_download

from sentence_transformers import CrossEncoder

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
random.seed(12)
torch.manual_seed(12)
numpy.random.seed(12)


# open this file in the good dataset format https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/blob/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz


def create_msmarco_cross_scores_dataset():
    """
    Create a dataset for MS MARCO cross-encoder scores.
    This function processes the triplet data and creates a dataset suitable for training.
    """
    # Load the triplet data
    data_triplets = load_dataset("sentence-transformers/msmarco-bm25", split="train", name="triplet-ids")
    # drop the negative_id column
    data_triplets = data_triplets.remove_columns(["negative"])

    # Download the file
    filepath = hf_hub_download(
        repo_id="sentence-transformers/msmarco-hard-negatives",
        filename="cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz",
        repo_type="dataset",
    )

    # Load the file
    with gzip.open(filepath, "rb") as f:
        data_scores = pickle.load(f)

    msarco_corpus = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")
    msarco_query = load_dataset("sentence-transformers/msmarco-corpus", "query", split="train")

    # First, create lookup tables if not already done
    query_lookup = {item["qid"]: item for item in msarco_query}
    corpus_lookup = {item["pid"]: item for item in msarco_corpus}

    def process_function(batch):
        new_batch = {
            "query": [query_lookup[i]["text"] for i in batch["query"]],
            "positive": [corpus_lookup[i]["text"] for i in batch["positive"]],
        }

        for j in range(1, 5):
            new_batch[f"negative_{j}"] = []
        new_batch["label"] = []

        for i, query_id in enumerate(batch["query"]):
            scores = data_scores[query_id]
            pos_score = scores[batch["positive"][i]]
            available = {k: v for k, v in scores.items() if k != batch["positive"][i]}
            negatives = sorted(random.sample(list(available.items()), 4), key=lambda x: x[1], reverse=True)
            for j, (neg_id, _) in enumerate(negatives):
                new_batch[f"negative_{j + 1}"].append(corpus_lookup[neg_id]["text"])
            new_batch["label"].append([pos_score] + [score for _, score in negatives])

        return new_batch

    print("Processing triplets...")
    data_triplets = data_triplets.map(
        process_function,
        batched=True,
        num_proc=1,  # or keep at 8 if speed is more important than UI
        desc="Processing triplets",
    )

    # Save the processed dataset and make a train eval split with 5k eval samples
    data_triplets = data_triplets.train_test_split(test_size=5000, seed=42)
    data_triplets = DatasetDict(
        {
            "train": data_triplets["train"],
            "eval": data_triplets["test"],
        }
    )
    output_dir = "datasets/msmarco-cross-scores-4"
    data_triplets.save_to_disk(output_dir)
    return


def create_nomic_dataset():
    data = load_dataset("nomic-ai/nomic-embed-v2-supervised-data")

    def process_function(batch):
        new_batch = {
            "query": [],
            "positive": [],
            "negative_1": [],
            "negative_2": [],
            "negative_3": [],
            "negative_4": [],
        }
        for query, pos, negs in zip(batch["query"], batch["pos"], batch["neg"]):
            new_batch["query"].append(query)
            new_batch["positive"].append(pos)
            selected_negs = random.sample(negs, 4)
            for j, neg in enumerate(selected_negs, start=1):
                new_batch[f"negative_{j}"].append(neg)
        return new_batch

    print("Processing triplets...")
    data_triplets = data.map(
        process_function,
        batched=True,
        num_proc=1,
        remove_columns=["pos", "neg"],
        desc="Processing triplets",
    )

    output_dir = "datasets/nomic-4"
    os.makedirs(output_dir, exist_ok=True)

    # Save each inner DatasetDict (train/eval for each subset) separately
    for subset_name, subset in data_triplets.items():
        split_subset = subset.train_test_split(test_size=0.01, seed=42)
        split = DatasetDict(
            {
                "train": split_subset["train"],
                "eval": split_subset["test"],
            }
        )
        split.save_to_disk(os.path.join(output_dir, subset_name))

    return output_dir


def re_score():
    # model = SentenceTransformer(
    #     "Qwen/Qwen3-Embedding-0.6B",
    #     # model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "flash_attention_2", "device_map": "auto"},
    #     # tokenizer_kwargs={"padding_side": "left"},
    #     model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
    #     # tokenizer_kwargs={"padding_side": "left"},
    # )
    # cross_encoder = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", activation_fn=torch.nn.Identity())

    # dataset = load_dataset("NetherlandsForensicInstitute/msmarco-nl", split="train")
    # dataset = dataset  # .select(range(10_000))

    # dataset_save_path = f"/fsx/tom_aarsen/st_data/msmarco/"
    # embedding_dataset_save_path = os.path.join(dataset_save_path, "ms-marco-margin-mse-Qwen3-Embedding-0.6B")
    # reranker_dataset_save_path = os.path.join(dataset_save_path, "ms-marco-margin-mse-Qwen3-Reranker-0.6B")

    # mined_dataset = mine_hard_negatives(
    #     dataset,
    #     model,
    #     anchor_column_name="queryen",
    #     positive_column_name="passageen",
    #     # cross_encoder=cross_encoder,
    #     num_negatives=8,
    #     # relative_margin=0.05,
    #     range_max=8,
    #     output_format="n-tuple-scores",
    #     batch_size=256,
    #     use_faiss=True,
    # )
    # mined_dataset.save_to_disk(embedding_dataset_save_path)

    reranker_dataset_save_path = "datasets/msmarco-Qwen3-8B-scores-4"
    dataset = load_from_disk("datasets/msmarco-cross-scores-4")
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    cross_encoder = CrossEncoder(
        "tomaarsen/Qwen3-Reranker-8B-seq-cls",
        model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
        # tokenizer_kwargs={"padding_side": "left"},
        activation_fn=torch.nn.Identity(),
    )

    def rerank(batch):
        queries = batch["query"]
        all_scores = []
        for key in [
            "positive",
            "negative_1",
            "negative_2",
            "negative_3",
            "negative_4",
            # "negative_5",
            # "negative_6",
            # "negative_7",
            # "negative_8",
        ]:
            passages = batch[key]
            pairs = [(query, passage) for query, passage in zip(queries, passages)]
            scores = cross_encoder.predict(pairs, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
            all_scores.append(scores)
        all_scores = torch.stack(all_scores, dim=1)
        batch["label"] = all_scores.tolist()
        return batch

    eval_reranked_dataset = eval_dataset.map(rerank, batched=True, batch_size=1024)
    train_reranked_dataset = train_dataset.map(rerank, batched=True, batch_size=1024)
    reranked_dataset = DatasetDict(
        {
            "train": train_reranked_dataset,
            "eval": eval_reranked_dataset,
        }
    )
    reranked_dataset.save_to_disk(reranker_dataset_save_path)


def rescore_nomic():
    input_dataset_dir = "datasets/nomic-4"
    output_dataset_dir = "datasets/nomic-Qwen3-8B-scores-4"
    os.makedirs(output_dataset_dir, exist_ok=True)

    # SPLITS = os.listdir(input_dataset_dir)  # e.g., ["msmarco", "miracl-en", ...]
    # SPLITS.remove("dataset_dict.json")

    SPLITS = ["msmarco"]  # "nq", "stack", "fever", "hotpot", "squad"]
    dataset_dict = {"splits": SPLITS}
    import json

    with open(os.path.join(output_dataset_dir, "dataset_dict.json"), "w") as f:
        json.dump(dataset_dict, f, indent=4)

    # Load CrossEncoder model
    cross_encoder = CrossEncoder(
        "tomaarsen/Qwen3-Reranker-8B-seq-cls",  # "cross-encoder/ms-marco-MiniLM-L6-v2",  #
        model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
        activation_fn=torch.nn.Identity(),
    )

    def rerank(batch):
        queries = batch["query"]
        all_scores = []

        for key in ["positive", "negative_1", "negative_2", "negative_3", "negative_4"]:
            passages = batch[key]
            pairs = list(zip(queries, passages))
            scores = cross_encoder.predict(
                pairs,
                convert_to_tensor=True,
                batch_size=4,
                show_progress_bar=False,
            )
            all_scores.append(scores)

        all_scores = torch.stack(all_scores, dim=1)
        batch["label"] = all_scores.tolist()
        return batch

    for split in SPLITS:
        print(f"Processing split: {split}")
        subset_path = os.path.join(input_dataset_dir, split)
        subset = DatasetDict.load_from_disk(subset_path)

        train_dataset = subset["train"].map(rerank, batched=True, batch_size=1024)
        eval_dataset = subset["eval"].map(rerank, batched=True, batch_size=1024)

        rescored_subset = DatasetDict(
            {
                "train": train_dataset,
                "eval": eval_dataset,
            }
        )

        output_path = os.path.join(output_dataset_dir, split)
        rescored_subset.save_to_disk(output_path)
        print(f"Saved rescored split to: {output_path}")


if __name__ == "__main__":
    rescore_nomic()
