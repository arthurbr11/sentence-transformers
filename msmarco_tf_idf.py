#!/usr/bin/env python3
"""
Optimized TF-IDF weight generator for MS MARCO corpus with SparseStaticEmbedding.

Key improvements:
- Batch processing for 10-100x speed improvement
- Memory-efficient streaming
- Better progress tracking
- Robust error handling
- Configurable parameters
- Multi-processing support
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from sentence_transformers.sparse_encoder.models.SparseStaticEmbedding import SparseStaticEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TFIDFWeightGenerator:
    """
    Optimized TF-IDF weight generator with batch processing and memory efficiency.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 10000,
        num_workers: int = None,
        cache_dir: str = "./cache",
        min_doc_freq: int = 2,
        max_doc_freq_ratio: float = 0.95,
    ):
        """
        Initialize the TF-IDF weight generator.

        Args:
            model_name: HuggingFace model name for tokenizer
            batch_size: Number of documents to process in each batch
            num_workers: Number of CPU cores to use (None = auto-detect)
            cache_dir: Directory to cache intermediate results
            min_doc_freq: Minimum document frequency for a token to be included
            max_doc_freq_ratio: Maximum document frequency ratio (to filter stop words)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers or min(8, mp.cpu_count())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.min_doc_freq = min_doc_freq
        self.max_doc_freq_ratio = max_doc_freq_ratio

        logger.info(f"Initializing with model: {model_name}")
        logger.info(f"Batch size: {batch_size}, Workers: {self.num_workers}")

        # Load tokenizer once
        self.tokenizer = self._load_tokenizer()
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)

        # Create reverse mapping for efficient lookups
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer with error handling."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            logger.info(f"Loaded tokenizer with vocab size: {len(tokenizer.get_vocab())}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _batch_tokenize(self, texts: list[str]) -> list[set]:
        """
        Fast batch tokenization with error handling.

        Args:
            texts: list of text documents

        Returns:
            list of sets containing unique token IDs for each document
        """
        try:
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                return []

            # Batch tokenization is much faster than individual tokenization
            batch_encoding = self.tokenizer(
                valid_texts,
                padding=False,
                truncation=True,
                max_length=512,  # Reasonable limit for efficiency
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,  # Skip special tokens for TF-IDF
            )

            # Convert to sets of unique token IDs for each document
            return [set(input_ids) for input_ids in batch_encoding["input_ids"]]

        except Exception as e:
            logger.warning(f"Tokenization error for batch: {e}")
            return []

    def _process_batch(self, texts: list[str]) -> Counter:
        """
        Process a batch of texts and return document frequencies.

        Args:
            texts: Batch of text documents

        Returns:
            Counter with token_id -> document_frequency mapping
        """
        token_sets = self._batch_tokenize(texts)
        doc_freq = Counter()

        for token_set in token_sets:
            for token_id in token_set:
                doc_freq[token_id] += 1

        return doc_freq

    def _load_dataset_streaming(self, max_docs: int = None):
        """
        Load dataset in streaming mode for memory efficiency.

        Args:
            max_docs: Maximum number of documents to process

        Yields:
            Batches of text documents
        """
        try:
            logger.info("Loading MS MARCO dataset...")
            dataset = load_dataset(
                "sentence-transformers/msmarco-corpus", "passage", streaming=True if max_docs is None else False
            )

            if max_docs is None:
                texts = dataset["train"]
                total_docs = "unknown (streaming)"
            else:
                texts = dataset["train"]["text"][:max_docs]
                total_docs = min(max_docs, len(texts))

            logger.info(f"Processing {total_docs} documents")

            batch = []
            processed = 0

            for item in texts:
                text = item["text"] if isinstance(item, dict) else item

                if text and text.strip():
                    batch.append(text)

                if len(batch) >= self.batch_size:
                    yield batch
                    processed += len(batch)
                    batch = []

                    if max_docs and processed >= max_docs:
                        break

            # Yield remaining documents
            if batch:
                yield batch

        except Exception as e:
            logger.error(f"Dataset loading error: {e}")
            raise

    def compute_document_frequencies(self, max_docs: int = None) -> tuple[Counter, int]:
        """
        Compute document frequencies for all tokens in the corpus.

        Args:
            max_docs: Maximum number of documents to process

        Returns:
            Tuple of (document_frequencies, total_documents)
        """
        cache_file = self.cache_dir / f"doc_freq_{self.model_name.replace('/', '_')}_{max_docs or 'full'}.json"

        # Try loading from cache
        if cache_file.exists():
            logger.info(f"Loading document frequencies from cache: {cache_file}")
            with open(cache_file) as f:
                cache_data = json.load(f)
                doc_freq = Counter({int(k): v for k, v in cache_data["doc_frequencies"].items()})
                return doc_freq, cache_data["total_docs"]

        logger.info("Computing document frequencies...")
        total_doc_freq = Counter()
        total_docs = 0

        # Process in batches with progress bar
        batch_generator = self._load_dataset_streaming(max_docs)

        with tqdm(desc="Processing batches", unit="batch") as pbar:
            if self.num_workers > 1:
                # Multi-processing for faster computation
                with mp.Pool(self.num_workers) as pool:
                    for batch in batch_generator:
                        # Split batch among workers
                        chunk_size = max(1, len(batch) // self.num_workers)
                        chunks = [batch[i : i + chunk_size] for i in range(0, len(batch), chunk_size)]

                        # Process chunks in parallel
                        batch_results = pool.map(self._process_batch, chunks)

                        # Merge results
                        for result in batch_results:
                            total_doc_freq += result

                        total_docs += len(batch)
                        pbar.update(1)
                        pbar.set_postfix({"docs": total_docs, "unique_tokens": len(total_doc_freq)})
            else:
                # Single-threaded processing
                for batch in batch_generator:
                    batch_freq = self._process_batch(batch)
                    total_doc_freq += batch_freq
                    total_docs += len(batch)
                    pbar.update(1)
                    pbar.set_postfix({"docs": total_docs, "unique_tokens": len(total_doc_freq)})

        logger.info(f"Processed {total_docs} documents")
        logger.info(f"Found {len(total_doc_freq)} unique tokens")

        # Cache results
        cache_data = {
            "doc_frequencies": {str(k): v for k, v in total_doc_freq.items()},
            "total_docs": total_docs,
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Cached document frequencies to: {cache_file}")

        return total_doc_freq, total_docs

    def create_idf_weights(
        self, doc_frequencies: Counter, total_docs: int, smoothing: float = 1.0, ones_unseen: bool = True
    ) -> torch.Tensor:
        """
        Create IDF weight tensor from document frequencies.

        Args:
            doc_frequencies: Token document frequencies
            total_docs: Total number of documents
            smoothing: Smoothing factor for IDF calculation
            ones_unseen: If True, set unseen tokens to 1 (recommended)

        Returns:
            IDF weights tensor
        """
        logger.info("Creating IDF weight tensor...")

        # Filter tokens by frequency (only tokens actually seen in corpus)
        max_doc_freq = int(total_docs * self.max_doc_freq_ratio)
        valid_freq = {
            token_id: freq
            for token_id, freq in doc_frequencies.items()
            if self.min_doc_freq <= freq <= max_doc_freq and token_id < self.vocab_size
        }

        logger.info(f"Valid tokens after filtering: {len(valid_freq)} (from {len(doc_frequencies)} seen)")

        if ones_unseen:
            # Initialize all weights to 1 (unseen tokens stay 1)
            weights = torch.ones(self.vocab_size, dtype=torch.float32)
            logger.info("Unseen tokens will have weight 1 (recommended)")
        else:
            # Initialize with small default weight for unseen tokens
            default_idf = math.log((total_docs + smoothing) / (smoothing * 10))  # Lower default
            weights = torch.full((self.vocab_size,), default_idf, dtype=torch.float32)
            logger.info(f"Unseen tokens will have weight {default_idf:.4f}")

        # Compute IDF weights for valid tokens only
        computed_weights = 0
        idf_values = []

        for token_id, freq in valid_freq.items():
            idf = math.log((total_docs + smoothing) / (freq + smoothing))
            weights[token_id] = idf
            idf_values.append(idf)
            computed_weights += 1

        # Statistics
        unseen_tokens = self.vocab_size - len(doc_frequencies)
        filtered_out = len(doc_frequencies) - len(valid_freq)

        logger.info("Token statistics:")
        logger.info(f"  - Computed IDF: {computed_weights} tokens")
        logger.info(f"  - Filtered out (too rare/common): {filtered_out} tokens")
        logger.info(f"  - Never seen in corpus: {unseen_tokens} tokens")

        if idf_values:
            idf_array = np.array(idf_values)
            logger.info(
                f"IDF stats (computed only) - min: {idf_array.min():.4f}, "
                f"max: {idf_array.max():.4f}, mean: {idf_array.mean():.4f}"
            )

        logger.info(
            f"Final weight stats - min: {weights.min().item():.4f}, "
            f"max: {weights.max().item():.4f}, "
            f"mean: {weights.mean().item():.4f}"
        )

        return weights

    def generate_weights(
        self,
        max_docs: int = None,
        save_path: str = None,
        smoothing: float = 1.0,
        ones_unseen: bool = True,
    ) -> tuple[torch.Tensor, PreTrainedTokenizer]:
        """
        Complete pipeline to generate TF-IDF weights.

        Args:
            max_docs: Maximum documents to process
            save_path: Path to save the weights
            smoothing: IDF smoothing factor
            ones_unseen: Set unseen tokens to 1 weight (recommended)

        Returns:
            Tuple of (weights tensor, tokenizer)
        """
        # Compute document frequencies
        doc_freq, total_docs = self.compute_document_frequencies(max_docs)

        # Create IDF weights
        weights = self.create_idf_weights(doc_freq, total_docs, smoothing, ones_unseen)

        # Save weights if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "weights": weights,
                    "model_name": self.model_name,
                    "total_docs": total_docs,
                    "vocab_size": self.vocab_size,
                    "config": {
                        "min_doc_freq": self.min_doc_freq,
                        "max_doc_freq_ratio": self.max_doc_freq_ratio,
                        "smoothing": smoothing,
                        "zero_unseen": ones_unseen,
                    },
                },
                save_path,
            )
            logger.info(f"Saved weights to: {save_path}")

        return weights, self.tokenizer


def demo_sparse_embedding(weights: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """Enhanced demo with better statistics and error handling."""
    logger.info("Creating SparseStaticEmbedding demo...")

    try:
        sparse_embedding = SparseStaticEmbedding(tokenizer=tokenizer, weight=weights, frozen=False)

        test_texts = [
            "What is machine learning and artificial intelligence?",
            "How does a neural network architecture work in deep learning?",
            "Information retrieval systems using sparse embeddings and TF-IDF",
            "Microsoft Research MS MARCO dataset for passage ranking",
            "Natural language processing with transformer models",
        ]

        logger.info(f"Testing with {len(test_texts)} examples...")

        features = sparse_embedding.tokenize(test_texts)
        output = sparse_embedding(features)
        embeddings = output["sentence_embedding"]

        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Overall sparsity: {(embeddings == 0).float().mean().item():.2%}")

        print("\n📊 Detailed Statistics:")
        print("-" * 80)
        for i, text in enumerate(test_texts):
            emb = embeddings[i]
            non_zero = (emb != 0).sum().item()
            sparsity = (emb == 0).float().mean().item()
            max_val = emb.max().item()
            mean_val = emb[emb != 0].mean().item() if non_zero > 0 else 0

            print(f"Text {i + 1:2d}: {non_zero:4d} active dims ({sparsity:.1%} sparse)")
            print(f"         Max: {max_val:6.4f}, Mean: {mean_val:6.4f}")
            print(f"         '{text[:60]}{'...' if len(text) > 60 else ''}'")
            print()

        return sparse_embedding

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


def main():
    """Main execution with configurable parameters."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate TF-IDF weights for MS MARCO")
    parser.add_argument("--model", default="bert-base-uncased", help="Model name")
    parser.add_argument("--max-docs", type=int, help="Max documents to process")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size")
    parser.add_argument("--workers", type=int, help="Number of workers")
    parser.add_argument("--output", default="msmarco_idf_weights.pt", help="Output file")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    parser.add_argument("--keep-unseen", action="store_true", help="Keep unseen tokens with small weights")

    args = parser.parse_args()

    logger.info("🚀 Starting optimized TF-IDF weight generation...")

    # Create generator
    generator = TFIDFWeightGenerator(
        model_name=args.model, batch_size=args.batch_size, num_workers=args.workers, cache_dir=args.cache_dir
    )

    # Generate weights
    weights, tokenizer = generator.generate_weights(
        max_docs=args.max_docs,
        save_path=args.output,
        zero_unseen=not args.keep_unseen,  # Default: ones_unseen tokens
    )

    if args.demo:
        _ = demo_sparse_embedding(weights, tokenizer)

    logger.info("✅ Generation complete!")
    print("\n📝 To load weights later:")
    print(f"   data = torch.load('{args.output}')")
    print("   weights = data['weights']")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.model}')")
    print("   model = SparseStaticEmbedding(tokenizer=tokenizer, weight=weights)")


if __name__ == "__main__":
    # main()
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    data = torch.load("msmarco_idf_weights_distilbert.pt")
    weights = data["weights"]
    # filter all the scores above or equal  15.9950
    # weights = weights[weights < 15.9950]
    # tokenizer = AutoTokenizer.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill")
    # sparse_embedding = SparseStaticEmbedding.from_json(
    #     tokenizer=tokenizer, json_path="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"
    # )
    # weights = sparse_embedding.weight.detach().cpu()

    # print stat of weights
    print(
        f"Weight stats - min: {weights.min().item():.4f}, "
        f"max: {weights.max().item():.4f}, "
        f"mean: {weights.mean().item():.4f}, "
        f"std: {weights.std().item():.4f}, "
        f"non-zero: {(weights != 0).sum().item()} out of {len(weights)}"
    )
    # plot the weights
    import matplotlib.pyplot as plt

    plt.hist(weights.numpy(), bins=100, alpha=0.75)
    plt.title("TF-IDF Weights Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    plt.savefig("tfidf_weights_distribution_distil.png")

    #  print the top 100 weights and bottom 100 weights
    top_weights = torch.topk(weights, 100)
    bottom_weights = torch.topk(-weights, 100)
    print("\nTop 100 Weights:")
    for i in range(100):
        token = tokenizer.convert_ids_to_tokens(top_weights.indices[i].item())
        print(f"{i + 1:2d}: {token} - {top_weights.values[i].item():.4f}")
    print("\nBottom 100 Weights:")
    for i in range(100):
        token = tokenizer.convert_ids_to_tokens(bottom_weights.indices[i].item())
        print(f"{i + 1:2d}: {token} - {-bottom_weights.values[i].item():.4f}")
