import numpy as np

from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling


def main():
    # Initialize the SPLADE model
    model_name = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"  # "prithivida/Splade_PP_en_v1"  # "naver/splade-cocondenser-ensembledistil"
    model = SparseEncoder(
        modules=[
            MLMTransformer(model_name),
            SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
        ],
        device="cpu",
    )

    # Sample texts
    texts = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_sparse_tensor=True)
    print(type(embeddings))

    # Print embedding shape and sparsity
    print(f"Embedding shape: {embeddings.shape}")

    print(f"Embedding sparsity: {model.get_sparsity_stats(embeddings)}%")

    # Compute similarity matrix
    similarity_matrix = model.similarity(embeddings, embeddings)

    # Print similarity matrix
    print("\nSimilarity Matrix:")
    for i, text in enumerate(texts):
        print(f"{i}: {text[:50]}...")
    print("\n" + " " * 10 + " ".join([f"{i:5d}" for i in range(len(texts))]))
    for i, row in enumerate(similarity_matrix):
        print(f"{i:5d}     " + " ".join([f"{val:.3f}" for val in row]))
    vocab_size = embeddings.shape[1]
    print(f"Vocabulary size: {vocab_size}")

    # Visualize top tokens for each text
    top_k = 20

    print(f"\nTop tokens {top_k} for each text:")

    for i, text in enumerate(texts):
        # Get top k indices in sparse tensor
        # Get top k indices in sparse tensor (sorted from highest to lowest)
        top_indices = np.argsort(-embeddings[i].to_dense().numpy())[:top_k]
        top_values = embeddings[i].to_dense().numpy()[top_indices]
        top_tokens = [model.tokenizer.decode([idx]) for idx in top_indices]
        print(f"{i}: {text}")
        print(f"Top tokens: {top_tokens}")
        print(f"Top values: {top_values}")
        print()


if __name__ == "__main__":
    main()
