import random

from datasets import load_dataset

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator import (
    SparseInformationRetrievalEvaluator,
)
from sentence_transformers.sparse_encoder.models import CSRSparsity

# Initialize model components
model_name = "sentence-transformers/all-mpnet-base-v2"
transformer = Transformer(model_name)
pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
csr_sparsity = CSRSparsity(
    input_dim=transformer.get_word_embedding_dimension(),
    hidden_dim=4 * transformer.get_word_embedding_dimension(),
    k=32,  # Number of top values to keep
    k_aux=512,  # Number of top values for auxiliary loss
)
# Create the SparseEncoder model
model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])

# Load the Touche-2020 IR dataset (https://huggingface.co/datasets/BeIR/webis-touche2020, https://huggingface.co/datasets/BeIR/webis-touche2020-qrels)
corpus = load_dataset("BeIR/webis-touche2020", "corpus", split="corpus")
queries = load_dataset("BeIR/webis-touche2020", "queries", split="queries")
relevant_docs_data = load_dataset("BeIR/webis-touche2020-qrels", split="test")

# For this dataset, we want to concatenate the title and texts for the corpus
corpus = corpus.map(lambda x: {"text": x["title"] + " " + x["text"]}, remove_columns=["title"])

# Shrink the corpus size heavily to only the relevant documents + 30,000 random documents
required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))
required_corpus_ids |= set(random.sample(corpus["_id"], k=30_000))
corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)

# Convert the datasets to dictionaries
corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_ids)

# Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.
ir_evaluator = SparseInformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="BeIR-touche2020-subset-test",
    show_progress_bar=True,
)
results = ir_evaluator(model)
"""
# Print the results
"""
print(ir_evaluator.primary_metric)
print(results[ir_evaluator.primary_metric])
