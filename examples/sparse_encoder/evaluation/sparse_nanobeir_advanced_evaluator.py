import logging
import os

from tqdm import tqdm

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)
model_to_eval = [
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
]

# Add custom models
custom_models = [
    #     "models/splade-co-condenser-marco-msmarco-cross_scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.08/checkpoint-105030",
    #     "models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.1/checkpoint-70020",
    #     "models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.3/checkpoint-46680",
    #     "models/merged_model-1",
    #     "models/merged_model-2",
    "models/splade-ettin-encoder-150m-msmarco-Qwen3-8B-scores-4-bs_128-lr_8e-05-lq_0.1-ld_0.1/checkpoint-58350",
    "models/splade-ettin-encoder-150m-msmarco-Qwen3-8B-scores-4-bs_128-lr_8e-05-lq_0.1-ld_0.1/checkpoint-85580",
]

for model_name in tqdm(model_to_eval + custom_models):
    for context_length in [256, 512]:
        path = f"results/{model_name.replace('/', '__')}_trunc_d256_q128/NanoBEIR_{context_length}"
        if os.path.exists(path):
            print(f"Results already exist for {model_name} with context length {context_length}. Skipping...")
            continue
        # Load a model
        model = SparseEncoder(model_name, trust_remote_code=True)
        model.max_seq_length = context_length  # Set the max sequence length for the evaluation

        evaluator = SparseNanoBEIREvaluator(
            dataset_names=None,  # None means evaluate on all datasets
            show_progress_bar=True,
            batch_size=32,
            max_active_dims=256,
        )

        os.makedirs(path, exist_ok=True)
        # Run evaluation
        results = evaluator(model, output_path=path)
"""
----------------------------------------------- naver/splade-cocondenser-ensembledistil_512
Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 59.18%
Accuracy@3: 75.37%
Accuracy@5: 80.76%
Accuracy@10: 86.92%
Precision@1: 59.18%
Recall@1: 35.62%
Precision@3: 36.26%
Recall@3: 50.85%
Precision@5: 27.75%
Recall@5: 56.57%
Precision@10: 19.24%
Recall@10: 64.31%
MRR@10: 0.6848
NDCG@10: 0.6218
Model Query Sparsity: Active Dimensions: 72.7, Sparsity Ratio: 0.9976
Model Corpus Sparsity: Active Dimensions: 165.9, Sparsity Ratio: 0.9946

----------------------------------------------- naver/splade-cocondenser-ensembledistil_256
Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 58.72%
Accuracy@3: 75.37%
Accuracy@5: 80.92%
Accuracy@10: 87.54%
Precision@1: 58.72%
Recall@1: 35.01%
Precision@3: 35.74%
Recall@3: 50.49%
Precision@5: 27.72%
Recall@5: 56.43%
Precision@10: 19.28%
Recall@10: 64.27%
MRR@10: 0.6832
NDCG@10: 0.6182
MAP@100: 0.5372
Model Query Sparsity: Active Dimensions: 72.0, Sparsity Ratio: 0.9976
Model Corpus Sparsity: Active Dimensions: 157.8, Sparsity Ratio: 0.9948

----------------------------------------------- prithivida/Splade_PP_en_v2_512
Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 58.26%
Accuracy@3: 75.07%
Accuracy@5: 80.15%
Accuracy@10: 86.15%
Precision@1: 58.26%
Recall@1: 34.62%
Precision@3: 35.85%
Recall@3: 50.24%
Precision@5: 27.51%
Recall@5: 55.95%
Precision@10: 19.19%
Recall@10: 63.37%
MRR@10: 0.6784
NDCG@10: 0.6137
MAP@100: 0.5356
Model Query Sparsity: Active Dimensions: 65.4, Sparsity Ratio: 0.9979
Model Corpus Sparsity: Active Dimensions: 153.2, Sparsity Ratio: 0.9950
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6137

----------------------------------------------- naver/splade-v3_256
Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 59.20%
Accuracy@3: 76.61%
Accuracy@5: 82.46%
Accuracy@10: 88.15%
Precision@1: 59.20%
Recall@1: 35.01%
Precision@3: 36.93%
Recall@3: 51.34%
Precision@5: 28.47%
Recall@5: 57.30%
Precision@10: 19.74%
Recall@10: 64.98%
MRR@10: 0.6924
NDCG@10: 0.6272
MAP@100: 0.5449
Model Query Sparsity: Active Dimensions: 74.7, Sparsity Ratio: 0.9976
Model Corpus Sparsity: Active Dimensions: 211.6, Sparsity Ratio: 0.9931

----------------------------------------------- naver/splade-v3_512
Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 60.59%
Accuracy@3: 77.07%
Accuracy@5: 82.15%
Accuracy@10: 88.92%
Precision@1: 60.59%
Recall@1: 35.39%
Precision@3: 37.55%
Recall@3: 51.89%
Precision@5: 28.53%
Recall@5: 57.42%
Precision@10: 19.88%
Recall@10: 65.66%
MRR@10: 0.7017
NDCG@10: 0.6337
MAP@100: 0.5502
Model Query Sparsity: Active Dimensions: 75.7, Sparsity Ratio: 0.9975
Model Corpus Sparsity: Active Dimensions: 223.9, Sparsity Ratio: 0.9927

-----------------------------------------------opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte_256 IF

Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 60.72%
Accuracy@3: 75.99%
Accuracy@5: 81.99%
Accuracy@10: 88.15%
Precision@1: 60.72%
Recall@1: 36.09%
Precision@3: 37.39%
Recall@3: 52.15%
Precision@5: 28.80%
Recall@5: 58.32%
Precision@10: 19.99%
Recall@10: 65.64%
MRR@10: 0.7004
NDCG@10: 0.6383
MAP@100: 0.5568
Model Query Sparsity: Active Dimensions: 21.5, Sparsity Ratio: 0.9993
Model Corpus Sparsity: Active Dimensions: 211.6, Sparsity Ratio: 0.9931
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6383

------------------------------------------------ opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill_256 IF
Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 57.79%
Accuracy@3: 74.61%
Accuracy@5: 79.38%
Accuracy@10: 86.46%
Precision@1: 57.79%
Recall@1: 34.49%
Precision@3: 36.21%
Recall@3: 51.13%
Precision@5: 27.42%
Recall@5: 56.32%
Precision@10: 19.09%
Recall@10: 63.78%
MRR@10: 0.6745
NDCG@10: 0.6142
MAP@100: 0.5365
Model Query Sparsity: Active Dimensions: 21.5, Sparsity Ratio: 0.9993
Model Corpus Sparsity: Active Dimensions: 214.9, Sparsity Ratio: 0.9930
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6142

------------------------------------------------ opensearch-neural-sparse-encoding-v2-distill_256

Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 59.95%
Accuracy@3: 75.22%
Accuracy@5: 81.99%
Accuracy@10: 88.30%
Precision@1: 59.95%
Recall@1: 35.36%
Precision@3: 36.05%
Recall@3: 50.51%
Precision@5: 28.12%
Recall@5: 57.34%
Precision@10: 19.06%
Recall@10: 64.48%
MRR@10: 0.6934
NDCG@10: 0.6218
MAP@100: 0.5399
Model Query Sparsity: Active Dimensions: 141.2, Sparsity Ratio: 0.9954
Model Corpus Sparsity: Active Dimensions: 219.0, Sparsity Ratio: 0.9928
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6218

------------------------------------------------ ibm-granite/granite-embedding-30m-sparse_256

Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 57.18%
Accuracy@3: 76.14%
Accuracy@5: 81.99%
Accuracy@10: 88.00%
Precision@1: 57.18%
Recall@1: 33.93%
Precision@3: 35.78%
Recall@3: 51.29%
Precision@5: 27.07%
Recall@5: 57.61%
Precision@10: 18.45%
Recall@10: 64.68%
MRR@10: 0.6784
NDCG@10: 0.6113
MAP@100: 0.5330
Model Query Sparsity: Active Dimensions: 83.0, Sparsity Ratio: 0.9983
Model Corpus Sparsity: Active Dimensions: 217.6, Sparsity Ratio: 0.9957
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6113

----------------------------------------------- models/splade-co-condenser-marco-msmarco-cross_scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.08/checkpoint-105030_256

Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 57.95%
Accuracy@3: 75.84%
Accuracy@5: 81.69%
Accuracy@10: 86.92%
Precision@1: 57.95%
Recall@1: 34.20%
Precision@3: 36.00%
Recall@3: 50.31%
Precision@5: 27.94%
Recall@5: 56.92%
Precision@10: 19.32%
Recall@10: 63.63%
MRR@10: 0.6792
NDCG@10: 0.6140
MAP@100: 0.5330
Model Query Sparsity: Active Dimensions: 84.7, Sparsity Ratio: 0.9972
Model Corpus Sparsity: Active Dimensions: 247.7, Sparsity Ratio: 0.9919
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6140

----------------------------------------------- models/splade-co-condenser-marco-msmarco-cross_scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.08/checkpoint-105030_512

Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 59.34%
Accuracy@3: 76.30%
Accuracy@5: 80.92%
Accuracy@10: 87.23%
Precision@1: 59.34%
Recall@1: 34.88%
Precision@3: 36.52%
Recall@3: 51.26%
Precision@5: 28.22%
Recall@5: 56.96%
Precision@10: 19.56%
Recall@10: 64.36%
MRR@10: 0.6873
NDCG@10: 0.6225
MAP@100: 0.5410
Model Query Sparsity: Active Dimensions: 85.4, Sparsity Ratio: 0.9972
Model Corpus Sparsity: Active Dimensions: 256.4, Sparsity Ratio: 0.9916
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6225

--------------------------------- models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.1/checkpoint-70020_256
Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 60.26%
Accuracy@3: 76.61%
Accuracy@5: 80.61%
Accuracy@10: 87.38%
Precision@1: 60.26%
Recall@1: 35.71%
Precision@3: 36.42%
Recall@3: 50.63%
Precision@5: 27.92%
Recall@5: 56.01%
Precision@10: 19.77%
Recall@10: 64.48%
MRR@10: 0.6939
NDCG@10: 0.6263
MAP@100: 0.5447
Model Query Sparsity: Active Dimensions: 91.0, Sparsity Ratio: 0.9970
Model Corpus Sparsity: Active Dimensions: 245.0, Sparsity Ratio: 0.9920
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6263

--------------------------------- models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.1/checkpoint-70020_512

Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 60.26%
Accuracy@3: 76.30%
Accuracy@5: 80.61%
Accuracy@10: 87.38%
Precision@1: 60.26%
Recall@1: 36.31%
Precision@3: 36.41%
Recall@3: 50.95%
Precision@5: 27.88%
Recall@5: 56.31%
Precision@10: 19.69%
Recall@10: 64.52%
MRR@10: 0.6930
NDCG@10: 0.6284
MAP@100: 0.5512
Model Query Sparsity: Active Dimensions: 91.8, Sparsity Ratio: 0.9970
Model Corpus Sparsity: Active Dimensions: 253.6, Sparsity Ratio: 0.9917
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6284

---------------------------------- models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.3/checkpoint-46680_512

Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 60.57%
Accuracy@3: 75.22%
Accuracy@5: 80.76%
Accuracy@10: 87.69%
Precision@1: 60.57%
Recall@1: 36.19%
Precision@3: 35.84%
Recall@3: 50.26%
Precision@5: 27.41%
Recall@5: 56.32%
Precision@10: 19.43%
Recall@10: 64.63%
MRR@10: 0.6932
NDCG@10: 0.6249
MAP@100: 0.5459
Model Query Sparsity: Active Dimensions: 80.5, Sparsity Ratio: 0.9974
Model Corpus Sparsity: Active Dimensions: 192.4, Sparsity Ratio: 0.9937
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6249

---------------------------------- models/splade-co-condenser-marco-msmarco-Qwen3-8B-scores-4-bs_128-lr_2e-05-lq_0.1-ld_0.3/checkpoint-46680_256

Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 60.41%
Accuracy@3: 74.28%
Accuracy@5: 80.91%
Accuracy@10: 88.15%
Precision@1: 60.41%
Recall@1: 36.30%
Precision@3: 35.63%
Recall@3: 50.31%
Precision@5: 27.57%
Recall@5: 56.53%
Precision@10: 19.54%
Recall@10: 65.02%
MRR@10: 0.6910
NDCG@10: 0.6275
MAP@100: 0.5498
Model Query Sparsity: Active Dimensions: 81.1, Sparsity Ratio: 0.9973
Model Corpus Sparsity: Active Dimensions: 199.8, Sparsity Ratio: 0.9935
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6275

--------------------------------- models/merged_model-1

Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 61.50%
Accuracy@3: 75.53%
Accuracy@5: 81.07%
Accuracy@10: 87.69%
Precision@1: 61.50%
Recall@1: 36.42%
Precision@3: 36.32%
Recall@3: 49.90%
Precision@5: 27.85%
Recall@5: 56.85%
Precision@10: 19.52%
Recall@10: 64.62%
MRR@10: 0.7002
NDCG@10: 0.6282
MAP@100: 0.5485
Model Query Sparsity: Active Dimensions: 82.4, Sparsity Ratio: 0.9973
Model Corpus Sparsity: Active Dimensions: 211.9, Sparsity Ratio: 0.9931
Average FLOPS: 5.5
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6282

--------------------------------- models/merged_model-2_256

Aggregated for Score Function: dot
Accuracy@1: 61.03%
Accuracy@3: 76.46%
Accuracy@5: 81.69%
Accuracy@10: 88.92%
Precision@1: 61.03%
Recall@1: 36.68%
Precision@3: 36.46%
Recall@3: 50.57%
Precision@5: 28.38%
Recall@5: 57.28%
Precision@10: 19.88%
Recall@10: 65.98%
MRR@10: 0.6980
NDCG@10: 0.6337
MAP@100: 0.5514
Model Query Sparsity: Active Dimensions: 77.1, Sparsity Ratio: 0.9975
Model Corpus Sparsity: Active Dimensions: 204.3, Sparsity Ratio: 0.9933
Average FLOPS: 6.1
Primary metric: NanoBEIR_mean_dot_ndcg@10
Primary metric value: 0.6337

"""
