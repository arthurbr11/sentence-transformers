import logging
import random

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.models import Router
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.sparse_encoder.models import MLMTransformer, SparseStaticEmbedding, SpladePooling
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# 1. Load a model to finetune with 2. (Optional) model card data
model_name = "FacebookAI/xlm-roberta-large"
mlm_transformer = MLMTransformer(model_name)
splade_pooling = SpladePooling(
    pooling_strategy="max", word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension()
)
router = Router.for_query_document(
    query_modules=[SparseStaticEmbedding(tokenizer=mlm_transformer.tokenizer, frozen=False)],
    document_modules=[mlm_transformer, splade_pooling],
)

model = SparseEncoder(
    modules=[router],
    model_card_data=SparseEncoderModelCardData(
        language="en",
        license="apache-2.0",
        model_name="Inference-free SPLADE xlm-roberta-large trained on MS MARCO triplets",
    ),
)
print(model.max_seq_length)

# 3. Load a dataset to finetune on
dataset_size = 100_000  # We only use the first 100k samples for training
logging.info("The dataset has not been fully stored as texts on disk yet. We will do this now.")
corpus = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
corpus = dict(zip(corpus["passage_id"], corpus["passage"]))
queries = load_dataset("sentence-transformers/msmarco", "queries", split="train")
queries = dict(zip(queries["query_id"], queries["query"]))
dataset = load_dataset("sentence-transformers/msmarco", "triplets", split="train")
random_indices = random.sample(range(len(dataset)), dataset_size)
dataset = dataset.select(random_indices)


def id_to_text_map(batch):
    return {
        "query": [queries[qid] for qid in batch["query_id"]],
        "positive": [corpus[pid] for pid in batch["positive_id"]],
        "negative": [corpus[pid] for pid in batch["negative_id"]],
    }


dataset = dataset.map(id_to_text_map, batched=True, remove_columns=["query_id", "positive_id", "negative_id"])
dataset = dataset.train_test_split(test_size=10_000)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
logging.info(train_dataset)

# 4. Define a loss function
loss = SpladeLoss(
    model=model,
    loss=SparseMultipleNegativesRankingLoss(model=model),
    query_regularizer_weight=0,
    document_regularizer_weight=3e-4,
)

# 5. (Optional) Specify training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"if-splade-{short_model_name}-msmarco-mrl"
args = SparseEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    learning_rate_mapping={
        r"SparseStaticEmbedding\.weight": 1e-3
    },  # Set a higher learning rate for the SparseStaticEmbedding module
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    router_mapping={
        "query": "query",
        "positive": "document",
        "negative": "document",
    },  # Map the column names to the routes
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    seed=42,
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = SparseNanoBEIREvaluator(
    dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=16, max_active_dims=512
)

# 7. Create a trainer & train
trainer = SparseEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 8. Evaluate the model performance again after training
dev_evaluator(model)

# 9. Save the trained model
model.save_pretrained(f"models/{run_name}/final")

# 10. (Optional) Push it to the Hugging Face Hub
model.push_to_hub(run_name)
