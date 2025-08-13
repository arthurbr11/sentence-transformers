""" """

import logging
import traceback

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.sparse_encoder import callbacks, evaluation, losses, models

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name = "almanach/camembert-base"
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]

    train_batch_size = 32
    num_epochs = 10
    query_regularizer_weight = 0.001
    document_regularizer_weight = 0.1
    learning_rate = 2e-5

    # 1. Define our SparseEncoder model
    model = SparseEncoder(
        modules=[models.MLMTransformer(model_name, model_args={}), models.SpladePooling("max")],
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"splade-{short_model_name}-v2",
        ),
    )

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/sentence-transformers/msmarco
    dataset = load_dataset("arthurbresnu/msmarco-Qwen3-Reranker-0.6B-french")
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    logging.info(train_dataset)

    # 3. Define our training loss
    loss = losses.SpladeLoss(
        model,
        losses.SparseMarginMSELoss(model),
        query_regularizer_weight=query_regularizer_weight,
        document_regularizer_weight=document_regularizer_weight,
    )

    # 4. Define the evaluator. We use the SparseNanoBEIREvaluator, which is a light-weight evaluator for English
    nano_beir_3_evaluator = evaluation.SparseNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=train_batch_size
    )

    sts_french_dataset = load_dataset("CATIE-AQ/frenchSTS", split="validation")
    sts_french_evaluator = evaluation.SparseEmbeddingSimilarityEvaluator(
        sentences1=sts_french_dataset["sentence1"],
        sentences2=sts_french_dataset["sentence2"],
        scores=sts_french_dataset["score"],
        main_similarity="cosine",
        name="frenchSTS-dev",
    )

    evaluator = SequentialEvaluator(
        evaluators=[nano_beir_3_evaluator, sts_french_evaluator], main_score_function=lambda score: score[0]
    )

    # 5. Define the training arguments
    run_name = f"splade-{short_model_name}-v2"
    args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="eval_FrenchNanoBEIR_mean_dot_ndcg@10",
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=0.05,
        save_strategy="steps",
        save_steps=0.05,
        save_total_limit=2,
        logging_steps=50,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=42,
        save_safetensors=False,
    )

    # 6. Create the trainer & start training
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callbacks.SpladeRegularizerWeightSchedulerCallback(loss=loss, warmup_ratio=0.5)],
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, using the complete NanoBEIR dataset
    full_nanobeir = evaluation.SparseNanoBEIREvaluator(show_progress_bar=True, batch_size=train_batch_size)
    test_evaluator = SequentialEvaluator(
        evaluators=[full_nanobeir, sts_french_evaluator],
        main_score_function=lambda score: score[0],  # The main score to optimize for
    )
    test_evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir, safe_serialization=False)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = SparseEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
