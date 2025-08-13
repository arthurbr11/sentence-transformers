from __future__ import annotations

import json

import torch
from tokenizers import Tokenizer
from tokenizers.normalizers import NFC, Lowercase, Sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def create_lowercase_modernbert_tokenizer(
    original_model_name="answerdotai/ModernBERT-base", output_dir="./modernbert-lowercase"
):
    """
    Create a lowercase-only tokenizer from ModernBERT-base while preserving GPT-2 style structure
    """
    # '''
    print("Loading original tokenizer...")
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)

    # Load the original tokenizer.json to get the full vocabulary structure
    tokenizer_path = "./temp_tokenizer"
    original_tokenizer.save_pretrained(tokenizer_path)

    # with open(os.path.join(tokenizer_path, "tokenizer.json"), "r", encoding="utf-8") as f:
    #     tokenizer_data = json.load(f)

    # Get the original vocabulary from the tokenizer.json
    # original_vocab = tokenizer_data["model"]["vocab"]

    # print(f"Original vocabulary size: {len(original_vocab)}")

    # Create new vocabulary with lowercase tokens
    # old_to_new_mapping = {}
    # print(tokenizer_data.keys())

    """
    added_tokens = [tokenizer_data["added_tokens"][i]["content"] for i in range(len(tokenizer_data["added_tokens"]))]

    # Step 2: Process regular tokens
    # First, collect all lowercase versions and their frequencies
    lowercase_tokens = {}

    for token, old_id in original_vocab.items():
        if token not in added_tokens:
            # Handle Ġ and Ċ prefixes (GPT-2 style space prefix)
            if "Ġ" in token or "Ċ" in token:
                lowercase_token = ""
                for char in token:
                    if char in ["Ġ", "Ċ"]:
                        lowercase_token += char
                    else:
                        lowercase_token += char.lower()
            else:
                # Regular token, just lowercase
                lowercase_token = token.lower()

            if lowercase_token not in lowercase_tokens:
                lowercase_tokens[lowercase_token] = []
            lowercase_tokens[lowercase_token].append((token, old_id))
        else:
            # If it's an added token, keep it as is
            lowercase_tokens[token] = [(token, old_id)]

    # Step 3: Assign new IDs to unique lowercase tokens
    current_id = 0

    for lowercase_token, original_tokens in lowercase_tokens.items():
        # Use the first occurrence's ID if possible, otherwise assign new ID
        if lowercase_token not in new_vocab:
            new_vocab[lowercase_token] = current_id
            current_id += 1

        # Map all original tokens to this lowercase version
        for orig_token, orig_id in original_tokens:
            old_to_new_mapping[orig_id] = new_vocab[lowercase_token]
    """

    new_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    new_tokenizer._tokenizer.normalizer = Sequence([NFC(), Lowercase()])
    tokenizer_data = json.loads(new_tokenizer.backend_tokenizer.to_str())

    original_vocab = original_tokenizer.get_vocab()

    # original_token_to_id = original_tokenizer.get_vocab()
    # print(f"Original vocabulary size: {len(original_token_to_id)}")

    # sort tokens and ids by ids
    sorted_tokens_ids = sorted(original_vocab.items(), key=lambda x: x[1])
    tokens, ids = zip(*sorted_tokens_ids)
    tokens_to_tokenize = [token.replace("Ġ", " ").replace("Ċ", " ") for token in tokens]
    retokenized = new_tokenizer(
        tokens_to_tokenize, add_special_tokens=False, truncation=False, padding=False
    ).input_ids

    deleted = [
        retokenized_ids != [original_id] and token != "Ġ" * len(token)
        for retokenized_ids, original_id, token in zip(retokenized, ids, tokens)
    ]
    new_tokens = [token for token, del_flag in zip(tokens, deleted) if not del_flag]
    # new_vocab_size = len(tokens) - sum(deleted)
    new_vocab = {token: i for i, token in enumerate(new_tokens)}

    old_to_new_mapping = {
        original_id: new_vocab[token] for token, original_id in original_vocab.items() if token in new_vocab
    }

    # print(current_id)
    print(f"New vocabulary size: {len(new_vocab)}")
    print(
        f"Vocabulary reduction: {len(original_vocab)} -> {len(new_vocab)} ({len(original_vocab) - len(new_vocab)} tokens removed)"
    )
    # '''
    # '''
    # Update the tokenizer.json structure
    tokenizer_data["model"]["vocab"] = new_vocab

    # Update added_tokens in tokenizer.json
    new_added_tokens = []
    for token_info in tokenizer_data["added_tokens"]:
        token_content = token_info["content"]
        # Get the new ID from our re-indexed vocabulary
        if token_content in new_vocab:
            old_to_new_mapping[token_info["id"]] = new_vocab[token_content]
            token_info["id"] = new_vocab[token_content]
            new_added_tokens.append(token_info)
        else:
            old_to_new_mapping[token_info["id"]] = len(old_to_new_mapping)
            token_info["id"] = len(old_to_new_mapping)
            new_added_tokens.append(token_info)

    tokenizer_data["added_tokens"] = new_added_tokens

    for special_token in tokenizer_data["post_processor"]["special_tokens"].keys():
        for token_info in tokenizer_data["added_tokens"]:
            if special_token == token_info["content"]:
                good_id = token_info["id"]
                break
        old_to_new_mapping[tokenizer_data["post_processor"]["special_tokens"][special_token]["ids"][0]] = good_id
        tokenizer_data["post_processor"]["special_tokens"][special_token]["ids"] = [good_id]

    # make sure that the merges in tokenizer.json are still valid so with lower case
    # we can use the same merges as in original tokenizer
    merges = tokenizer_data["model"]["merges"]
    lowercase_merges = []
    for i, merge in enumerate(merges):
        # for j, token in enumerate(merge):
        #     '''
        #     if "Ġ" in token or "Ċ" in token:
        #         lowercase_token = ""
        #         for char in token:
        #             if char in ["Ġ", "Ċ"]:
        #                 lowercase_token += char
        #             else:
        #                 lowercase_token += char.lower()
        #     else:
        #         # Regular token, just lowercase
        #         lowercase_token = token.lower()
        #     if lowercase_token not in new_vocab:
        #         print(f"Token '{token}' not in new vocabulary")
        #     '''
        #     if token not in new_vocab:
        #         print(f"Token '{token}' not in new vocabulary")

        #     merges[i][j] = lowercase_token
        # if any([token == "ĠĠ" for token in merge]):
        #     breakpoint()
        if all([token in new_vocab for token in merge]):
            lowercase_merges.append(merge)
    tokenizer_data["model"]["merges"] = lowercase_merges
    print(len(lowercase_merges), "merges kept after lowercasing out of", len(merges))

    # breakpoint()
    new_backend_tokenizer = Tokenizer.from_str(json.dumps(tokenizer_data))
    new_tokenizer = PreTrainedTokenizerFast(tokenizer_object=new_backend_tokenizer, **new_tokenizer.init_kwargs)

    return new_tokenizer, old_to_new_mapping

    """
    # Load tokenizer_config.json
    with open(os.path.join(tokenizer_path, "tokenizer_config.json"), "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    # Set add_prefix_space to True
    tokenizer_config["add_prefix_space"] = True
    tokenizer_config["do_lower_case"] = True

    # Update added_tokens_decoder for new vocabulary
    new_added_tokens_decoder = {}
    for token_id_str, token_info in tokenizer_config["added_tokens_decoder"].items():
        old_id = int(token_id_str)
        if old_id in old_to_new_mapping:
            new_id = old_to_new_mapping[old_id]
            new_added_tokens_decoder[str(new_id)] = token_info

    # The unused tokens are already handled in the regular added_tokens_decoder processing above
    # No need to add them separately since they're already in the original added_tokens_decoder

    tokenizer_config["added_tokens_decoder"] = new_added_tokens_decoder

    """
    """
    # make sure that the merges in tokenizer.json are still valid so with lower case
    # we can use the same merges as in original tokenizer
    merges = tokenizer_data["model"]["merges"]
    for i, merge in enumerate(merges):
        for j, token in enumerate(merge):
            if "Ġ" in token or "Ċ" in token:
                lowercase_token = ""
                for char in token:
                    if char in ["Ġ", "Ċ"]:
                        lowercase_token += char
                    else:
                        lowercase_token += char.lower()
            else:
                # Regular token, just lowercase
                lowercase_token = token.lower()
            merges[i][j] = lowercase_token
    """

    """
    # Save the modified tokenizer
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    # Copy special_tokens_map.json and update IDs if needed
    with open(os.path.join(tokenizer_path, "special_tokens_map.json"), "r", encoding="utf-8") as f:
        special_tokens_map = json.load(f)

    with open(os.path.join(output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

    # Clean up temp directory
    import shutil

    shutil.rmtree(tokenizer_path)

    return new_vocab, old_to_new_mapping, current_id - 1
    """
    # '''
    """
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    new_tokenizer = AutoTokenizer.from_pretrained(original_model_name)

    original_token_to_id = original_tokenizer.get_vocab()
    print(f"Original vocabulary size: {len(original_token_to_id)}")

    new_tokenizer._tokenizer.normalizer = Sequence([NFC(), Lowercase()])

    # sort tokens and ids by ids
    sorted_tokens_ids = sorted(original_token_to_id.items(), key=lambda x: x[1])
    tokens, ids = zip(*sorted_tokens_ids)
    tokens_to_tokenize = [token.replace("Ġ", " ").replace("Ċ", " ") for token in tokens]
    retokenized = new_tokenizer(tokens_to_tokenize, add_special_tokens=False, truncation=False, padding=False).input_ids

    deleted = [
        retokenized_ids != [original_id]
        for retokenized_ids, original_id in zip(retokenized, ids)
    ]
    new_tokens = [token for token, del_flag in zip(tokens, deleted) if not del_flag]
    new_vocab_size = len(tokens) - sum(deleted)
    new_vocabulary = {token: i for i, token in enumerate(new_tokens)}

    old_to_new_mapping = {original_id: new_vocabulary[token] for token, original_id in original_token_to_id.items() if token in new_vocabulary}

    tokenizer_json = json.loads(new_tokenizer.backend_tokenizer.to_str())
    tokenizer_json['model']['vocab'] = new_vocabulary
    breakpoint()
    new_backend_tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    new_tokenizer = PreTrainedTokenizerFast(tokenizer_object=new_backend_tokenizer, **new_tokenizer.init_kwargs)

    return old_to_new_mapping, new_tokenizer
    # """


def update_modernbert_embeddings(model, old_to_new_mapping, new_vocab_size):
    """
    Update the embedding layer and MLM head for ModernBERT with new vocabulary
    """
    print("Updating ModernBERT embeddings...")

    model.config.vocab_size = new_vocab_size
    model.config.pad_token_id = old_to_new_mapping[model.config.pad_token_id]
    model.config.bos_token_id = old_to_new_mapping[model.config.bos_token_id]
    model.config.eos_token_id = old_to_new_mapping[model.config.eos_token_id]
    model.config.cls_token_id = old_to_new_mapping[model.config.cls_token_id]
    model.config.sep_token_id = old_to_new_mapping[model.config.sep_token_id]

    # Get original embeddings
    original_embeddings = model.model.embeddings.tok_embeddings.weight.data
    original_vocab_size = original_embeddings.size(0)
    embedding_dim = original_embeddings.size(1)

    print(f"Original embedding matrix: {original_embeddings.shape}")
    print(f"New vocabulary size: {new_vocab_size}")

    # Create new embedding matrix
    new_embeddings = torch.zeros(new_vocab_size, embedding_dim)

    # Initialize with small random values (similar to ModernBERT's initialization)
    std = model.config.initializer_range
    torch.nn.init.normal_(new_embeddings, mean=0.0, std=std)

    # Map old embeddings to new positions
    for old_id, new_id in old_to_new_mapping.items():
        if old_id < original_vocab_size and new_id < new_vocab_size:
            new_embeddings[new_id] = original_embeddings[old_id]

    # Update the embedding layer
    model.model.embeddings.tok_embeddings = torch.nn.Embedding(
        new_vocab_size, embedding_dim, padding_idx=model.config.pad_token_id
    )
    model.model.embeddings.tok_embeddings.weight.data = new_embeddings

    # Update MLM head (ModernBERT has tied embeddings, so we need to update the MLM head too)
    if hasattr(model, "decoder"):
        print("Updating MLM head...")

        # If not tied, update the decoder separately
        original_decoder_weight = model.decoder.weight.data
        new_decoder_weight = torch.zeros(new_vocab_size, embedding_dim)
        torch.nn.init.normal_(new_decoder_weight, mean=0.0, std=std)
        original_bias = model.decoder.bias.data
        new_bias = torch.zeros(new_vocab_size)

        for old_id, new_id in old_to_new_mapping.items():
            if old_id < original_vocab_size and new_id < new_vocab_size:
                new_decoder_weight[new_id] = original_decoder_weight[old_id]
                new_bias[new_id] = original_bias[old_id]

        model.decoder = torch.nn.Linear(embedding_dim, new_vocab_size, bias=model.config.decoder_bias)
        model.decoder.weight.data = new_decoder_weight

        model.decoder.bias.data = new_bias

    print(f"Embeddings updated: {original_embeddings.shape} -> {new_embeddings.shape}")
    return model


def create_lowercase_modernbert_model(
    original_model_name="answerdotai/ModernBERT-base", output_dir="./modernbert-lowercase", save_model=True
):
    """
    Complete pipeline to create a lowercase ModernBERT model
    """
    print("=== Creating Lowercase ModernBERT Model ===")

    # Step 1: Create lowercase tokenizer
    new_tokenizer, old_to_new_mapping = create_lowercase_modernbert_tokenizer(original_model_name, output_dir)

    # Step 2: Load the original model
    print("Loading original ModernBERT model...")
    model = AutoModelForMaskedLM.from_pretrained(original_model_name)

    # Step 3: Update embeddings and MLM head
    model = update_modernbert_embeddings(model, old_to_new_mapping, len(old_to_new_mapping))

    # Step 5: Save the modified model
    if save_model:
        print(f"Saving modified model to {output_dir}...")
        model.save_pretrained(output_dir)

        # Update and save config
        model.config.save_pretrained(output_dir)

    # Step 6: Test the new tokenizer
    print("Testing new tokenizer...")
    # new_tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # If you need to add lowercasing:
    # from tokenizers.normalizers import Lowercase, Sequence

    # new_tokenizer._tokenizer.normalizer = Sequence([Lowercase()])

    new_tokenizer.save_pretrained(output_dir)
    # Test cases
    test_cases = [
        "Hello World!",
        "The Quick BROWN Fox",
        "UPPERCASE text",
        "MixedCase Words",
        "renew the CONTRACT",
        "The quick [MASK] fox jumps over the lazy dog.",
    ]

    print("\n=== Tokenizer Test Results ===")
    original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)

    for test_text in test_cases:
        print(f"\nInput: '{test_text}'")
        try:
            orig_tokens = original_tokenizer.tokenize(test_text)
            new_tokens = new_tokenizer.tokenize(test_text)
            print(f"Original: {orig_tokens}")
            print(f"New:      {new_tokens}")
        except Exception as e:
            print(f"Error tokenizing: {e}")

    print("\nVocabulary sizes:")
    print(f"Original: {len(original_tokenizer.get_vocab())}")
    print(f"New:      {len(new_tokenizer.get_vocab())}")

    print(f"\nAdd prefix space: {new_tokenizer.add_prefix_space}")

    return model, new_tokenizer, old_to_new_mapping


def test_model_inference(model, tokenizer):
    """Test that the model can perform inference with the new tokenizer"""
    print("\n=== Testing Model Inference ===")

    # test_text = "The sun is usually of the color [MASK] in the sky."
    test_text = "The Sun Is Usually Of The Color [MASK] In The Sky."

    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    print(f"Input text: {test_text}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Input IDs: {inputs['input_ids']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Get prediction for the [MASK] token
        print(tokenizer.mask_token_id)
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        if len(mask_token_index) > 0:
            mask_logits = logits[0, mask_token_index[0], :]
            predicted_token_id = torch.argmax(mask_logits).item()
            predicted_token = tokenizer.decode([predicted_token_id])
            print(f"Predicted token for [MASK]: '{predicted_token}'")
            input_ids = inputs["input_ids"][0].tolist()
            input_ids[mask_token_index[0]] = predicted_token_id
            print("Full decoded output:", tokenizer.decode(input_ids))

        print(f"Output logits shape: {logits.shape}")
        print("Model inference successful!")


# Example usage
if __name__ == "__main__":
    for size in ["400m"]:  # ["17m", "32m", "68m", "150m", "400m", "1b"]:
        # Create the lowercase ModernBERT model
        model, tokenizer, mapping = create_lowercase_modernbert_model(
            original_model_name=f"jhu-clsp/ettin-encoder-{size}",
            output_dir=f"./ettin-encoder-{size}-lowercase",
            save_model=True,
        )

        # Test the model
        test_model_inference(model, tokenizer)
        original_model_name = f"jhu-clsp/ettin-encoder-{size}"
        original_model = AutoModelForMaskedLM.from_pretrained(original_model_name)
        original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
        test_model_inference(original_model, original_tokenizer)
