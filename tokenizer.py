from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from huggingface_hub import HfApi, snapshot_download, whoami
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, pipeline

LANGUAGES = {
    "french": {"emoji": "🇫🇷", "nllb_code": "fra_Latn", "hf_code": "fr"},
    "english": {"emoji": "🇬🇧", "nllb_code": "eng_Latn", "hf_code": "en"},
    "german": {"emoji": "🇩🇪", "nllb_code": "deu_Latn", "hf_code": "de"},
    "italian": {"emoji": "🇮🇹", "nllb_code": "ita_Latn", "hf_code": "it"},
    "spanish": {"emoji": "🇪🇸", "nllb_code": "spa_Latn", "hf_code": "es"},
    "portuguese": {"emoji": "🇵🇹", "nllb_code": "por_Latn", "hf_code": "pt"},
}

MODELS = [
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "Alibaba-NLP/gte-multilingual-base",
    # "jinaai/jina-embeddings-v3", # TODO: uses ParametrizedEmbedding
]


def estimate_pruned_vocabulary(tokenizer: PreTrainedTokenizerFast, language: str):
    """
    Estimate the most common tokens in the language. You should first download the 1M sentences dataset
    for the desired language. Source: https://wortschatz.uni-leipzig.de/en/download/English
    """
    sentences_file = f"data.nosync/{language}_news_2020_1M-sentences.txt"
    if os.path.exists(sentences_file):
        my_bar = st.progress(0)
        df = pd.read_csv(sentences_file, sep="\t", header=None, quoting=csv.QUOTE_NONE, names=["id", "text"])
        counter = Counter(tokenizer.all_special_ids)
        for i, text in enumerate(df.text):
            counter.update(tokid for tokid in tokenizer.encode(text))
            my_bar.progress(i / len(df), text=f"{i / len(df) * 100:.0f}%")
        filtered_token_ids = sorted(counter.keys())
        filtered_tokens = tokenizer.convert_ids_to_tokens(filtered_token_ids)
        return set(filtered_tokens)
    else:
        raise FileNotFoundError


@st.cache_resource
def load_model_and_tokenizer(model_name: str):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    return model, tokenizer


def count_parameters(model, layer_name: str = None):
    return sum(p.numel() for name, p in model.named_parameters() if layer_name is None or name.startswith(layer_name))


@st.cache_resource
def get_test_sentence(target_lang: str, source_lang: str = "eng_Latn"):
    text = """
    Alan Mathison Turing (23 June 1912 - 7 June 1954) was an English mathematician,
    computer scientist, logician, cryptanalyst, philosopher and theoretical biologist.
    """
    if target_lang == "eng_Latn":
        return text
    model_name = "facebook/nllb-200-distilled-600M"
    translator = pipeline(task="translation", tokenizer=model_name, model=model_name)
    return translator(text, src_lang=source_lang, tgt_lang=target_lang)[0]["translation_text"]


def push_to_hub(hf_username: str, hf_token: str, model_dir: str, private: bool = False):
    api = HfApi(endpoint="https://huggingface.co", token=hf_token)
    repo_id = f"{hf_username}/{model_dir.split('/')[-1]}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=private)
    api.upload_folder(repo_id=repo_id, folder_path=model_dir, commit_message="Upload pruned model")


def prune_model(model_name: str, language: str, hf_username: str, hf_token: str, keep_english: bool):
    st.markdown(
        f"- Let's prune the [**{model_name}**](https://huggingface.co/{model_name}) model to keep its **{language.capitalize()}** tokens only."
    )

    # Load the model and its tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Calculate parameters for the original model
    all_params = count_parameters(model)
    encoder_params = count_parameters(model, layer_name="encoder")
    embedding_params = count_parameters(model, layer_name="embeddings")

    st.markdown(
        f"- The original model has **{all_params / 1e6:.1f}M** parameters, of which **{embedding_params / all_params * 100:.0f}%** "
        + f"(i.e., {embedding_params / 1e6:.1f}M params) come from the *embedding matrix* and its {tokenizer.vocab_size} token entries. "
        + f"This means that the contextualization of text sequences is actually done by a *{model.config.num_hidden_layers}-layer Transformer encoder* "
        + f"with **{encoder_params / 1e6:.1f}M** parameters only."
    )

    with st.status(f"Computing the {language.capitalize()} vocabulary...", expanded=True) as status:
        filtered_tokens = estimate_pruned_vocabulary(tokenizer, language)
        num_filtered_tokens = len(filtered_tokens)
        st.write(
            f"{language.capitalize()} only uses **{num_filtered_tokens / tokenizer.vocab_size * 100:.0f}%** "
            + f"of the model vocabulary (i.e., {num_filtered_tokens} out of the original {tokenizer.vocab_size} tokens)."
        )
        status.update(state="complete", expanded=True)

    if keep_english:
        with st.status("Computing the English vocabulary...", expanded=True) as status:
            english_tokens = estimate_pruned_vocabulary(tokenizer, "english")
            filtered_tokens.update(english_tokens)
            st.write(
                f"Considering the **English** tokens adds **{len(filtered_tokens) - num_filtered_tokens}** tokens to the vocabulary."
            )
            num_filtered_tokens = len(filtered_tokens)
            status.update(state="complete", expanded=True)

    with st.status("Pruning the model...", expanded=True) as status:
        st.write("- *Updating the tokenizer*")
        outdir = f"{language}-{model_name.split('/')[-1]}"

        # Export the tokenizer to a JSON string and access its vocabulary (list of lists: [[token, score], ...])
        tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
        original_vocab = tokenizer_json["model"]["vocab"]
        original_token_to_id = {entry[0]: idx for idx, entry in enumerate(original_vocab)}

        # Filter out the tokens to remove and reassign new IDs
        new_id = 0
        new_token_to_id = {}
        new_id_to_original_id = {}
        filtered_vocab_entries = []

        for token, score in original_vocab:
            if token in filtered_tokens:
                filtered_vocab_entries.append([token, score])
                new_token_to_id[token] = new_id
                new_id_to_original_id[new_id] = original_token_to_id[token]
                new_id += 1

        # Update the vocab in the tokenizer JSON and rebuild the tokenizer from the modified JSON
        tokenizer_json["model"]["vocab"] = filtered_vocab_entries
        new_backend_tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

        # Create a new tokenizer instance and save it
        new_tokenizer = PreTrainedTokenizerFast(tokenizer_object=new_backend_tokenizer, **tokenizer.init_kwargs)
        new_tokenizer.save_pretrained(outdir)

        st.write("- *Updating the embedding matrix*")
        new_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Create a new embedding matrix and map the original vectors to their new IDs
        original_embeddings = new_model.get_input_embeddings().weight.data
        new_embeddings = torch.nn.Embedding(
            num_embeddings=new_tokenizer.vocab_size,
            embedding_dim=model.config.hidden_size,
            padding_idx=new_tokenizer.pad_token_id,
        )

        for new_id in range(new_tokenizer.vocab_size):
            original_id = new_id_to_original_id.get(new_id)
            new_embeddings.weight.data[new_id] = original_embeddings[original_id]

        new_model.set_input_embeddings(new_embeddings)
        new_model.config.vocab_size = new_tokenizer.vocab_size
        new_model.save_pretrained(outdir)

        status.update(state="complete", expanded=True)

    with st.status("Testing the conversion...", expanded=True) as status:
        st.write("- *Checking the pruned tokenizer*")
        assert (
            len(new_tokenizer) == num_filtered_tokens
        ), f"ERROR: new tokenizer size ({len(new_tokenizer)}) != number of filtered tokens ({num_filtered_tokens})"
        assert filtered_tokens == set(
            new_tokenizer.convert_ids_to_tokens(range(len(new_tokenizer)))
        ), "ERROR: The new tokenizer vocabulary doesn't match number of the filtered tokens"

        st.write("- *Checking the pruned model*")
        test_sentence = get_test_sentence(LANGUAGES[language]["nllb_code"])
        with torch.inference_mode():
            emb1 = model(**tokenizer(test_sentence, return_tensors="pt")).last_hidden_state[:, 0][0].numpy()
            emb2 = new_model(**new_tokenizer(test_sentence, return_tensors="pt")).last_hidden_state[:, 0][0].numpy()
        diff = np.abs(emb1 - emb2).max()
        assert diff < 1e-6, f"ERROR: Some dimensions of the two vectors have a non negligible difference ({diff})"

        st.write(
            f"""All good! The output *[cls]* token embedding of the test sentence *"{test_sentence}"* should be similar:"""
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Original model:")
            st.code(f"{emb1.tolist()}")
        with col2:
            st.markdown("Pruned model:")
            st.code(f"{emb2.tolist()}")

        status.update(state="complete", expanded=True)

    # Show visually the result of the pruning process
    pruned_all_params = count_parameters(new_model)
    pruned_encoder_params = count_parameters(new_model, layer_name="encoder")
    pruned_embedding_params = count_parameters(new_model, layer_name="embeddings")
    st.markdown(f"The pruned model is **{pruned_all_params / all_params * 100:.1f}%** of the original model size.")
    data = {
        "Model": ["Original", "Pruned"],
        "Embedding": [embedding_params / 1e6, pruned_embedding_params / 1e6],
        "Encoder": [encoder_params / 1e6, pruned_encoder_params / 1e6],
    }
    fig = go.Figure(
        data=[
            go.Bar(
                name="Embedding matrix",
                x=data["Model"],
                y=data["Embedding"],
                text=data["Embedding"],
                textposition="inside",
                marker_color="#E5B4B4",
            ),
            go.Bar(
                name="Transformer encoder",
                x=data["Model"],
                y=data["Encoder"],
                text=data["Encoder"],
                textposition="inside",
                marker_color="#7FBFE0",
            ),
        ]
    )
    fig.update_layout(barmode="stack", yaxis_title="# Params (M)", height=400, margin=dict(t=10, b=10))
    fig.update_traces(texttemplate="%{text:.1f}M", textposition="inside", insidetextanchor="middle")
    st.plotly_chart(fig)

    with st.status("Pushing the pruned model to your Hugging Face account...", expanded=True) as status:
        st.write("- *Adding sentence-transformers files*")
        with tempfile.TemporaryDirectory() as tmpdirname:
            snapshot_download(repo_id=model_name, local_dir=tmpdirname, token=hf_token)

            src_modules_json = os.path.join(tmpdirname, "modules.json")
            if os.path.exists(src_modules_json):
                shutil.copy2(src_modules_json, os.path.join(outdir, "modules.json"))

            src_sentence_bert_config = os.path.join(tmpdirname, "sentence_bert_config.json")
            if os.path.exists(src_sentence_bert_config):
                shutil.copy2(src_sentence_bert_config, os.path.join(outdir, "sentence_bert_config.json"))

            src_pooling_folder = os.path.join(tmpdirname, "1_Pooling")
            if os.path.exists(src_pooling_folder):
                shutil.copytree(src_pooling_folder, os.path.join(outdir, "1_Pooling"), dirs_exist_ok=True)

            src_readme = os.path.join(tmpdirname, "README.md")
            if os.path.exists(src_readme):
                with open(src_readme, encoding="utf-8") as file:
                    content = file.read()
                    match = re.search(r"license:\s*(\S+)", content, re.IGNORECASE)
                    if match:
                        original_license = match.group(1)

        st.write("- *Adding a README*")
        new_model_name = f"{hf_username}/{outdir.split('/')[-1]}"
        readme_content = textwrap.dedent(f"""
        ---
        pipeline_tag: sentence-similarity
        language: {LANGUAGES[language]["hf_code"]}
        license: {original_license}
        tags:
        - passage-retrieval
        - sentence-similarity
        - pruned
        library_name: sentence-transformers
        base_model: {model_name}
        base_model_relation: quantized
        ---
        # {LANGUAGES[language]["emoji"]} {new_model_name.split("/")[-1]}

        This model is a {100 - pruned_all_params / all_params * 100:.1f}% smaller version of [{model_name}](https://huggingface.co/{model_name})
        for the {language.capitalize()} language, created using the [mtem-pruner](https://huggingface.co/spaces/antoinelouis/mtem-pruner) space.

        This pruned model should perform similarly to the original model for {language.capitalize()} language tasks with a much smaller
        memory footprint. However, it may not perform well for other languages present in the original multilingual model as tokens not
        commonly used in {language.capitalize()} were removed from the original multilingual model's vocabulary.

        ## Usage

        You can use this model with the Transformers library:

        ```python
        from transformers import AutoModel, AutoTokenizer

        model_name = "{new_model_name}"
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        ```

        Or with the sentence-transformers library:

        ```python
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("{new_model_name}")
        ```

        **Credits**: cc [@antoinelouis](https://huggingface.co/antoinelouis)
        """)
        with open(os.path.join(outdir, "README.md"), "w") as f:
            f.write(readme_content)

        st.write("- *Pushing to Hub*")
        push_to_hub(hf_username, hf_token, outdir)

        shutil.rmtree(outdir)
        status.update(state="complete", expanded=False)

    st.markdown("Done! You can now load your pruned model like this:")
    st.code(
        f"""
    from transformers import AutoModel, AutoTokenizer

    model_name = "{new_model_name}"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    """,
        language="python",
    )


def main():
    st.header("Multilingual Text Embedding Model Pruner")
    st.markdown("""
    This space helps you create a smaller, language-specific version of a multilingual text embedding model. Here's what it does:

    1. 🌎 Takes a state-of-the-art text embedding model that was trained on many languages
    2. ✂️ Trims it down to focus on just one language by removing unused tokens from its vocabulary
     3. 🚀 Gives you a smaller model that works just as well for your chosen language

    #### Why is this useful?

    - 💾 Get the same performance in your language with a much smaller model size
    - 🌐 Great for low-resource environments with limited RAM

    Ready to shrink your model? Let's get started!
    """)

    model_name = st.selectbox("Choose a multilingual model", MODELS)

    col1, col2 = st.columns([3, 1])
    with col1:
        language = st.selectbox(
            "Pick your target language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: f"{LANGUAGES[x]['emoji']} {x.capitalize()}",
        )
    with col2:
        st.write("")
        st.write("")
        keep_english = st.checkbox(
            "Keep English", value=False, help="Keep English tokens in addition to the selected language"
        )

    col3, col4 = st.columns(2)
    with col3:
        hf_username = st.text_input("Your Hugging Face username", placeholder="antoinelouis")
    with col4:
        hf_token = st.text_input(
            "Your Hugging Face access token", type="password", placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )

    if st.button("Prune model"):
        if not hf_username or not hf_token:
            st.error("Your HF username and access token are required to save the pruned model on your account.")
        else:
            _ = whoami(token=hf_token)
            prune_model(model_name, language, hf_username, hf_token, keep_english)

    st.markdown(
        """
        <style>
        .credits {
            position: fixed;
            right: 10px;
            bottom: 10px;
            color: #888888;
            font-size: 11px;
        }
        </style>
        <div class="credits">
            Credits to <a href="https://gist.github.com/avidale/44cd35bfcdaf8bedf51d97c468cc8001" target="_blank">@avidale</a> for inspiration.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
