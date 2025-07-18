from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.utils import is_ci

try:
    from optimum.intel import OVModelForFeatureExtraction
    from optimum.onnxruntime import ORTModelForFeatureExtraction
except ImportError:
    pytest.skip("OpenVINO and ONNX backends are not available", allow_module_level=True)

from sentence_transformers import SentenceTransformer

if is_ci():
    pytest.skip("Skip test in CI to try and avoid 429 Client Error", allow_module_level=True)


## Testing exporting:
@pytest.mark.parametrize(
    ["backend", "expected_auto_model_class"],
    [
        ("onnx", ORTModelForFeatureExtraction),
        ("openvino", OVModelForFeatureExtraction),
    ],
)
@pytest.mark.parametrize(
    "model_kwargs", [{}, {"file_name": "wrong_file_name"}]
)  # <- Using a file_name is fine when exporting
def test_backend_export(backend, expected_auto_model_class, model_kwargs) -> None:
    model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", backend=backend, model_kwargs=model_kwargs
    )
    assert model.get_backend() == backend
    assert isinstance(model.transformers_model, expected_auto_model_class)
    embedding = model.encode("Hello, World!")
    assert embedding.shape == (model.get_sentence_embedding_dimension(),)


def test_backend_no_export_crash():
    # ONNX Crashes when it can't export & the model repo/path doesn't contain an exported model
    with pytest.raises(OSError):
        SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors", backend="onnx", model_kwargs={"export": False}
        )

    # OpenVINO will forcibly override the export=False if the model repo/path doesn't contain an exported model
    # But only starting from v1.19.0
    model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", backend="openvino", model_kwargs={"export": False}
    )
    assert isinstance(model.transformers_model, OVModelForFeatureExtraction)


## Testing loading exported models:
@pytest.mark.parametrize(
    ["backend", "model_id"],
    [
        ("onnx", "sentence-transformers-testing/stsb-bert-tiny-onnx"),
        ("openvino", "sentence-transformers-testing/stsb-bert-tiny-openvino"),
    ],
)
@pytest.mark.parametrize(
    ["model_kwargs", "exception"],
    [
        [{}, False],
        [{"file_name": "wrong_file_name", "export": True}, False],  # Using a file_name is fine when exporting
        [{"file_name": "wrong_file_name", "export": False}, True],  # ... but fails when not exporting
    ],
)
def test_backend_load(backend, model_id, model_kwargs, exception) -> None:
    if exception:
        with pytest.raises((OSError, RuntimeError)):
            SentenceTransformer(model_id, backend=backend, model_kwargs=model_kwargs)
    else:
        model = SentenceTransformer(model_id, backend=backend, model_kwargs=model_kwargs)
        assert model.get_backend() == backend
        embedding = model.encode("Hello, World!")
        assert embedding.shape == (model.get_sentence_embedding_dimension(),)


def test_onnx_provider_crash() -> None:
    with pytest.raises(ValueError):
        SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-onnx",
            backend="onnx",
            model_kwargs={"provider": "incorrect_provider"},
        )


def test_openvino_provider() -> None:
    model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-openvino",
        backend="openvino",
        model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "precision_1"}},
    )
    assert model.transformers_model.ov_config == {
        "INFERENCE_PRECISION_HINT": "precision_1",
        "PERFORMANCE_HINT": "LATENCY",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        ov_config_path = os.path.join(temp_dir, "ov_config.json")
        with open(ov_config_path, "w") as ov_config_file:
            json.dump({"INFERENCE_PRECISION_HINT": "precision_2"}, ov_config_file)

        model = SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-openvino",
            backend="openvino",
            model_kwargs={"ov_config": ov_config_path},
        )
        assert model.transformers_model.ov_config == {
            "INFERENCE_PRECISION_HINT": "precision_2",
            "PERFORMANCE_HINT": "LATENCY",
        }


def test_incorrect_backend() -> None:
    with pytest.raises(ValueError):
        SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", backend="incorrect_backend")


def test_openvino_backend() -> None:
    model_id = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    # Test that OpenVINO output is close to PyTorch output
    pytorch_model = SentenceTransformer(model_id)
    openvino_model = SentenceTransformer(
        model_id,
        backend="openvino",
        model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "f32"}},
    )
    pytorch_result = pytorch_model.encode(["Hello there!"])
    openvino_result = openvino_model.encode(["Hello there!"])
    assert np.allclose(openvino_result, pytorch_result, atol=0.000001), "OpenVINO and Pytorch outputs are not close"

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test that loading with ov_config file works as expected
        config_file = str(Path(tmpdirname) / "ov_config.json")
        with open(Path(config_file), "w") as f:
            f.write('{"NUM_STREAMS" : "2"}')
        openvino_model_with_config = SentenceTransformer(
            model_id,
            backend="openvino",
            model_kwargs={"ov_config": config_file},
        )
        # The transformers model is an Optimum model with an OpenVINO inference request property
        assert openvino_model_with_config.transformers_model.request.get_property("NUM_STREAMS") == 2

        # Test that saving and loading local OpenVINO models works as expected
        openvino_model_with_config.save_pretrained(tmpdirname)
        local_openvino_model = SentenceTransformer(
            tmpdirname, backend="openvino", model_kwargs={"ov_config": {"INFERENCE_PRECISION_HINT": "f32"}}
        )
        local_openvino_result = local_openvino_model.encode(["Hello there!"])
        assert np.allclose(
            local_openvino_result, openvino_result
        ), "OpenVINO saved model output differs from in-memory converted model"
        del local_openvino_model
        gc.collect()


def test_export_false_subfolder() -> None:
    model_id = "sentence-transformers-testing/stsb-bert-tiny-openvino"

    def from_pretrained_decorator(method):
        def decorator(*args, **kwargs):
            assert not kwargs["export"]
            assert kwargs["subfolder"] == "openvino"
            assert kwargs["file_name"] == "openvino_model.xml"
            return method(*args, **kwargs)

        return decorator

    OVModelForFeatureExtraction.from_pretrained = from_pretrained_decorator(
        OVModelForFeatureExtraction.from_pretrained
    )
    SentenceTransformer(model_id, backend="openvino", model_kwargs={"export": False})


def test_export_set_nested_filename() -> None:
    model_id = "sentence-transformers-testing/stsb-bert-tiny-openvino"

    def from_pretrained_decorator(method):
        def decorator(*args, **kwargs):
            assert kwargs["subfolder"] == "openvino"
            assert kwargs["file_name"] == "openvino_model.xml"
            return method(*args, **kwargs)

        return decorator

    OVModelForFeatureExtraction.from_pretrained = from_pretrained_decorator(
        OVModelForFeatureExtraction.from_pretrained
    )
    SentenceTransformer(model_id, backend="openvino", model_kwargs={"file_name": "openvino/openvino_model.xml"})
