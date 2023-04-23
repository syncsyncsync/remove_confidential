# tests/test_remove_confidential.py
import os
import torch
import tempfile
import shutil
import pytest
from remove_confidential import ConfidentialDataset, train, predict
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from train import get_train_parser
from predict import get_predict_parser
from argparse import ArgumentParser

def test_train():
    model_name = "bert-base-multilingual-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = [("This is a sample input text.", "This is a sample output text.")]
    dataset = ConfidentialDataset(data, tokenizer)

    with tempfile.TemporaryDirectory() as temp_dir:
        train(fine_tuned_model_name=temp_dir, model_type="bert", num_epochs=1)
        assert os.path.exists(os.path.join(temp_dir, "pytorch_model.bin"))
        assert os.path.exists(os.path.join(temp_dir, "config.json"))
        assert os.path.exists(os.path.join(temp_dir, "tokenizer_config.json"))
        assert os.path.exists(os.path.join(temp_dir, "special_tokens_map.json"))
        assert os.path.exists(os.path.join(temp_dir, "vocab.txt"))



def test_predict():
    input_text = "This is a test [MASK]."

    with tempfile.TemporaryDirectory() as temp_dir:
        model_name = "bert-base-multilingual-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)

        predicted_text = predict(input_text, fine_tuned_model_name=temp_dir, model_type="bert")
        assert predicted_text is not None
        assert isinstance(predicted_text, str)





