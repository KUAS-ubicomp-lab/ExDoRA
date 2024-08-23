import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(engine):
    tokenizer = AutoTokenizer.from_pretrained(engine)
    model = AutoModelForCausalLM.from_pretrained(engine, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def load_dsm_criterion():
    dsm_criteria = [
        "Persistent sad, anxious, or 'empty' mood",
        "Feelings of hopelessness or pessimism",
        "Irritability",
        "Feelings of guilt, worthlessness, or helplessness",
        "Loss of interest or pleasure in hobbies and activities"
    ]
    return dsm_criteria


def load_data(source, text, label_1, label_2=None):
    expert_data = {}
    for root, _, files in os.walk(source):
        for file in files:
            data = pd.read_csv(os.path.join(root, file))
            expert_data[file.split('.')[0]] = [data[text].tolist(), data[label_1].tolist(),
                                               data[label_2].tolist() if label_2 else []]
    return expert_data


def average_scores(scores):
    avg_score = sum(scores) / len(scores)
    return avg_score
