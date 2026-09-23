import os
import re
import json
import pandas as pd
import numpy as np
from typing import List

from nltk.stem import LancasterStemmer

from agents.base import BaseAgent

PROJECT_BASE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(PROJECT_BASE, 'prompt_templates')

def log_outputs(outputs, dataset, model, tracing_model="none", particle="0", **kwargs):
    outputs_dir = os.path.join(PROJECT_BASE, "outputs", dataset, kwargs['condition'])
    os.makedirs(outputs_dir, exist_ok=True)

    # log outputs in json lines format
    model_name = model.split("/")[-1]
    tracing_model_name = tracing_model.split("/")[-1]
    output_file = os.path.join(outputs_dir, f"model-{model_name}_tracer-{tracing_model_name}_particle-n-{particle}_{dataset}.jsonl")

    try:
        with open(output_file, 'a') as f:
            for output in outputs:
                f.write(json.dumps(output) + '\n')
    except:
        print("Error writing to file")

def read_logs(dataset, model, tracing_model="none", particle="0", **kwargs):
    outputs_dir = os.path.join(PROJECT_BASE, "outputs", dataset, kwargs['condition'])
    model_name = model.split("/")[-1]
    tracing_model_name = tracing_model.split("/")[-1]
    output_file = os.path.join(outputs_dir, f"model-{model_name}_tracer-{tracing_model_name}_particle-n-{particle}_{dataset}.jsonl")

    if os.path.exists(output_file):
        logs = pd.read_json(output_file, lines=True)
        return logs
    else:
        return None

def load_prompt(path: str):
    if path is None:
        return None
    
    template_path = os.path.join(TEMPLATE_DIR, path)
    with open(template_path, 'r') as file:
        prompt_template = file.read()
    return prompt_template

def capture_and_parse_ordered_list(text: str):
    # Regular expression to match ordered list items (numbers followed by a period and a space)
    ordered_list_pattern = re.compile(r'(\d+\.\s+.+?)(?=\d+\.\s|$)', re.DOTALL)
    
    # Find all matches
    matches = ordered_list_pattern.findall(text)
    
    # Remove numbers from the beginning of each match
    ordered_list = [re.sub(r'^\d+\.\s+', '', match).strip() for match in matches]
    
    return ordered_list

def prompting_for_ordered_list(model: BaseAgent, prompt: str, n: int, history: List = None, system_prompt: str = None) -> List[str]:
    tolerance = 3
    temperature = 0
    while tolerance > 0:
        output = model.interact(prompt, max_tokens=1024, temperature=temperature, history=history, system_prompt=system_prompt)
        parsed_list = capture_and_parse_ordered_list(output)
        if len(parsed_list) >= n:
            parsed_list = parsed_list[:n]
            break
        else:
            temperature += 0.33
            tolerance -= 1
    return parsed_list

def list_to_unordered_list_string(items, list_bullet="-"):
    """
    Converts a list of items into a string formatted as an unordered list.

    Args:
        items (list): A list of items to be converted into an unordered list.

    Returns:
        str: A string formatted as an unordered list.
    """
    # Convert list items to strings and format as an unordered list
    unordered_list = "\n".join([f"{list_bullet} {item}".strip() for item in items])
    
    return unordered_list

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def jaccard_similarity(sent1, sent2):
    lancaster = LancasterStemmer()
    _tokens1 = sent1.lower().split()
    _tokens2 = sent2.lower().split()
    tokens1 = set([lancaster.stem(token) for token in _tokens1])
    tokens2 = set([lancaster.stem(token) for token in _tokens2])
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def overall_jaccard_similarity(sentences):
    n = len(sentences)
    total_similarity = 0.0
    pair_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):  # j starts at i + 1 to skip self-similarity and redundant pairs
            total_similarity += jaccard_similarity(sentences[i], sentences[j])
            pair_count += 1
    
    # If pair_count is 0 (e.g., when n=1), avoid division by zero by returning 1 (self-similarity)
    return total_similarity / pair_count if pair_count > 0 else 1.0

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)