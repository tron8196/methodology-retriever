import yaml
import argparse
from transformers import AutoModel
import pandas as pd
import os
from transformers.trainer_callback import TrainerState
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

import os
import shutil

def encode_text_batch(texts, model, tokenizer, device, batch_size=4):
    encodings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        encodings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(encodings)

def average_precision(actual, predicted):
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(actual) if len(actual) > 0 else 0.0

def copy_files(source_folder, destination_folder):
    # Ensure destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    print(os.listdir(source_folder))
    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.json') or filename.endswith('.txt'):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            
            # Copy the file
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {filename}")


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(config):
    specter2_config_files_path = "specter2_base"
    op_path = config["output"]["output_path"]
    copy_files(specter2_config_files_path, op_path)
    model = AutoModel.from_pretrained(config["model"]["base_model"])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    config_file = args.config
    config = load_config(config_file)
    model = load_model(config)
    df = pd.read_csv("data/test.csv")

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = "cuda:0"
    df = pd.read_csv("data/test.csv")
    model = AutoModel.from_pretrained("allenai/specter2_base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

    # Encode all abstracts in the dataframe
    all_abstracts = df['cited_paper_abstracts'].tolist()
    all_abstract_encodings = torch.load("all_test_abstracts_encodings.pt")
    grouped = df.groupby('actual_ids')

    total_recall = 0
    total_map = 0
    total_papers = 0

    for citing_id, group in tqdm(grouped):
        proposal = group['proposal'].iloc[0]
        intents = group['intent'].tolist()
        
        # Encode proposal
        proposal_encoding = encode_text_batch([proposal], model, tokenizer, device)
        
        # Compute similarities with all abstracts in the dataframe
        similarities = cosine_similarity(proposal_encoding, all_abstract_encodings)[0]
        
        # Get all indices sorted by similarity
        all_indices_sorted = similarities.argsort()[::-1]
        
        # Calculate recall (let's say k=5 for this example)
        k = 3
        top_k_indices = all_indices_sorted[:k]
        
        # Get indices of abstracts cited by this paper
        cited_indices = df.index[df['actual_ids'] == citing_id].tolist()
        
        actual_method_abstracts = set([i for i in cited_indices if df.loc[i, 'triple_type'] == 1])
        retrieved_method_abstracts = set(top_k_indices) & actual_method_abstracts
        
        if len(actual_method_abstracts) > 0:
            recall = len(retrieved_method_abstracts) / len(actual_method_abstracts)
            total_recall += recall
            
            # Calculate Average Precision
            ap = average_precision(actual_method_abstracts, all_indices_sorted)
            total_map += ap
            
            total_papers += 1

    # Compute average recall and MAP
    average_recall = total_recall / total_papers if total_papers > 0 else 0
    mean_average_precision = total_map / total_papers if total_papers > 0 else 0

    print(f"Average Recall@{k}: {average_recall}")
    print(f"Mean Average Precision: {mean_average_precision}")

