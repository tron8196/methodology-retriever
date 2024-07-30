import yaml
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch.nn.functional as F
from models.specter2_ft import Specter2Ft
from models.TripletDataset import CitationTripletDataset, TripletDataCollator
import os
from transformers import Trainer, TrainingArguments

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(config):

    specter2_model_name = config["model"]["base_model"]
    model = Specter2Ft(specter2_model_name)
    data_root = config["data"]["data_dir"]
    train_path = os.path.join(data_root, "dev.csv")
    op_path = config["output"]["output_path"]
    dataset = CitationTripletDataset(train_path, specter2_model_name)
    collator_fn = TripletDataCollator()

    training_args = TrainingArguments(
    output_dir=op_path,
    num_train_epochs=int(config["model"]["num_train_epochs"]),
    per_device_train_batch_size=int(config["model"]["per_device_train_batch_size"]),
    gradient_accumulation_steps=int(config["model"]["gradient_accumulation_steps"]),
    learning_rate=float(config["model"]["learning_rate"]),
    weight_decay=float(config["model"]["weight_decay"]),
    logging_dir=config["model"]["logging_dir"],
    logging_steps=config["model"]["logging_steps"],
    save_strategy=config["model"]["save_strategy"],
    evaluation_strategy=config["model"]["evaluation_strategy"],
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator_fn,
        eval_dataset=dataset
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(op_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    config_file = args.config
    config = load_config(config_file)
    # print(config)
    train(config)



