import yaml
import argparse
from transformers import AutoModel

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(config):
    op_path = config["output"]["output_path"]
    model = AutoModel.from_pretrained(op_path)
    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    config_file = args.config
    config = load_config(config_file)
    model = load_model(config)