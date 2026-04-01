import os, torch, time, argparse

import yaml
from utils import Config
from model_trainer import Trainer

if __name__ == "__main__":

  os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

  parser = argparse.ArgumentParser(description="trainer")
  parser.add_argument('--config_file', type=str, default='config/config.yaml', help="path to YAML config")
  parser.add_argument('--output_dir', type=str, default=None, help="path to output directory (optional); defaults to outputs/model_name")
  parser.add_argument('--data_dir', type=str, default=None, help="path to data directory")
  args = parser.parse_args()

  # Load YAML configuration
  with open(args.config_file, 'r') as file:
    config_dict = yaml.safe_load(file)
    config = Config(config_dict=config_dict)

  data_dir = args.data_dir

  trainer = Trainer(config=config, output_dir=args.output_dir, data_dir=data_dir)

  trainer.fit()

