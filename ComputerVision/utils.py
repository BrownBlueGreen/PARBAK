import os
import torch
import fiftyone as fo
import fiftyone.zoo as foz
import random
import numpy as np
from class_labels import TARGET_CLASSES

def download_data():
  train = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=TARGET_CLASSES,
    only_matching=True,
    max_samples=5000,
    shuffle=True,
    seed=42,
    dataset_name="oi_train",
  )
  val = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=TARGET_CLASSES,
    only_matching=True,
    max_samples=1000,
    shuffle=True,
    seed=42,
    dataset_name="oi_val",
  )
  return train, val

class AverageMeter:
  """
  Computes and stores current value, sums, averages, counts, etc...
  Useful for tracking batch and epoch loss
  """

  def __init__(self, name: str = "metric"):
    self.name = name
    self.reset()

  def reset(self):
    self.val = 0.0
    self.sum = 0.0
    self.count = 0
    self.avg = 0.0

  def update(self, val: float, n: int = 1):
    self.val = float(val)
    self.sum += float(val) * n
    self.count += n
    self.avg = self.sum / self.count if self.count > 0 else 0.0

  def __str__(self):
    return f"{self.name}: val={self.val:.4f}, avg={self.avg:.4f}"
  
class Config:
  def __init__(self, config_dict):
    for key, value in config_dict.items():
      if isinstance(value, dict):
        value = Config(value)
      setattr(self, key, value)

def get_device():
  # if you want to default to cuda first change order.
  if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS.")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA (gpu: {torch.cuda.get_device_name(0)}).")
  else:
    device = torch.device("cpu")
    print("Using CPU")
  return device

def compute_params(model):
  total = 0
  for param in model.parameters():
    total += param.numel()
  print(f'total parameters {total}')

def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  # print(f'set seed to {seed}')

def load_pt_data(load_path):
  if not os.path.exists(load_path):
      raise FileNotFoundError(f"dataset not found at {load_path}")
  
  data = torch.load(load_path)
  return data


