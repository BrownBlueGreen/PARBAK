import os, torch, time, argparse, json
from data_interface import DatasetInterface
from detection_collator import DetectionCollator
from torch.utils.data import DataLoader
import yaml
from utils import Config
from model_trainer import Trainer
from collections import defaultdict
from augmentation import build_train_augmentations, build_val_augmentations

def print_class_distribution(dataset):
    class_counts = defaultdict(int)

    for idx in range(len(dataset)):
        anns = dataset.get_remapped_annotations_by_idx(idx)

        for ann in anns:
            class_id = ann["category_id"]
            class_counts[class_id] += 1

    # Sort by count (descending)
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    print("Class distribution (sorted by frequency):\n")

    for class_id, count in sorted_items:
        class_name = dataset.get_label_name(class_id)
        print(f"{class_id:3d} | {class_name:20s} | {count}")

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

  # with open("/tmp/coco/train/labels.json", "r") as f:
  #   train = json.load(f)
  # with open("/tmp/coco/val/labels.json", "r") as f:
  #   val = json.load(f)

  # ds_train = DatasetInterface(coco=train, img_dir="/tmp/coco/train/data", transforms=build_train_augmentations(image_size=640))
  # ds_val = DatasetInterface(coco=val, img_dir="/tmp/coco/val/data", transforms=build_val_augmentations())

  # print("Training dataset length:", len(ds_train))
  # print("Validation dataset lenth:", len(ds_val))

  trainer = Train(config=config, data_dir=data_dir)
  trainer.fit()
  # print_class_distribution(ds_train)

  # MODEL_NAME = "PekingU/rtdetr_v2_r18vd"
  # collator_fn = DetectionCollator(MODEL_NAME)

  # train_loader = DataLoader(
  #   ds_train, 
  #   batch_size=4,
  #   shuffle=True,
  #   num_workers=1,
  #   collate_fn=collator_fn,
  #   pin_memory = False,
  # )

  # for batch in train_loader:
  #   label = batch["labels"][0]
  #   print(label["boxes"][:5])

  #   print(label.keys())
  #   break


