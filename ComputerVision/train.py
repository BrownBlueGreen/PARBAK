import os, torch, time, argparse, json
from data_interface import DatasetInterface
from detection_collator import HFObjectDetectionCollator
from torch.utils.data import DataLoader
import yaml
from utils import Config
# from model_trainer import Trainer
from collections import defaultdict
from augmentation import build_train_augmentations, build_val_augmentations
from transformers import Trainer, TrainingArguments, AutoImageProcessor, AutoModelForObjectDetection
from pprint import pprint
from map_evaluator import MAPEvaluator

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
  # parser.add_argument('--output_dir', type=str, default=None, help="path to output directory (optional); defaults to outputs/model_name")
  parser.add_argument('--data_dir', type=str, default=None, help="path to data directory")
  args = parser.parse_args()

  # Load YAML configuration
  with open(args.config_file, 'r') as file:
    config_dict = yaml.safe_load(file)
    config = Config(config_dict=config_dict)

  data_dir = args.data_dir
  
  train_labels_path = os.path.join(data_dir, "train/labels.json")
  train_img_dir = os.path.join(data_dir, "train/data")

  val_labels_path = os.path.join(data_dir, "val/labels.json")
  val_img_dir = os.path.join(data_dir, "val/data")

  print("data_dir:", data_dir)
  print("labels_path:", train_labels_path)
  print("image_dir:", train_img_dir)

  with open(train_labels_path, "r") as f:
    train_labels = json.load(f)
  with open(val_labels_path, "r") as f:
    val_labels = json.load(f)

  train_dataset = DatasetInterface(train_labels, train_img_dir, build_train_augmentations(image_size=640))
  val_dataset = DatasetInterface(val_labels, val_img_dir, build_train_augmentations(image_size=640))

  checkpoint = "PekingU/rtdetr_v2_r18vd"
  image_processor = AutoImageProcessor.from_pretrained(checkpoint)

  model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=train_dataset.id_2_label,
    label2id=train_dataset.label_2_id,
    ignore_mismatched_sizes=True,
  )

  data_collator = HFObjectDetectionCollator(image_processor=image_processor)

  training_args = TrainingArguments(
    output_dir="outputs/rtdetr",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    learning_rate=1e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_map",
  )

  eval_compute_metrics_fn = MAPEvaluator(
    image_processor=image_processor,
    threshold=0.01,
    id2label=train_dataset.id_2_label,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=image_processor,
    compute_metrics=eval_compute_metrics_fn,
  )

  # trainer.train()
  metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
  pprint(metrics)


