import json
import torch
import os
import sys
from torch.utils.data import DataLoader
from utils import (DatasetInterface, 
                   AverageMeter, 
                   detection_collate_fn)


from class_labels import TARGET_CLASSES
import fiftyone as fo
import fiftyone.zoo as foz

COCO_DIR = "/tmp/coco"
IMG_DIR = os.path.join(COCO_DIR, "data")
LABELS_PATH = os.path.join(COCO_DIR, "labels.json")

def download_data():
  dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=TARGET_CLASSES,
    only_matching=True,   
    max_samples=5000,
    shuffle=True,
    seed=42,
    # dataset_name="oi_food_subset",
  )

  return dataset



def main():

  # Down the open image data 
  # images = download_data()
  
  # # Export in COCO format
  # images.export(
  #   export_dir="/tmp/coco",
  #   dataset_type=fo.types.COCODetectionDataset,
  # )

  with open(LABELS_PATH, "r") as f:
    coco = json.load(f)


  print(coco.keys())
  print(coco["info"])
  print("IMAGE:", coco["images"][0])
  print("CATEGORY:", coco["categories"][0])
  print("ANNOTATIONS:", coco["annotations"][0])

  # Build mapping for image id to annotations
  # ann_map = defaultdict(list)
  # for ann in coco["annotations"]:
  #   ann_map[ann["image_id"]].append(ann)


  # # Build category mappings so model knows, 0: apple, 1: pear, etc...
  # categories = coco["categories"]
  # id2label = {cat["id"]: cat["name"] for cat in categories}
  # label2id = {cat["name"]: cat["id"] for cat in categories}

  # model = RTDetrV2ForObjectDetection.from_pretrained(
  #   CHECKPOINT,
  #   id2label=id2label,
  #   label2id=label2id,
  #   ignore_mismatched_sizes=True
  # )

  # train_dataset = CoCoDetectionDataset(coco, IMG_DIR, ann_map)
  # train_loader = DataLoader(
  #   train_dataset,
  #   batch_size=4,
  #   shuffle=True,
  #   num_workers=0,
  #   collate_fn=collate_fn
  # )


  # # Minimum training loop
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # model.to(device)

  # optimizer = torch.optim_AdamW(model.parameters(), lr=5e-5)
  # model.train()

  # for epoch in range(3):
  #   for batch in train_loader:
  #     pixel_values = batch["pixel_values"].to(device)
  #     pixel_mask = batch["pixel_mask"]
  #     if pixel_mask is not None:
  #       pixel_mask = pixel_mask.to(device)
      
  #     labels = []
  #     for label in batch["labels"]:
  #       labels.append({
  #         k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in label.items()
  #       })

  #     outputs = model(
  #       pixel_values=pixel_values,
  #       pixel_mask=pixel_mask,
  #       labels=labels,
  #     )

  #     loss = outputs.loss
  #     optimizer.zero_grad()
  #     loss.backward()
  #     optimizer.step()

  #     print("loss:", loss.item())

  return 0

if __name__ == "__main__":
  sys.exit(main())

# url = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
# response = requests.get(url)

# classes = []
# for line in response.text.strip().split("\n"):
#     parts = line.split(",", 1)
#     if len(parts) == 2:
#         classes.append(parts[1].strip())

# print(classes)
# print(f"Total classes: {len(classes)}")