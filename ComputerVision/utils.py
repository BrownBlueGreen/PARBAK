import os
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import RTDetrV2ForObjectDetection, AutoImageProcessor

MODEL_NAME = "PekingU/rtdetr_v2_r50vd"

def detection_raw_collate_fn(batch):
  """
  Collate function for object detection

  since images have different sizes and each image can have varying number of 
  annotations, don't stack them yet 

  returns:
    images: list of images
    targets: list of target dicts
  """

  images = [ item[0] for item in batch ]
  targets = [ item[1] for item in batch ]

  return images, targets

def detection_collate_fn(batch):
  """
  Collate function for object detection

  since images have different sizes and each image can have varying number of 
  annotations, don't stack them yet 

  returns:
    images: list of images
    targets: list of target dicts
  """
  image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
  images = [item[0] for item in batch]
  targets = [item[1] for item in batch]

  encoding = image_processor(
    images=images,
    annotations=targets,
    return_tensors="pt"
  )

  return {
    "pixel_values": encoding["pixel_values"],
    "pixel_mask": encoding.get("pixel_mask"),
    "labels": encoding["labels"],
  }

class DatasetInterface(Dataset):
  def __init__(self, coco: dict, img_dir: str):
    self.images = coco["images"]
    self.img_dir = img_dir
    self.anno_map = {ann["image_id"]: ann for ann in coco["annotations"]}
    self.id2label = {cat["id"]: cat["name"] for cat in coco["categories"]}
    self.label2id = {cat["name"]: cat["id"] for cat in coco["categories"]}

  def __len__(self) -> int:
    return len(self.images)
  
  def __getitem__(self, idx: int):
    image_info = self.images[idx]
    image_id = image_info["id"]

    image_path = os.path.join(self.img_dir, image_info["file_name"])
    image = Image.open(image_path).convert("RGB")

    target = {
      "image_id": image_id,
      "annotations": self.ann_map.get(image_id, []),
    }

    return image, target

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
    self.cout += n
    self.avg = self.sum / self.count if self.count > 0 else 0.0

  def __str__(self):
    return f"{self.name}: val={self.val:.4f}, avg={self.avg:.4f}"
  




