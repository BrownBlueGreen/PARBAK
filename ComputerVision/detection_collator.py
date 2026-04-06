from typing import Any, Dict, List, Tuple
from PIL import Image
from transformers import AutoImageProcessor

class HFObjectDetectionCollator:
  def __init__(self, image_processor):
      self.image_processor = image_processor

  def __call__(self, batch):
    images = []
    annotations = []

    for image, target in batch:
        images.append(image)
        annotations.append(target)

    encoding = self.image_processor(
        images=images,
        annotations=annotations,
        return_tensors="pt",
    )

    batch_out = {
        "pixel_values": encoding["pixel_values"],
        "labels": encoding["labels"],
    }

    if "pixel_mask" in encoding:
        batch_out["pixel_mask"] = encoding["pixel_mask"]

    return batch_out