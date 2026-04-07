from typing import Any, Dict, List, Tuple
from PIL import Image
from transformers import AutoImageProcessor
import torch

class HFObjectDetectionCollator:
  def __init__(self, image_processor):
    self.image_processor = image_processor

  def __call__(self, batch):
    processed_samples = []

    for image, target in batch:
      encoding = self.image_processor(
        images=image,
        annotations=target,
        return_tensors="pt",
      )

      sample = {
        "pixel_values": encoding["pixel_values"].squeeze(0),
        "labels": encoding["labels"][0],
      }

      if "pixel_mask" in encoding:
        sample["pixel_mask"] = encoding["pixel_mask"].squeeze(0)

      processed_samples.append(sample)

    data = {
      "pixel_values": torch.stack([x["pixel_values"] for x in processed_samples]),
      "labels": [x["labels"] for x in processed_samples],
    }

    if "pixel_mask" in processed_samples[0]:
      data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in processed_samples])

    return data