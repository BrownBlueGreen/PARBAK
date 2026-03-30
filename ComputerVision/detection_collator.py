from typing import Any, Dict, List, Tuple
from PIL import Image
from transformers import AutoImageProcessor

class DetectionCollator:
  def __init__(self, model_name: str):
    self.image_processor = AutoImageProcessor.from_pretrained(model_name)

  def __call__(self, batch: List[Tuple[Image.Image, Dict[str, Any]]]) -> Dict[str, Any]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    encoding = self.image_processor(
      images = images,
      annotations = targets,
      return_tensors="pt"
    )

    return {
      "pixel_values": encoding["pixel_values"],
      "pixel_mask": encoding.get("pixel_mask"),
      "labels": encoding["labels"],
    }
  