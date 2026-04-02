import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset

class DatasetInterface(Dataset):
  def __init__(self, coco:dict, img_dir: str):
    self.images: List[Dict[str, Any]] = coco["images"]
    self.image_dir: str = img_dir
    self.categories: List[Dict[str, Any]] = coco.get("categories", [])
    
    self.ann_map: Dict[int, List[Dict[str, Any]]] = self._build_annotation_map(
      coco.get("annotations", [])
    )

    categories = sorted(coco["categories"], key=lambda c: c["id"])

    self.orig_id_to_train_id = {
        cat["id"]: i for i, cat in enumerate(categories)
    }

    self.id_2_label = {
        i: cat["name"] for i, cat in enumerate(categories)
    }

    self.label_2_id = {
        cat["name"]: i for i, cat in enumerate(categories)
    }
    # self.id_2_label: Dict[int, str] = {
    #   cat["id"]: cat["name"] for cat in self.categories
    # }
    # self.label_2_id: Dict[str, int] = {
    #   cat["name"]: cat["id"] for cat in self.categories
    # }
    self.image_id_2_info: Dict[int, Dict[str, Any]] = {
      img["id"]: img for img in self.images
    }

  def __len__(self) -> int:
    return len(self.images)
    
  def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
    image_info = self.images[idx]
    image_id = image_info["id"]

    image_path = os.path.join(self.image_dir, image_info["file_name"])
    image = Image.open(image_path).convert("RGB")

    raw_annotations = self.ann_map.get(image_id, [])
    remapped_annotations = []

    for ann in raw_annotations:
        remapped_ann = ann.copy()
        remapped_ann["category_id"] = self.orig_id_to_train_id[ann["category_id"]]
        remapped_annotations.append(remapped_ann)

    target = {
        "image_id": image_id,
        "annotations": remapped_annotations,
    }

    return image, target
  
  # def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
  #   image_info = self.images[idx]
  #   image_id = image_info["id"]

  #   image_path = os.path.join(self.image_dir, image_info["file_name"])
  #   image = Image.open(image_path).convert("RGB")

  #   target = {
  #     "image_id": image_id,
  #     "annotations": self.ann_map.get(image_id, []),
  #   }

  #   return image, target

  @staticmethod
  def _build_annotation_map(annotations: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    ann_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
      ann_map[ann["image_id"]].append(ann)
    return dict(ann_map)

  # GETTERS
  def get_image_info(self, idx: int) -> Dict[str, Any]:
    return self.images[idx]
  
  def get_image_info_by_id(self, image_id: int) -> Optional[Dict[str, Any]]:
    return self.image_id_2_info.get(image_id)
  
  def get_image_id(self, idx: int) -> int:
    return self.images[idx]["id"]
  
  def get_image_path(self, idx:int) -> str:
    image_info = self.images[idx]
    return os.path.join(self.image_dir, image_info["file_name"])

  def get_annotations_by_idx(self, idx: int) -> List[Dict[str, Any]]:
    image_id = self.get_image_id(idx)
    return self.ann_map.get(image_id, [])

  def get_annotations_by_image_id(self, image_id: int) -> List[Dict[str, Any]]:
    return self.ann_map.get(image_id, [])

  def get_label_name(self, category_id: int) -> Optional[str]:
    return self.id_2_label.get(category_id)

  def get_label_id(self, label_name: str) -> Optional[int]:
    return self.label_2_id.get(label_name)

  def get_num_classes(self) -> int:
    return len(self.id_2_label)

  def get_all_category_ids(self) -> List[int]:
    return list(self.id_2_label.keys())

  def get_all_category_names(self) -> List[str]:
    return list(self.label_2_id.keys())
  
  # @property
  # def get_id_2_labels(self) -> Dict[int, str]:
  #   return self.id_2_label
  
  # @property
  # def get_label_2_id(self) -> Dict[int, str]:
  #   return self.label_2_id
  
  # SETTERS
  def set_img_dir(self, img_dir: str) -> None:
    self.image_dir = img_dir  

  def set_label_maps(self, id2label: Dict[int, str], label2id: Dict[str, int]) -> None:
    self.id_2_label = id2label
    self.label_2_id = label2id

  def rebuild_annotation_map(self, annotations: List[Dict[str, Any]]) -> None:
    self.ann_map = self._build_annotation_map(annotations)


