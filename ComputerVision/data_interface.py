import os
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset


class DatasetInterface(Dataset):
  def __init__(
      self,
      data_labels: dict[str, Any],
      img_dir: str,
      transforms: Optional[A.Compose] = None,
  ):
    if "images" not in data_labels:
        raise ValueError("Data labels missing required key: 'images'")
    if "categories" not in data_labels:
        raise ValueError("Data labels missing required key: 'categories'")

    self.images: list[dict[str, Any]] = data_labels["images"]
    self.image_dir: str = img_dir
    self.categories: list[dict[str, Any]] = sorted(data_labels["categories"], key=lambda c: c["id"])
    self.transforms = transforms

    raw_annotations = data_labels.get("annotations", [])

    # Original COCO category id -> contiguous training id [0, N-1]
    self.orig_id_to_train_id: dict[int, int] = {
      cat["id"]: i for i, cat in enumerate(self.categories)
    }

    # Contiguous training id -> original COCO category id
    self.train_id_to_orig_id: dict[int, int] = {
      i: cat["id"] for i, cat in enumerate(self.categories)
    }

    # Contiguous training id -> label name
    self.id_2_label: dict[int, str] = {
      i: cat["name"] for i, cat in enumerate(self.categories)
    }

    # Label name -> contiguous training id
    self.label_2_id: dict[str, int] = {
      cat["name"]: i for i, cat in enumerate(self.categories)
    }

    # Original COCO category id -> label name
    self.orig_id_2_label: dict[int, str] = {
      cat["id"]: cat["name"] for cat in self.categories
    }

    self.image_id_2_info: dict[int, dict[str, Any]] = {
      img["id"]: img for img in self.images
    }

    self.ann_map: dict[int, list[dict[str, Any]]] = self._build_annotation_map(raw_annotations)

    self._validate_images()
    self._validate_annotations()

  def __len__(self) -> int:
    return len(self.images)

  def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
    if idx < 0 or idx >= len(self.images):
      raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")

    image_info = self.images[idx]
    image_id = image_info["id"]

    image_path = os.path.join(self.image_dir, image_info["file_name"])
    if not os.path.exists(image_path):
      raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    annotations = self.get_remapped_annotations_by_image_id(image_id)

    if self.transforms is not None:
      image, annotations = self._apply_transforms(
        image=image,
        image_id=image_id,
        annotations=annotations,
      )

    target = {
      "image_id": image_id,
      "annotations": annotations,
    }

    return image, target

  def _apply_transforms(
      self,
      image: Image.Image,
      image_id: int,
      annotations: list[dict[str, Any]],
  ) -> tuple[Image.Image, list[dict[str, Any]]]:
    image_np = np.array(image)

    bboxes: list[list[float]] = []
    class_labels: list[int] = []
    ann_ids: list[int | None] = []
    iscrowd: list[int] = []

    for ann in annotations:
      bbox = ann.get("bbox")
      if bbox is None or len(bbox) != 4:
        continue

      x, y, w, h = bbox
      if w <= 0 or h <= 0:
        continue

      bboxes.append([float(x), float(y), float(w), float(h)])
      class_labels.append(int(ann["category_id"]))
      ann_ids.append(ann.get("id"))
      iscrowd.append(int(ann.get("iscrowd", 0)))

    transformed = self.transforms(
      image=image_np,
      bboxes=bboxes,
      class_labels=class_labels,
      ann_ids=ann_ids,
      iscrowd=iscrowd,
    )

    transformed_image = Image.fromarray(transformed["image"])

    out_bboxes = transformed["bboxes"]
    out_labels = transformed["class_labels"]
    out_ann_ids = transformed["ann_ids"]
    out_iscrowd = transformed["iscrowd"]

    transformed_annotations: list[dict[str, Any]] = []

    for bbox, class_label, ann_id, crowd in zip(
      out_bboxes, out_labels, out_ann_ids, out_iscrowd
    ):
      x, y, w, h = bbox
      if w <= 0 or h <= 0:
        continue

      new_ann = {
        "image_id": image_id,
        "category_id": int(class_label),
        "bbox": [float(x), float(y), float(w), float(h)],
        "area": float(w * h),
        "iscrowd": int(crowd),
      }

      if ann_id is not None:
        new_ann["id"] = ann_id

      transformed_annotations.append(new_ann)

    return transformed_image, transformed_annotations

  @staticmethod
  def _build_annotation_map(annotations: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    ann_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
      if "image_id" not in ann:
        raise ValueError(f"Annotation missing required key 'image_id': {ann}")
      ann_map[ann["image_id"]].append(ann)
    return dict(ann_map)

  def _validate_images(self) -> None:
    seen_ids = set()

    for img in self.images:
      if "id" not in img:
        raise ValueError(f"Image entry missing required key 'id': {img}")
      if "file_name" not in img:
        raise ValueError(f"Image entry missing required key 'file_name': {img}")

      image_id = img["id"]
      if image_id in seen_ids:
        raise ValueError(f"Duplicate image id found: {image_id}")
      seen_ids.add(image_id)

  def _validate_annotations(self) -> None:
    valid_image_ids = set(self.image_id_2_info.keys())
    valid_category_ids = set(self.orig_id_to_train_id.keys())

    for image_id, anns in self.ann_map.items():
      if image_id not in valid_image_ids:
        raise ValueError(f"Annotation references unknown image_id {image_id}")

      for ann in anns:
        if "category_id" not in ann:
          raise ValueError(f"Annotation missing required key 'category_id': {ann}")

        cat_id = ann["category_id"]
        if cat_id not in valid_category_ids:
          raise ValueError(f"Annotation uses unknown category_id {cat_id} for image_id {image_id}")

  # ---------------------------
  # Annotation accessors
  # ---------------------------

  def get_raw_annotations_by_idx(self, idx: int) -> list[dict[str, Any]]:
    image_id = self.get_image_id(idx)
    return self.get_raw_annotations_by_image_id(image_id)

  def get_raw_annotations_by_image_id(self, image_id: int) -> list[dict[str, Any]]:
    return self.ann_map.get(image_id, [])

  def get_remapped_annotations_by_idx(self, idx: int) -> list[dict[str, Any]]:
    image_id = self.get_image_id(idx)
    return self.get_remapped_annotations_by_image_id(image_id)

  def get_remapped_annotations_by_image_id(self, image_id: int) -> list[dict[str, Any]]:
    raw_annotations = self.ann_map.get(image_id, [])
    remapped_annotations: list[dict[str, Any]] = []

    for ann in raw_annotations:
      orig_cat_id = ann["category_id"]
      if orig_cat_id not in self.orig_id_to_train_id:
        raise ValueError(
          f"Unknown category_id {orig_cat_id} in annotation for image_id {image_id}"
        )

      remapped_ann = ann.copy()
      remapped_ann["category_id"] = self.orig_id_to_train_id[orig_cat_id]
      remapped_annotations.append(remapped_ann)

    return remapped_annotations

  def get_annotations_by_idx(self, idx: int) -> list[dict[str, Any]]:
    return self.get_remapped_annotations_by_idx(idx)

  def get_annotations_by_image_id(self, image_id: int) -> list[dict[str, Any]]:
    return self.get_remapped_annotations_by_image_id(image_id)

  # ---------------------------
  # Image / metadata getters
  # ---------------------------

  def get_image_info(self, idx: int) -> dict[str, Any]:
    return self.images[idx]

  def get_image_info_by_id(self, image_id: int) -> Optional[dict[str, Any]]:
    return self.image_id_2_info.get(image_id)

  def get_image_id(self, idx: int) -> int:
    return self.images[idx]["id"]

  def get_image_path(self, idx: int) -> str:
    image_info = self.images[idx]
    return os.path.join(self.image_dir, image_info["file_name"])

  # ---------------------------
  # Label / category getters
  # ---------------------------

  def get_label_name(self, train_id: int) -> Optional[str]:
    return self.id_2_label.get(train_id)

  def get_label_id(self, label_name: str) -> Optional[int]:
    return self.label_2_id.get(label_name)

  def get_original_category_id(self, train_id: int) -> Optional[int]:
    return self.train_id_to_orig_id.get(train_id)

  def get_train_id_from_original_id(self, orig_category_id: int) -> Optional[int]:
    return self.orig_id_to_train_id.get(orig_category_id)

  def get_original_label_name(self, orig_category_id: int) -> Optional[str]:
    return self.orig_id_2_label.get(orig_category_id)

  def get_num_classes(self) -> int:
    return len(self.id_2_label)

  def get_all_train_ids(self) -> list[int]:
    return list(self.id_2_label.keys())

  def get_all_original_category_ids(self) -> list[int]:
    return list(self.orig_id_to_train_id.keys())

  def get_all_category_names(self) -> list[str]:
    return list(self.label_2_id.keys())

  # ---------------------------
  # Controlled mutation
  # ---------------------------

  def set_img_dir(self, img_dir: str) -> None:
    self.image_dir = img_dir

  def set_transforms(self, transforms: Optional[A.Compose]) -> None:
    self.transforms = transforms

  def rebuild_annotation_map(self, annotations: list[dict[str, Any]]) -> None:
    new_ann_map = self._build_annotation_map(annotations)

    old_ann_map = self.ann_map
    self.ann_map = new_ann_map
    try:
      self._validate_annotations()
    except Exception:
      self.ann_map = old_ann_map
      raise