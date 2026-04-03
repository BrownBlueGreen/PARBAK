from __future__ import annotations

import albumentations as A


def build_train_augmentations(image_size: int | None = None) -> A.Compose:
  """
  Augmentations for object detection training.

  Notes:
  - bbox format is COCO: [x_min, y_min, width, height]
  - label_fields tells Albumentations which lists must stay aligned with boxes
  - min_visibility helps remove boxes that become too truncated after transforms
  """

  transforms = [
    A.HorizontalFlip(p=0.5),

    A.Affine(
      scale=(0.90, 1.10),
      translate_percent=(-0.08, 0.08),
      rotate=(-12, 12),
      shear=(-5, 5),
      p=0.7,
    ),

    A.RandomSizedBBoxSafeCrop(
      height=image_size if image_size is not None else 640,
      width=image_size if image_size is not None else 640,
      erosion_rate=0.1,
      p=0.25,
    ) if image_size is not None else A.NoOp(),

    A.OneOf(
      [
        A.RandomBrightnessContrast(
          brightness_limit=0.20,
          contrast_limit=0.20,
          p=1.0,
        ),
        A.HueSaturationValue(
          hue_shift_limit=10,
          sat_shift_limit=15,
          val_shift_limit=10,
          p=1.0,
        ),
        A.CLAHE(
          clip_limit=2.0,
          tile_grid_size=(8, 8),
          p=1.0,
        ),
      ],
      p=0.6,
    ),

    A.OneOf(
      [
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
        A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
        A.ImageCompression(quality_range=(50, 95), p=1.0),
      ],
      p=0.35,
    ),
  ]

  return A.Compose(
    transforms,
    bbox_params=A.BboxParams(
      format="coco",
      label_fields=["class_labels", "ann_ids", "iscrowd"],
      min_visibility=0.25,
      clip=True,
    ),
  )


def build_val_augmentations() -> A.Compose:
  return A.Compose(
    [],
    bbox_params=A.BboxParams(
      format="coco",
      label_fields=["class_labels", "ann_ids", "iscrowd"],
      clip=True,
    ),
  )