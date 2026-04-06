from utils import download_data
import numpy as np
from PIL import Image, ImageDraw
import fiftyone as fo
import json
import os 
import albumentations as A
from data_interface import DatasetInterface
from augmentation import build_train_augmentations

if __name__ == "__main__":
  # train, val = download_data()

  # print(train.info)

  # train.export(
  #   export_dir="/tmp/coco_train",
  #   dataset_type=fo.types.COCODetectionDataset,
  # )

  # val.export(
  #     export_dir="/tmp/coco_val",
  #     dataset_type=fo.types.COCODetectionDataset,
  # )

  with open("/tmp/coco_train/labels.json", "r") as f:
    train_coco = json.load(f)

  with open("/tmp/coco_val/labels.json", "r") as f:
    val_coco = json.load(f)

  # print("Train keys")
  # for k in train_coco:
  #   print(k)


  # print(train_coco["annotations"][2])
  img_dir = "/tmp/coco_train/data"

  ds_train_tr = DatasetInterface(
    train_coco, 
    img_dir,
    # transforms=build_train_augmentations(image_size=640),
  )

  image, target = ds_train_tr[582]
  anns = target["annotations"]

  # draw = ImageDraw.Draw(image)
  # for a in anns:
  #   x, y, w, h = a["bbox"]
  #   draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
  #   draw.text((x, y), ds_train_tr.id_2_label[a["category_id"]], fill="white")
  image.show()

  # pick one image
  img_info = train_coco["images"][582]
  image_id = img_info["id"]

  # open image
  img_path = os.path.join(img_dir, img_info["file_name"])
  image = Image.open(img_path).convert("RGB")
  image.show()
  # image_np = np.array(image)

  # # gather annotations for this image
  # anns = [ann for ann in train_coco["annotations"] if ann["image_id"] == image_id]

  # bboxes = [ann["bbox"] for ann in anns]                  # COCO: [x, y, w, h]
  # class_labels = [ann["category_id"] for ann in anns]    # must align with bboxes

  # train_augmentation_and_transform = A.Compose(
  #   [
  #     A.Perspective(p=0.25),
  #     A.HorizontalFlip(p=0.5),
  #     A.RandomBrightnessContrast(p=0.25),
  #     A.HueSaturationValue(p=0.4),
  #   ],
  #   bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25, min_width=1, min_height=1),
  # )

  # # to make sure boxes are clipped to image size and there is no boxes with area < 1 pixel
  # validation_transform = A.Compose(
  #   [A.NoOp()],
  #   bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1, min_width=1, min_height=1),
  # )
  
  # output = train_augmentation_and_transform(image=np.array(image), bboxes=bboxes, category=class_labels)

  # image = Image.fromarray(output["image"])
  # categories, boxes = output["category"], output["bboxes"]

  # draw = ImageDraw.Draw(image)
  # for category, box in zip(categories, boxes):
  #   x, y, w, h = box
  #   draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
  #   draw.text((x, y), id2label[category], fill="white")

  # image.show()



