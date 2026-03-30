import os, time
import json
import torch
from pathlib import Path
from utils import download_data
from data_interface import DatasetInterface
from detection_collator import DetectionCollator
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoModelForObjectDetection
from utils import set_seed, get_device, AverageMeter 
import fiftyone as fo
from PIL import Image, ImageDraw, ImageFont


COCO_DIR = "/tmp/coco"
DATA_DIR = os.path.join(COCO_DIR, "data")
# Directory structure
COCO_TRAIN_DIR = "/../coco/train"
COCO_VAL_DIR   = "/../coco/val"

TRAIN_DATA_DIR   = os.path.join(COCO_TRAIN_DIR, "data")
VAL_DATA_DIR     = os.path.join(COCO_VAL_DIR, "data")

TRAIN_LABELS_PATH = os.path.join(COCO_TRAIN_DIR, "labels.json")
VAL_LABELS_PATH   = os.path.join(COCO_VAL_DIR, "labels.json")
MODEL_NAME = "PekingU/rtdetr_v2_r50vd"

class Trainer:
  def __init__(self, config, output_dir=None, device=None):
    self.config = config
    self.num_workers = self.config.train.num_workers
    self.batch_size = self.config.train.batch_size
    self.lr = self.config.train.lr
    self.n_epochs = self.config.train.n_epochs
    
    self.val_split = getattr(self.config.train, "val_split", 0.2)
    self.seed = getattr(self.config.train, "seed", 42)
    self.checkpoint = getattr(self.config.train, "checkpoint", 1)
    self.vis_every = getattr(self.config.train, "vis", 1)
    self.num_fixed_eval_samples = getattr(self.config.train, "num_fixed_eval_samples", 4)

    set_seed(self.seed)
    
    self.data_dir = self.config.data.data_directory

    self.device = device if device is not None else get_device()

    if output_dir is None:
      self.output_dir = Path(f"./outputs/{self.config.network.model.lower()}")
    else:
      self.output_dir = output_dir

    # Make the appropriate 
    self.output_dir.mkdir(parents=True, exist_ok=True)
    (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (self.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    # 1. Download, export, and initialise interfaces
    # train_data, val_data = download_data()

    # train_data.export(
    #   export_dir=COCO_TRAIN_DIR,
    #   dataset_type=fo.types.COCODetectionDataset,
    # )
    # val_data.export(
    #   export_dir=COCO_VAL_DIR,
    #   dataset_type=fo.types.COCODetectionDataset,
    # )

    with open(TRAIN_LABELS_PATH, "r") as f:
      train_coco = json.load(f)
    with open(VAL_LABELS_PATH, "r") as f:
      val_coco = json.load(f)

    self.train_dataset = DatasetInterface(train_coco, TRAIN_DATA_DIR)
    self.val_dataset   = DatasetInterface(val_coco,   VAL_DATA_DIR)
    # OLD
    # 1. Download the dataset, export it in COCO format and create initialize interface
    # data = download_data()
    # data.export(
    #   export_dir="/tmp/coco",
    #   dataset_type=fo.types.COCODetectionDataset,
    # )

    # with open(LABELS_PATH, "r") as f:
    #   coco = json.load(f)

    # self.dataset = DatasetInterface(coco, DATA_DIR)

    # # 2. Create training and testing split
    # val_len = max(1, int(len(self.dataset) * self.val_split))
    # train_len = self.dataset.len - val_len

    # generator = torch.Generator().manual_seed(self.seed)
    # self.train_dataset, self.val_dataset = random_split(
    #   self.dataset,
    #   [train_len, val_len],
    #   generator=generator
    # )

    # 3. Set up the model
    assert self.train_dataset.id_2_label == self.val_dataset.id_2_label, \
      "Train and val label maps don't match — check your TARGET_CLASSES"

    # 3. Set up the model — use train_dataset arbitrarily, both are equivalent
    self.model = AutoModelForObjectDetection.from_pretrained(
      MODEL_NAME,
      id2label=self.train_dataset.id_2_label,
      label2id=self.train_dataset.label_2_id,
      ignore_mismatched_sizes=True,
    )
    self.model.to(self.device)
    
    # 4. torch DataLoader and collator

    collator_fn = DetectionCollator(MODEL_NAME)
    self.train_loader = DataLoader(
      self.train_dataset, 
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      collate_fn=collator_fn,
      pin_memory=(self.device == "cuda"),
    )

    self.val_loader = DataLoader(
      self.val_dataset, 
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      collate_fn=collator_fn,
      pin_memory=(self.device == "cuda"),
    )
    
    # 5. Optimizer
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    # Fixed samples for qualitative evaluation
    self.fixed_eval_samples = self.get_fixed_samples(self.val_dataset, self.num_fixed_eval_samples)
    self.best_val_loss = float("inf")

  def get_fixed_samples(self, dataset, n_samples=8, start_idx=100):
    n_samples = min(n_samples, len(dataset))
    return [dataset[i] for i in range(start_idx, n_samples)]
  
  def _move_labels_to_device(self, labels, device):
    moved = []
    for label_dict in labels:
      moved_item = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v, in label_dict.items()
      }
      moved.append(moved_item)
    return moved

  def save_checkpoint(self, epoch, val_loss=None, is_best=False):
    ckpt_dir = self.output_dir / "checkpoints"
    checkpoint = {
      "epoch": epoch,
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "val_loss": val_loss,
      "id_2_label": self.dataset.id_2_label,
      "label_2_id": self.dataset.label_2_id,
      "model_name": MODEL_NAME,
    }
  
    latest_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
    torch.save(checkpoint, latest_path)

    # Also save hugging-face style weights/config just in case 
    hf_dir = ckpt_dir / f"hf_epoch_{epoch + 1}"
    hf_dir.mkdir(parents=True, exist_ok=True)
    self.model.save_pretrained(hf_dir)

    if is_best:
      best_path = ckpt_dir / "best.pt"
      torch.save(checkpoint, best_path)
    
      best_hf_dir = ckpt_dir / "best_hf"
      best_hf_dir.mkdir(parents=True, exist_ok=True)
      self.model.save_pretrained(best_hf_dir)
    
    print(f"checkpoint saved: {latest_path}")

  def load_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.best_val_loss = checkpoint.get("val_loss", float("inf"))
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint.get("epoch", -1)

  def train_one_epoch(self, epoch, debug=False):
    self.model.train()

    loss_meter = AverageMeter()
    iter_meter = AverageMeter()

    epoch_start = time.time()

    all_ids = set()

    if debug:
      for img in self.train_dataset.images:
        anns = self.train_dataset.ann_map.get(img["id"], [])
        for ann in anns:
          all_ids.add(ann["category_id"])

      print("min label:", min(all_ids))
      print("max label:", max(all_ids))
      print("num unique labels:", len(all_ids))
      print("sorted labels:", sorted(all_ids))
      for i, batch in enumerate(self.train_loader):
        # DEBUG — remove after fixing
        print("pixel_values shape:", batch["pixel_values"].shape)
        for j, label in enumerate(batch["labels"]):
            print(f"  label[{j}] keys:", label.keys())
            for k, v in label.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}, min={v.min():.3f}, max={v.max():.3f}")
        break  # only check first batch

    for i, batch in enumerate(self.train_loader):
      start = time.time()
      pixel_values = batch["pixel_values"].to(self.device)
      pixel_mask = batch.get("pixel_mask")
      if pixel_mask is not None:
        pixel_mask = pixel_mask.to(self.device)

      labels = self._move_labels_to_device(batch["labels"], self.device)

      outputs = self.model(
        pixel_values = pixel_values,
        pixel_mask = pixel_mask,
        labels = labels,
      )

      loss = outputs.loss
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      batch_size = pixel_values.size(0)
      loss_meter.update(loss.item(), batch_size)
      iter_meter.update(time.time() - start)

      if i % 100 == 0:
        print(
          f"Train Epoch: [{epoch + 1}/{self.n_epochs}]"
          f"[{i}/{len(self.train_loader)}]\t"
          f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
          f"Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})"
        )
    
    print(
      f"Train epoch {epoch + 1} completed in "
      f"{(time.time() - epoch_start) / 60:.3f} min. "
      f"Avg loss: {loss_meter.avg:.4f}"
    )
    return loss_meter.avg

  def evaluate(self, epoch):
    self.model.eval()

    loss_meter = AverageMeter()
    iter_meter = AverageMeter()

    epoch_start = time.time()

    map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    with torch.no_grad():
      for i, batch in enumerate(self.val_loader):
        start = time.time()

        pixel_values = batch["pixel_values"].to(self.device)

        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(self.device)

        labels = self._move_labels_to_device(batch["labels"], self.device)

        outputs = self.model(
          pixel_values=pixel_values,
          pixel_mask=pixel_mask,
          labels=labels,
        )

        loss = outputs.loss
        batch_size = pixel_values.size(0)

        loss_meter.update(loss.item(), batch_size)
        iter_meter.update(time.time() - start)

        # Convert model outputs to absolute-image detections for mAP.
        # HF object detection processors expose post_process_object_detection
        # for this step.
        target_sizes = torch.stack([
            target["orig_size"].to(self.device) for target in labels
        ])

        predictions = self.train_loader.collate_fn.image_processor.post_process_object_detection(
          outputs,
          threshold=0.0,
          target_sizes=target_sizes,
        )

        # TorchMetrics expects:
        # pred: list[{"boxes": Tensor[N, 4], "scores": Tensor[N], "labels": Tensor[N]}]
        # targets: list[{"boxes": Tensor[M, 4], "labels": Tensor[M]}]

        metric_preds = []
        metric_targets = []

        for pred, target in zip(predictions, labels):
          metric_preds.append({
            "boxes": pred["boxes"].detach().cpu(),
            "scores": pred["scores"].detach().cpu(),
            "labels": pred["labels"].detach().cpu(),
          })

          metric_targets.append({
            "boxes": target["boxes"].detach().cpu(),
            "labels": target["class_labels"].detach().cpu(),
          })
        
        map.update(metric_preds, metric_targets)

        if i % 100 == 0:
          print(
            f"Val Epoch: [{epoch + 1}][{i}/{len(self.val_loader)}]\t"
            f"Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t"
            f"Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t"
          )
    
    results = map.compute()

    metrics = {
      "val_loss": float(loss_meter.avg),
      "mAP": float(results["map"].item()),
      "AP50": float(results["map_50"].item()),
      "AP75": float(results["map_75"].item()),
    }

    print(
      f"Val completed in {(time.time() - epoch_start) / 60:.3f} min. "
      f"Loss {metrics['val_loss']:.3f} | "
      f"mAP {metrics['mAP']:.4f} | "
      f"AP50 {metrics['AP50']:.4f} | "
      f"AP75 {metrics['AP75']:.4f}"
    )

    return metrics

  def visualize_fixed_predictions(self, epoch, threshold=0.3):
    self.model.eval()

    vis_dir = self.output_dir / "visualizations"
    font = ImageFont.load_default()

    with torch.no_grad():
      for idx, (image, target) in enumerate(self.fixed_eval_samples):
        # Build a single-item batch using the collator
        batch = self.train_loader.collator([(image, target)])

        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
          pixel_mask = pixel_mask.to(self.device)
        
        outputs = self.model(
          pixel_values=pixel_values,
          pixel_mask=pixel_mask,
        )

        # Convert tensor/PIL image for drawing
        if isinstance(image, torch.Tensor):
          img = self.tensor_to_pil(image)
        elif isinstance(image, Image.Image):
          img = image.copy()
        else:
          raise TypeError(f"Unsupported image type: {type(image)}")
        
        draw = ImageDraw.Draw(img)

        # Process with collator post processor
        processor = getattr(self.collator, "image_processor", None)
        if processor is not None and hasattr(processor, "post_process_object_detection"):
          target_sizes = torch.tensor([[img.height, img.width]], device=self.device)
          results = processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes
          )[0]

          boxes = results["boxes"].detach().cpu()
          scores = results["scores"].detach().cpu()
          labels = results["labels"].detach().cpu()

          for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = box.tolist()
            class_name = self.dataset.id_2_label.get(int(label), str(int(label)))
            text = f"{class_name}: {float(score):.2f}"

            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, max(0, y0-12)), text, fill="red", font=font)

        save_path = vis_dir / f"epoch_{epoch + 1}_sample_{idx}.png"
        img.save(save_path)
      
    print(f"Saved qualitative predictions for epoch {epoch + 1}")

  @staticmethod
  def tensor_to_pil(image_tensor):
    "Convert CHW tensor in [0, 1] or [0, 255] to PIL image"
    image = image_tensor.detach().cpu()

    if image.ndim != 3:
      raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")

    if image.dtype != torch.uint8:
      image = image.clamp(0, 1)
      image = (image * 255).byte()
    
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)
  
  def fit(self):
    print(f"Training on device: {self.device}")

    for epoch in range(self.n_epochs):
      train_loss = self.train_one_epoch(epoch)
      metrics = self.evaluate(epoch)
      val_loss = metrics["val_loss"]
      is_best = val_loss < self.best_val_loss
      self.save_checkpoint(epoch, val_loss=val_loss, is_best=is_best)

      if(epoch + 1) % self.vis_every == 0:
        self.visualize_fixed_predictions(epoch)

      print(
        f"Epoch {epoch + 1} / {self.n_epochs} done. "
        f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
        f"Best val loss: {self.best_val_loss:.4f}"
      )

    