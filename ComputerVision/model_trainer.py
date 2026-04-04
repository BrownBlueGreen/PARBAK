import fiftyone as fo, math, json, os, random, time, torch
from augmentation import build_train_augmentations, build_val_augmentations
from data_interface import DatasetInterface
from detection_collator import DetectionCollator
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from utils import set_seed, get_device, AverageMeter 
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForObjectDetection

MODEL_NAME = "PekingU/rtdetr_v2_r18vd"

def get_fixed_sample_indices(dataset, n_samples=8, start_idx=0):
    end_idx = min(start_idx + n_samples, len(dataset))
    return list(range(start_idx, end_idx))


def _move_labels_to_device(labels, device):
    moved = []
    for label_dict in labels:
        moved_item = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in label_dict.items()
        }
        moved.append(moved_item)
    return moved
        
class Trainer:
  def __init__(self, config, output_dir=None, device=None, data_dir=None):
    self.config = config
    self.data_dir = data_dir if data_dir is not None else self.config.data.data_directory
    # self.model_type = getattr(self.config.model, "type", "rtdetr")

    self.batch_size = getattr(self.config.train, "batch_size", 1)
    self.n_epochs = getattr(self.config.train, "n_epochs", 1)
    self.num_workers = getattr(self.config.train, "num_workers", 1)
    self.seed = getattr(self.config.train, "seed", 42)
    self.checkpoint_every = max(1, getattr(self.config.train, "checkpoint", 1))
    self.vis_every = max(1, getattr(self.config.train, "vis", 10))
    self.num_fixed_eval_samples = getattr(self.config.train, "num_fixed_eval_samples", 4)
    self.unfreeze_epoch = getattr(self.config.train, "unfreeze_epoch", 3)

    self.grad_clip_norm = getattr(self.config.train, "grad_clip_norm", 1.0)
    self.use_amp = bool(getattr(self.config.train, "use_amp", True))

    self.frozen = True
    self.device = device if device is not None else get_device()
    self.output_dir = Path(output_dir) if output_dir is not None else Path("./outputs")

    self.output_dir.mkdir(parents=True, exist_ok=True)
    (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (self.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    self.amp_enabled = self.use_amp and str(self.device).startswith("cuda")
    self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    train_dir = os.path.join(self.data_dir, "train")
    val_dir = os.path.join(self.data_dir, "val")

    train_data_dir = os.path.join(train_dir, "data")
    val_data_dir = os.path.join(val_dir, "data")
    train_labels_path = os.path.join(train_dir, "labels.json")
    val_labels_path = os.path.join(val_dir, "labels.json")

    with open(train_labels_path, "r") as f:
        train = json.load(f)
    with open(val_labels_path, "r") as f:
        val = json.load(f)

    self.train_dataset = DatasetInterface(
      coco=train,
      img_dir=train_data_dir,
      transforms=build_train_augmentations(image_size=640),
    )

    self.val_dataset = DatasetInterface(
      coco=val,
      img_dir=val_data_dir,
      transforms=build_val_augmentations(),
    )

    assert self.train_dataset.id_2_label == self.val_dataset.id_2_label, (
      "Train and val label maps don't match — check your TARGET_CLASSES"
    )

    self.model = AutoModelForObjectDetection.from_pretrained(
      MODEL_NAME,
      id2label=self.train_dataset.id_2_label,
      label2id=self.train_dataset.label_2_id,
      ignore_mismatched_sizes=True,
    )
    self.model.to(self.device)

    collator_fn = DetectionCollator(MODEL_NAME)

    self.train_loader = DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      collate_fn=collator_fn,
      pin_memory=str(self.device).startswith("cuda"),
    )

    self.val_loader = DataLoader(
      self.val_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      collate_fn=collator_fn,
      pin_memory=str(self.device).startswith("cuda"),
    )

    self.freeze_backbone()
    self.optimizer = self.build_optimizer()
    self.scheduler = self.build_scheduler(current_epoch=0)

    self.fixed_eval_indices = get_fixed_sample_indices(
      self.val_dataset,
      n_samples=self.num_fixed_eval_samples,
    )

    self.best_val_loss = float("inf")
    self.best_map = float("-inf")

  def save_checkpoint(self, epoch, metrics=None, is_best_map=False):
    ckpt_dir = self.output_dir / "checkpoints"
    metrics = metrics or {}

    checkpoint = {
      "epoch": epoch,
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "scheduler_state_dict": self.scheduler.state_dict(),
      "best_val_loss": self.best_val_loss,
      "best_map": self.best_map,
      "frozen": self.frozen,
      "unfreeze_epoch": self.unfreeze_epoch,
      "metrics": metrics,
      "id_2_label": self.train_dataset.id_2_label,
      "label_2_id": self.train_dataset.label_2_id,
      "model_name": MODEL_NAME,
    }

    if self.amp_enabled:
      checkpoint["sacler_state_dict"] = self.scaler.state_dict()

    latest_path = ckpt_dir / f"epoch_{epoch + 1}.pt"
    torch.save(checkpoint, latest_path)

    hf_dir = ckpt_dir / f"hf_epoch_{epoch + 1}"
    hf_dir.mkdir(parents=True, exist_ok=True)
    self.model.save_pretrained(hf_dir)

    if is_best_map:
      best_path = ckpt_dir / "best_map.pt"
      torch.save(checkpoint, best_path)

      best_hf_dir = ckpt_dir / "best_map_hf"
      best_hf_dir.mkdir(parents=True, exist_ok=True)
      self.model.save_pretrained(best_hf_dir)

    print(f"checkpoint saved: {latest_path}")

  def load_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    self.model.load_state_dict(checkpoint["model_state_dict"])

    self.unfreeze_epoch = checkpoint.get("unfreeze_epoch", 3)
    self.frozen = checkpoint.get("frozen", True)

    if self.frozen:
        self.freeze_backbone()
    else:
        self.unfreeze_all()

    self.optimizer = self.build_optimizer()

    saved_epoch = checkpoint.get("epoch", -1)
    next_epoch = saved_epoch + 1
    self.scheduler = self.build_scheduler(current_epoch=next_epoch)

    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint and self.amp_enabled:
      self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    self.best_map = checkpoint.get("best_map", float("-inf"))

    print(f"Loaded checkpoint from {checkpoint_path}")
    return next_epoch

  def train_one_epoch(self, epoch):
    self.model.train()

    loss_meter = AverageMeter()
    iter_meter = AverageMeter()
    epoch_start = time.time()

    if self.frozen and epoch >= self.unfreeze_epoch:
      print(f"Unfreezing full model at epoch {epoch + 1}")
      self.unfreeze_all()
      self.frozen = False
      self.optimizer = self.build_optimizer()
      self.scheduler = self.build_scheduler(current_epoch=epoch)

    for i, batch in enumerate(self.train_loader):
      start = time.time()

      pixel_values = batch["pixel_values"].to(self.device)
      pixel_mask = batch.get("pixel_mask")
      if pixel_mask is not None:
        pixel_mask = pixel_mask.to(self.device)

      labels = _move_labels_to_device(batch["labels"], self.device)

      self.optimizer.zero_grad(set_to_none=True)

      with torch.amp.autocast("cuda", enabled=self.amp_enabled):
        outputs = self.model(
          pixel_values=pixel_values,
          pixel_mask=pixel_mask,
          labels=labels,
        )
        loss = outputs.loss

      if self.amp_enabled:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
      else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

      self.scheduler.step()

      batch_size = pixel_values.size(0)
      loss_meter.update(loss.item(), batch_size)
      iter_meter.update(time.time() - start)

      if i % 100 == 0:
        lrs = [group["lr"] for group in self.optimizer.param_groups]
        print(
          f"Train Epoch: [{epoch + 1}/{self.n_epochs}] "
          f"[{i}/{len(self.train_loader)}] "
          f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
          f"Time {iter_meter.val:.3f} ({iter_meter.avg:.3f}) "
          f"LRs {[f'{lr:.2e}' for lr in lrs]}"
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

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    with torch.no_grad():
      for i, batch in enumerate(self.val_loader):
        start = time.time()

        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
          pixel_mask = pixel_mask.to(self.device)

        labels = _move_labels_to_device(batch["labels"], self.device)

        with torch.amp.autocast("cuda", enabled=self.amp_enabled):
          outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
          )
          loss = outputs.loss

        batch_size = pixel_values.size(0)
        loss_meter.update(loss.item(), batch_size)

        target_sizes = torch.stack([target["orig_size"] for target in labels])

        predictions = self.val_loader.collate_fn.image_processor.post_process_object_detection(
          outputs,
          threshold=0.1,
          target_sizes=target_sizes,
        )

        metric_preds = []
        metric_targets = []

        for pred, target in zip(predictions, labels):
          metric_preds.append({
            "boxes": pred["boxes"].detach().cpu(),
            "scores": pred["scores"].detach().cpu(),
            "labels": pred["labels"].detach().cpu(),
          })

          gt_boxes = target["boxes"]
          gt_labels = target["class_labels"]

          gt_boxes_xyxy = self._cxcywh_to_xyxy(gt_boxes)

          h, w = target["orig_size"]
          scale = torch.tensor(
            [w, h, w, h],
            device=gt_boxes_xyxy.device,
            dtype=gt_boxes_xyxy.dtype,
          )
          gt_boxes_xyxy = gt_boxes_xyxy * scale

          metric_targets.append({
            "boxes": gt_boxes_xyxy.detach().cpu(),
            "labels": gt_labels.detach().cpu(),
          })

        metric.update(metric_preds, metric_targets)
        iter_meter.update(time.time() - start)

        if i % 100 == 0:
          print(
            f"Val Epoch: [{epoch + 1}/{self.n_epochs}] "
            f"[{i}/{len(self.val_loader)}] "
            f"Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) "
            f"Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})"
          )

    results = metric.compute()

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
    collator = self.val_loader.collate_fn
    processor = getattr(collator, "image_processor", None)

    with torch.no_grad():
      for idx, dataset_idx in enumerate(self.fixed_eval_indices):
        image, target = self.val_dataset[dataset_idx]
        batch = collator([(image, target)])

        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(self.device)

        with torch.amp.autocast("cuda", enabled=self.amp_enabled):
            outputs = self.model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
            )

        if isinstance(image, torch.Tensor):
          img = self.tensor_to_pil(image)
        elif isinstance(image, Image.Image):
          img = image.copy()
        else:
          raise TypeError(f"Unsupported image type: {type(image)}")

        draw = ImageDraw.Draw(img)

        if processor is not None and hasattr(processor, "post_process_object_detection"):
          target_sizes = torch.tensor([[img.height, img.width]], device=self.device)
          results = processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
          )[0]

          boxes = results["boxes"].detach().cpu()
          scores = results["scores"].detach().cpu()
          labels = results["labels"].detach().cpu()

          for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = box.tolist()
            class_name = self.val_dataset.id_2_label.get(int(label), str(int(label)))
            text = f"{class_name}: {float(score):.2f}"

            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, max(0, y0 - 12)), text, fill="red", font=font)

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
  
  def fit(self, start_epoch=0):
    print(f"Training on device: {self.device}")
    print(f"AMP enabled: {self.amp_enabled}")

    for epoch in range(start_epoch, self.n_epochs):
      train_loss = self.train_one_epoch(epoch)
      metrics = self.evaluate(epoch)

      val_loss = metrics["val_loss"]
      current_map = metrics["mAP"]

      if val_loss < self.best_val_loss:
        self.best_val_loss = val_loss

      is_best_map = current_map > self.best_map
      if is_best_map:
        self.best_map = current_map

      if ((epoch + 1) % self.checkpoint_every == 0) or is_best_map:
        self.save_checkpoint(epoch, metrics=metrics, is_best_map=is_best_map)

      if ((epoch + 1) % self.vis_every == 0):
        self.visualize_fixed_predictions(epoch)

      print(
        f"Epoch {epoch + 1}/{self.n_epochs} done. "
        f"Train loss: {train_loss:.4f}, "
        f"Val loss: {val_loss:.4f}, "
        f"mAP: {current_map:.4f}, "
        f"Best val loss: {self.best_val_loss:.4f}, "
        f"Best mAP: {self.best_map:.4f}"
      )

  def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [cx, cy, w, h] to [x0, y0, x1, y1].
    Assumes boxes shape is [N, 4].
    """
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)

  def visualize_random_val_prediction_with_gt(self, epoch=None, threshold=0.3, save_name=None):
    """
    Pick a random validation sample, run inference, and save a visualization
    with both ground-truth boxes and predicted boxes.

    Color convention:
      - Green: ground truth
      - Red: predictions

    Args:
        epoch (int | None):
            Optional epoch index, used in output filename.
        threshold (float):
            Score threshold for prediction post-processing.
        save_name (str | None):
            Optional custom filename.

    Returns:
        Path: path to the saved visualization
    """
    self.model.eval()

    vis_dir = self.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if len(self.val_dataset) == 0:
      raise ValueError("Validation dataset is empty.")

    sample_idx = random.randrange(len(self.val_dataset))
    image, target = self.val_dataset[sample_idx]

    collator = self.val_loader.collate_fn
    processor = getattr(collator, "image_processor", None)
    if processor is None:
        raise ValueError("Validation collate_fn does not have an image_processor.")

    batch = collator([(image, target)])

    pixel_values = batch["pixel_values"].to(self.device)
    pixel_mask = batch.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(self.device)

    labels = _move_labels_to_device(batch["labels"], self.device)

    with torch.no_grad():
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )

    if isinstance(image, torch.Tensor):
        img = self.tensor_to_pil(image)
    elif isinstance(image, Image.Image):
        img = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # -------- Predictions (red) --------
    target_sizes = torch.tensor([[img.height, img.width]], device=self.device)
    pred_results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    pred_boxes = pred_results["boxes"].detach().cpu()
    pred_scores = pred_results["scores"].detach().cpu()
    pred_labels = pred_results["labels"].detach().cpu()

    # -------- Ground truth (green) --------
    # batch["labels"] after collate/image_processor is usually in HF training format:
    #   boxes: normalized cxcywh
    #   class_labels: class ids
    gt = labels[0]

    gt_boxes = gt["boxes"]
    gt_labels = gt["class_labels"]

    # Convert normalized cxcywh -> absolute xyxy
    gt_boxes_xyxy = self._cxcywh_to_xyxy(gt_boxes)

    h, w = gt["orig_size"]
    scale = torch.tensor([w, h, w, h], device=gt_boxes_xyxy.device, dtype=gt_boxes_xyxy.dtype)
    gt_boxes_xyxy = (gt_boxes_xyxy * scale).detach().cpu()
    gt_labels = gt_labels.detach().cpu()

    # -------- Draw ground truth --------
    for box, label in zip(gt_boxes_xyxy, gt_labels):
        x0, y0, x1, y1 = box.tolist()
        class_id = int(label)

        if hasattr(self.val_dataset, "id_2_label"):
            class_name = self.val_dataset.id_2_label.get(class_id, str(class_id))
        elif hasattr(self.train_dataset, "id_2_label"):
            class_name = self.train_dataset.id_2_label.get(class_id, str(class_id))
        else:
            class_name = str(class_id)

        text = f"GT: {class_name}"

        draw.rectangle([x0, y0, x1, y1], outline="green", width=3)
        draw.text((x0, max(0, y0 - 12)), text, fill="green", font=font)

    # -------- Draw predictions --------
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x0, y0, x1, y1 = box.tolist()
        class_id = int(label)

        if hasattr(self.val_dataset, "id_2_label"):
            class_name = self.val_dataset.id_2_label.get(class_id, str(class_id))
        elif hasattr(self.train_dataset, "id_2_label"):
            class_name = self.train_dataset.id_2_label.get(class_id, str(class_id))
        else:
            class_name = str(class_id)

        text = f"Pred: {class_name} {float(score):.2f}"

        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, min(img.height - 12, y1)), text, fill="red", font=font)

    # -------- Save --------
    if save_name is None:
        if epoch is not None:
            save_name = f"epoch_{epoch + 1}_random_val_with_gt_{sample_idx}.png"
        else:
            save_name = f"random_val_with_gt_{sample_idx}.png"

    save_path = vis_dir / save_name
    img.save(save_path)

    print(f"Saved random validation visualization with GT to: {save_path}")
    print(f"Sample index: {sample_idx}")
    print(f"GT boxes: {len(gt_boxes_xyxy)}")
    print(f"Pred boxes kept: {len(pred_boxes)}")

    return save_path
  
  def build_optimizer(self):
    backbone_params = []
    pretrained_other_params = []
    new_head_params = []

    new_head_prefixes = [
      "model.denoising_class_embed",
      "model.enc_score_head",
      "model.decoder.class_embed",
    ]

    for name, param in self.model.named_parameters():
      if not param.requires_grad:
        continue

      if name.startswith("model.backbone."):
        backbone_params.append(param)
      elif any(name.startswith(prefix) for prefix in new_head_prefixes):
        new_head_params.append(param)
      else:
        pretrained_other_params.append(param)

    param_groups = []
    if backbone_params:
      param_groups.append({
        "params": backbone_params,
        "lr": self.config.optimizer.backbone_lr,
      })
    if pretrained_other_params:
      param_groups.append({
        "params": pretrained_other_params,
        "lr": self.config.optimizer.transformer_lr,
      })
    if new_head_params:
      param_groups.append({
        "params": new_head_params,
        "lr": self.config.optimizer.head_lr,
      })

    return torch.optim.AdamW(
      param_groups,
      weight_decay=self.config.optimizer.weight_decay,
    )

  def freeze_backbone(self):
    for name, param in self.model.named_parameters():
      if "backbone" in name:
        param.requires_grad = False

  def unfreeze_all(self):
    for param in self.model.parameters():
      param.requires_grad = True

  def build_scheduler(self, current_epoch=0):
    steps_per_epoch = len(self.train_loader)

    if self.frozen and self.unfreeze_epoch > 0:
      phase_epochs = min(self.unfreeze_epoch, self.n_epochs)
      total_steps = max(1, steps_per_epoch * phase_epochs)
    else:
      remaining_epochs = max(1, self.n_epochs - current_epoch)
      total_steps = max(1, steps_per_epoch * remaining_epochs)

    warmup_steps = int(0.05 * total_steps)
    return self.build_warmup_cosine_scheduler(
      warmup_steps=warmup_steps,
      total_steps=total_steps,
    )

  def build_warmup_cosine_scheduler(self, warmup_steps, total_steps):
    def lr_lambda(current_step):
      if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))

      progress = float(current_step - warmup_steps) / float(
          max(1, total_steps - warmup_steps)
      )
      progress = min(max(progress, 0.0), 1.0)
      return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(self.optimizer, lr_lambda)






