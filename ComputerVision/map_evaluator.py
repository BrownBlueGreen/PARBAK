import numpy as np
import torch
from dataclasses import dataclass
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:
    def __init__(self, image_processor, threshold=0.01, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def _collect_image_sizes(self, targets):
        image_sizes = []
        for batch in targets:
            # batch is usually a list of label dicts
            batch_sizes = []
            for x in batch:
                size = x["size"]
                size = torch.as_tensor(size)
                if size.ndim == 0:
                    raise ValueError(f"Invalid size field: {size}")
                if size.numel() != 2:
                    raise ValueError(f"Expected size to have 2 values (h, w), got {size}")
                batch_sizes.append(size)
            image_sizes.append(torch.stack(batch_sizes))
        return image_sizes

    def _collect_targets(self, targets, image_sizes):
        post_processed_targets = []

        for target_batch, size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, size_batch):
                # HF object detection labels are expected here as normalized cxcywh
                height, width = size.tolist()

                boxes = torch.as_tensor(target["boxes"])
                boxes = center_to_corners_format(boxes)
                boxes = boxes * torch.tensor([width, height, width, height], dtype=boxes.dtype)

                labels = torch.as_tensor(target["class_labels"], dtype=torch.int64)

                post_processed_targets.append({
                    "boxes": boxes,
                    "labels": labels,
                })

        return post_processed_targets

    def _collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []

        for batch_preds, target_sizes in zip(predictions, image_sizes):
            # Common Trainer output layout for RT-DETR/DETR tutorials:
            # batch_preds[1] = logits, batch_preds[2] = pred_boxes
            logits = torch.as_tensor(batch_preds[1])
            pred_boxes = torch.as_tensor(batch_preds[2])

            output = ModelOutput(logits=logits, pred_boxes=pred_boxes)

            processed = self.image_processor.post_process_object_detection(
                output,
                threshold=self.threshold,
                target_sizes=target_sizes,
            )
            post_processed_predictions.extend(processed)

        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, eval_pred):
      predictions = eval_pred.predictions
      targets = eval_pred.label_ids

      image_sizes = self._collect_image_sizes(targets)
      post_processed_targets = self._collect_targets(targets, image_sizes)
      post_processed_predictions = self._collect_predictions(predictions, image_sizes)

      metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
      metric.warn_on_many_detections = False
      metric.update(post_processed_predictions, post_processed_targets)
      metrics = metric.compute()

      classes = metrics.pop("classes")
      map_per_class = metrics.pop("map_per_class")
      mar_100_per_class = metrics.pop("mar_100_per_class")

      for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_id = int(class_id.item())
        class_name = self.id2label[class_id] if self.id2label is not None else str(class_id)
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

      return {
        k: round(v.item(), 4) if torch.is_tensor(v) else round(float(v), 4)
        for k, v in metrics.items()
      }