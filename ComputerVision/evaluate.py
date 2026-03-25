import torch
import requests

from PIL import Image

# Example evaluation code

# Get the data
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Get the model and image processor
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")

# Prepare input into the model 
inputs = image_processor(images=image, return_tensors="pt")

# We're not training, so use torch.no_grad() and pass input into the model
with torch.no_grad():
     outputs = model(**inputs)

# Post-process the results
results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

# Output results
for result in results:
     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
         score, label = score.item(), label_id.item()
         box = [round(i, 2) for i in box.tolist()]
         print(f"{model.config.id2label[label]}: {score:.2f} {box}")