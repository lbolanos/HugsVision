import json
import os
import torch
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForObjectDetection, pipeline

from hugsvision.models.InferenceDetr import InferenceDetr


class ObjectDetectionInference:

  """
  🤗 Constructor for the object detection trainer
  """
  def __init__(self, feature_extractor = None, model=None, model_file_path=None, model_file_name="pytorch_model.bin",
               model_path="facebook/detr-resnet-50", is_pth=False):

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if feature_extractor is None:
      self.feature_extractor = DetrFeatureExtractor.from_pretrained(model_path)
    else:
      self.feature_extractor = feature_extractor
    if model is None:
      model_file = os.path.join(model_file_path, model_file_name)
      if is_pth:
        labels_path = os.path.join(model_file_path, "labels.json")
        with open(labels_path, "r", encoding="utf8") as f:
          str_content = f.read()
          self.id2label = json.loads(str_content)
        self.model = InferenceDetr(model_path=model_path, id2label=id2label)
        self.model.load_state_dict(torch.load(model_file))
      else:
        self.model = InferenceDetr(model_path=model_file_path)
        self.id2label = self.model.model.config.id2label
    else:
      self.model = model
      self.id2label = self.model.model.config.id2label

    print("Model loaded!")



  # Bounding box post-processing
  def box_cxcywh_to_xyxy(self, x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

  def rescale_bboxes(self, out_bbox, size):
    img_w, img_h = size
    b = self.box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


  def visualize_predictions(self, image, outputs, threshold=0.2):

    # Get predictions probabilities
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    print(probas.max(-1).values)

    # Keep only predictions with confidence >= threshold
    keep = probas.max(-1).values > threshold
    print(keep)
    print(len(keep))

    # Convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = self.rescale_bboxes(
      outputs.pred_boxes[0,keep].cpu(),
      image.size
    )

    print(bboxes_scaled)
    print(len(bboxes_scaled))

    return image, probas[keep], bboxes_scaled

  """
  ⚙️ Predict the bounding boxes for each object in the image
  Return: image, probas, bboxes_scaled
  """
  def predict(self, img_path: str, threshold=0.2):
    print(f"predict img_path={img_path}")
    # Load the image
    image_array = Image.open(img_path)

    # # Change resolution to 128x128
    # image_array.thumbnail((self.resolution,self.resolution))

    # Transform the image
    encoding = self.feature_extractor(
      images=image_array,
      return_tensors="pt"
    )

    # Predict and get the corresponding bounding boxes
    outputs = self.model(
      pixel_values=encoding['pixel_values'],
      pixel_mask=None,
    )
    return self.visualize_predictions(image_array, outputs, threshold)
