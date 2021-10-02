import os
import json
from datetime import datetime

import torch
from transformers import DetrFeatureExtractor, DetrForObjectDetection, pipeline
from hugsvision.models.InferenceDetr import InferenceDetr
from PIL import Image

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

class ObjectDetectionInference:
    
  """
  🤗 Constructor for the object detection trainer
  """
  def __init__(self, feature_extractor = None, model=None, model_file_path=None, model_file_name="pytorch_model.bin",
               model_path="facebook/detr-resnet-50", IMG_OUT = "./out_img/", is_pth=False):

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
    self.IMG_OUT = IMG_OUT
    print("Model loaded!")
    
    # Colors for visualization
    self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

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

  def plot_results(self, pil_img, prob, boxes):
    #plt.figure(figsize=(10,16))
    my_dpi = 96
    img_w, img_h = pil_img.size
    plt.figure(figsize=(img_w / my_dpi, img_h / my_dpi), dpi=my_dpi)

    plt.imshow(pil_img)
    ax = plt.gca()

    print(self.id2label)

    colors = self.COLORS * 100

    # print(torchvision.ops.nms(
    #     boxes.cuda(),
    #     prob.cuda(),
    #     0.8,
    # ))

    # For each bbox
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):

        # Draw the bbox as a rectangle
        ax.add_patch(plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            color=c,
            linewidth=3
        ))

        # Get the highest probability
        cl = p.argmax()

        # Draw the label
        #text = f'{self.id2label[cl.item()]}: {p[cl]:0.2f}'
        #ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')

    if not os.path.exists(self.IMG_OUT):
      os.makedirs(self.IMG_OUT)

    file_name_jpg = os.path.join(self.IMG_OUT, datetime.today().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg")
    plt.savefig(file_name_jpg)
    print(f"Image Saved: {file_name_jpg}" )
    plt.show()

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

    # plot results
    self.plot_results(image, probas[keep], bboxes_scaled)

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
