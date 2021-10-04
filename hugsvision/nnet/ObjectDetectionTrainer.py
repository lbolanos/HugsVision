# -*- coding: utf-8 -*-

import os
import math
import random
import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torchmetrics
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from sklearn.metrics import precision_recall_fscore_support as f_score

from hugsvision.models.Detr import Detr
from hugsvision.dataio.CocoDetectionDataset import CocoDetection
from hugsvision.dataio.ObjectDetectionCollator import ObjectDetectionCollator
from hugsvision.inference.ObjectDetectionInference import ObjectDetectionInference
from hugsvision.utils.InferencePlot import InferencePlot
from hugsvision.datasets import get_coco_api_from_dataset
from hugsvision.datasets.coco_eval import CocoEvaluator
from tqdm.notebook import tqdm

from transformers import DetrFeatureExtractor

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('ObjectDetectionTrainer')
logger.setLevel(logging.DEBUG)
print = logger.info
class Logger(object):
    def __init__(self, std):
        self.terminal = std

    def write(self, message):
        self.terminal.write(message)
        logger.info(message)

    def flush(self):
        pass

sys.stdout = Logger(sys.stdout)
sys.stderr = Logger(sys.stderr)

class ObjectDetectionTrainer:

  """
  🤗 Constructor for the DETR Object Detection trainer
  """
  def __init__(
    self,
    model_name :str,
    train_path :str,
    dev_path   :str,
    test_path  :str,    
    output_dir :str,
    lr           = 1e-4,
    lr_backbone  = 1e-5,
    batch_size   = 4,
    max_epochs   = 1,
    shuffle      = True,
    augmentation = False,
    weight_decay = 1e-4,
    max_steps    = 10000,
    nbr_gpus     = -1,
    model_path   = "facebook/detr-resnet-50",
    num_workers  = None,
    save_pth = False,
    coco_file = "_annotations.coco.json"
  ):

    self.model_name        = model_name
    self.train_path        = train_path
    self.dev_path          = dev_path
    self.test_path         = test_path
    self.output_dir        = os.path.join(output_dir, self.model_name.upper())
    self.lr                = lr
    self.lr_backbone       = lr_backbone
    self.batch_size        = batch_size
    self.max_epochs        = max_epochs
    self.shuffle           = shuffle
    self.augmentation      = augmentation
    self.weight_decay      = weight_decay
    self.max_steps         = max_steps
    self.nbr_gpus          = nbr_gpus
    self.model_path        = model_path
    self.num_workers       = num_workers


    # Processing device (CPU / GPU)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup the metric
    self.metric = torchmetrics.Accuracy()

    # Load feature extractor
    self.feature_extractor = DetrFeatureExtractor.from_pretrained(self.model_path)
    
    # Get the classifier collator
    self.collator = ObjectDetectionCollator(self.feature_extractor)


    # Get the model output path
    self.output_path = self.__getOutputPath()
    self.logs_path   = self.output_path
    # Open the logs file
    self.__openLogs()

    print(f"model_name={model_name} "
          f"lr={lr} "
          f"lr_backbone={lr_backbone} "
          f"batch_size={batch_size} "
          f"max_epochs={max_epochs} "
          f"shuffle={shuffle} "
          f"augmentation={augmentation} "
          f"weight_decay={weight_decay} "
          f"max_steps={max_steps} "
          f"nbr_gpus={nbr_gpus} "
          f"model_path={model_path} "
          f"num_workers={num_workers} "
          f"train_path={train_path} "
          f"dev_path={dev_path} "
          f"test_path={test_path} "
          f"coco_file={coco_file} "
          f"output_dir={self.output_dir} ")

    # Split and convert to dataloaders
    self.train, self.dev, self.test = self.__splitDatasets(self.num_workers, coco_file=coco_file)

    self.sample_train()

    # Get labels and build the id2label

    # print("*"*100)
    categories = self.train_dataset.coco.dataset['categories']
    self.id2label = {}
    self.label2id = {}
    for category in categories:
        self.id2label[category['id']] = category['name']
        self.label2id[category['name']] = category['id']

    print(self.id2label)
    print(self.label2id)
    
    labels_path=os.path.join(self.output_path,"labels.json")
    str_content = json.dumps(self.id2label, ensure_ascii=False, sort_keys=False, indent='\t')
    with open(labels_path, "w", encoding="utf8") as writer:
        writer.write(str_content)
    
    
    """
    🏗️ Build the Model
    """
    self.model = Detr(
        lr               = self.lr,
        lr_backbone      = self.lr_backbone,
        weight_decay     = self.weight_decay,
        id2label         = self.id2label,
        label2id         = self.label2id,
        train_dataloader = self.train_dataloader,
        val_dataloader   = self.val_dataloader,
        model_path       = self.model_path,
    )

    """
    🏗️ Build the trainer
    """
    self.trainer = Trainer(
        gpus              = self.nbr_gpus,
        max_epochs        = self.max_epochs,
        max_steps         = self.max_steps,
        default_root_dir  = self.output_path,
        gradient_clip_val = 0.1
    )

    print("Trainer builded!")

    """
    ⚙️ Train the given model on the dataset
    """
    print("Start Training!")
    
    # Fine-tuning
    self.trainer.fit(self.model)

    if save_pth:
        torch.save(self.trainer.model.state_dict(), self.output_path + ".pth")
    else:
        # Save for huggingface
        self.model.model.save_pretrained(self.output_path)
    print("Model saved at: \033[93m" + self.output_path + "\033[0m")

    # Close the logs file
    self.logs_file.close()

  """
  📜 Open the logs file
  """
  def __openLogs(self):
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Open the logs file
    self.logs_file = open(self.logs_path + "/logs.txt", "a")
    # create file handler which logs even debug messages
    fh = logging.FileHandler(self.logs_path + "/logs.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

  """
  📍 Get the path of the output model
  """
  def __getOutputPath(self):

    path = os.path.join(
      self.output_dir,
        str(self.max_epochs if self.max_epochs else self.max_steps) + "_" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    )

    # Create the full path if doesn't exist yet
    if not os.path.isdir(path):
        os.makedirs(path)

    return path
    
  """
  ✂️ Split the dataset into sub-datasets
  """
  def __splitDatasets(self, workers=None, coco_file="_annotations.coco.json"):

    print("Load Datasets...")
    
    # Train Dataset in the COCO format
    self.train_dataset = CocoDetection(
        img_folder        = self.train_path,
        feature_extractor = self.feature_extractor,
        coco_file         = coco_file
    )

    # Dev Dataset in the COCO format
    self.val_dataset = CocoDetection(
        img_folder        = self.dev_path,
        feature_extractor = self.feature_extractor,
        coco_file         = coco_file
    )

    # Test Dataset in the COCO format
    self.test_dataset = CocoDetection(
        img_folder        = self.test_path,
        feature_extractor = self.feature_extractor,
        coco_file         = coco_file
    )

    print(self.train_dataset)
    print(self.val_dataset)
    print(self.test_dataset)

    if workers is None:
        workers = int(os.cpu_count() * 0.75)

    # Train Dataloader
    self.train_dataloader = DataLoader(
        self.train_dataset,
        collate_fn  = self.collator,
        batch_size  = self.batch_size,
        shuffle     = self.shuffle,
        num_workers = workers,
    )

    # Validation Dataloader
    self.val_dataloader = DataLoader(
        self.val_dataset,
        collate_fn  = self.collator,
        batch_size  = self.batch_size,
        num_workers = workers,
    )

    # Test Dataloader
    self.test_dataloader = DataLoader(
        self.test_dataset,
        collate_fn  = self.collator,
        batch_size  = self.batch_size,
        num_workers = workers,
    )

    return self.train_dataloader, self.val_dataloader, self.test_dataloader

  def sample_train(self):
      from PIL import Image, ImageDraw
      import numpy as np
      label2color = [ (0, 165, 255),
                      (36, 99, 154),
                      (225, 105, 65),
                      (196, 228, 255),
                      (0, 0, 255),
                     (255, 0, 0),
                      (0, 128, 0),
                     (128, 0, 0),
                      (0, 255, 0),  # 51, 153, 255
                      (25, 25, 112),
                      (0, 255, 255),
                      (0, 145, 255),
                      (195, 255, 170),
                      (128, 128, 0),
                      (0, 204, 255),
                      (255, 0, 191),
                      (128, 0, 128),
                      (0, 0, 255),
                      (192, 192, 192)
                     ]
      image_ids = self.train_dataset.coco.getImgIds()
      # let's pick a random image
      image_id = image_ids[np.random.randint(0, len(image_ids))]
      print('Image n°{}'.format(image_id))
      image = self.train_dataset.coco.loadImgs(image_id)[0]
      image = Image.open(os.path.join(self.train_path, image['file_name']))

      annotations = self.train_dataset.coco.imgToAnns[image_id]
      if image.mode == "L":
          image = image.convert('RGB')
      draw = ImageDraw.Draw(image, "RGBA")

      cats = self.train_dataset.coco.cats
      id2label = {k: v['name'] for k, v in cats.items()}

      for annotation in annotations:
          box = annotation['bbox']
          class_idx = annotation['category_id']
          x, y, w, h = tuple(box)
          draw.rectangle((x, y, x + w, y + h), outline=label2color[class_idx], width=2)
          draw.text((x, y), id2label[class_idx], fill='white')
      out_sample_dir = os.path.join(self.output_path, "out_img")
      if not os.path.isdir(out_sample_dir):
          os.makedirs(out_sample_dir)
      image.save(os.path.join(out_sample_dir, "sample_" + str(image_id) + ".jpg"))

  """
  🧪 Evaluate the performances of the system of the test sub-dataset
  """
  def evaluate(self):
    base_ds = get_coco_api_from_dataset(self.test_dataset)

    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(base_ds, iou_types)  # initialize evaluator with ground truths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.model.to(device)
    self.model.eval()

    print("Running evaluation...")

    for idx, batch in enumerate(tqdm(self.test_dataloader)):
      # get the inputs
      pixel_values = batch["pixel_values"].to(device)
      pixel_mask = batch["pixel_mask"].to(device)
      labels = [{k: v.to(device) for k, v in t.items()} for t in
                batch["labels"]]  # these are in DETR format, resized + normalized

      # forward pass
      outputs = self.model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

      orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
      results = self.feature_extractor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api
      res = {target['image_id'].item(): output for target, output in zip(labels, results)}
      coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return False

  """
  🧪 Test on a single image
  """
  def testing(self, img_path, show_tag=True, show_confidence=True, show_tags=None):

    inference = ObjectDetectionInference(
      self.feature_extractor,
      self.model
    )

    image, probas, bboxes_scaled = inference.predict(img_path)
    plot_inf = InferencePlot(inference.id2label,
                             IMG_OUT=os.path.join(self.output_path, "out_img"),
                             show_tag=show_tag, show_confidence=show_confidence, show_tags=show_tags)
    plot_inf.plot_results(image, probas, bboxes_scaled)
    return image, probas, bboxes_scaled