import pytorch_lightning as pl
from transformers import DetrForObjectDetection

class InferenceDetr(pl.LightningModule):

     def __init__(self, model_path="facebook/detr-resnet-50", id2label=None):
         super().__init__()
         # replace COCO classification head with custom head

         if id2label:
             num_labels = len(id2label)
             self.model = DetrForObjectDetection.from_pretrained(model_path,
                                                                 num_labels=num_labels,
                                                                 ignore_mismatched_sizes=True)
         else:
             self.model = DetrForObjectDetection.from_pretrained(model_path,
                                                                 ignore_mismatched_sizes=True)

         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896


     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
       return outputs
