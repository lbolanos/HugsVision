import argparse
import logging
import os
import sys

from hugsvision.inference.ObjectDetectionInference import ObjectDetectionInference
from hugsvision.nnet.ObjectDetectionTrainer import ObjectDetectionTrainer
from hugsvision.utils.InferencePlot import InferencePlot

from MedicalNLU import config

here = config.get_config_static('data_folder', os.path.dirname(__file__))
MODEL_STORE_DIR = config.get_config_static('MODEL_STORE_DIR', os.path.dirname(__file__))
huggingface_model = "facebook/detr-resnet-50"
# huggingface_model = "facebook/detr-resnet-101"

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--name', type=str, required=True,
                    help='The name of the model and The name of the folder in data_folder')
parser.add_argument('--model', type=str, default=huggingface_model,
                    help='The name of the pretrained model')
parser.add_argument('--train', type=str, default=None,
                    help='The directory of the train folder containing the _annotations.coco.json')
parser.add_argument('--dev', type=str, default=None,
                    help='The directory of the dev folder containing the _annotations.coco.json')
parser.add_argument('--test', type=str, default=None,
                    help='The directory of the test folder containing the _annotations.coco.json')
parser.add_argument('--output', type=str, default=None, help='The output directory of the model')
parser.add_argument('--epochs', type=int, default=None, help='Number of Epochs')
parser.add_argument('--max_steps', type=int, default=None, help='Max Number of steps')
parser.add_argument('--nbr_gpus', type=int, default=1, help='Number of GPUs')
parser.add_argument('--mode', type=str, default="train", help='mode train eval or test')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.  If GPU then > 1')
parser.add_argument('--test_jpg', type=str, default="test.jpg", help='JPG to test')
args = parser.parse_args()

if not args.train:
    args.train = os.path.join(here, args.name, "train")
if not args.dev:
    args.dev = os.path.join(here, args.name, "dev")
if not args.test:
    args.test = os.path.join(here, args.name, "test")

if not args.output:
    args.output = MODEL_STORE_DIR
if not os.path.exists(args.dev):
    args.dev = args.test

file_handler = logging.FileHandler(filename=os.path.join(args.output, args.name.upper(), 'logs.log'))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

if args.mode != "test":
    # Train the model
    trainer = ObjectDetectionTrainer(
        model_name=args.name,
        output_dir=args.output,

        train_path=args.train,
        dev_path=args.dev,
        test_path=args.test,

        model_path=args.model,

        max_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        nbr_gpus=args.nbr_gpus,
        num_workers=0
    )
    # Test on a single image
    trainer.testing(img_path=args.test_jpg)
    trainer.evaluate()
else:
    import os, glob

    last_model = max(glob.glob(os.path.join(args.output, args.name.upper(), '*/')), key=os.path.getmtime)
    inference = ObjectDetectionInference(
        model_path=args.model,
        model_file_path=last_model
    )
    image, probas, bboxes_scaled = inference.predict(img_path=args.test_jpg, threshold=0.72)
    # plot results
    plot_inf = InferencePlot(inference.id2label,
                             IMG_OUT=os.path.join(last_model, "out_img"),
                             show_tag=False, show_confidence=False, show_tags=['checked'], show_plot=True)
    plot_inf.plot_results(image, probas, bboxes_scaled)
