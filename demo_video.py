"""Image demo script."""
import argparse
import os
import sys

import cv2
import ffmpeg
from ffmpeg._probe import probe
import numpy as np
import torch
import random

from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer

from utils import get_golf_dicts
from utils import get_club_pos

def get_video_info(in_file):
    print(in_file)
    probe2 = probe(in_file)
    video_stream = next((stream for stream in probe2['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found', file=sys.stderr)
        sys.exit(1)
    return video_stream


parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--video-name',
                    help='video name',
                    default='./front/Driver/front_driver1.MP4',
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='./out/',
                    type=str)

opt = parser.parse_args()

video_basename = os.path.basename(opt.video_name).split('.')[0]

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_images')):
    os.makedirs(os.path.join(opt.out_dir, 'res_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_2d_images')):
    os.makedirs(os.path.join(opt.out_dir, 'res_2d_images'))

info = get_video_info(opt.video_name)
bitrate = info['bit_rate']
os.system(f'ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.jpg')


files = os.listdir(f'{opt.out_dir}/raw_images')
files.sort()

img_path_list = []
# set cfg
cfg = get_cfg()
golf_metadata = MetadataCatalog.get("golf_train")
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) # "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (Ball, Club, Person)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

cfg2 = get_cfg()
cfg2.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg2.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg2.DATALOADER.NUM_WORKERS = 2
cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
predictor = DefaultPredictor(cfg)
predictor2 = DefaultPredictor(cfg2)

filenames = os.listdir(os.path.join('data/golf_image/Association/Female/Swing',))
dataset_dicts = get_golf_dicts(filenames)

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:

        img_path = os.path.join(opt.out_dir, 'raw_images', file)
        img_path_list.append(img_path)

prev_box = None

print('### Run Model...')
idx = 0
x = 0 
x2 = 0 
y = 0 
y2 = 0 
for img_path in tqdm(img_path_list):
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)

    # Run Detection
    input_image = cv2.imread(img_path)
    outputs = predictor(input_image)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs2 = predictor2(input_image)

    # Visualization
    v = Visualizer(input_image[:, :, ::-1],
                metadata=golf_metadata, 
                scale=0.8, 
    )
    v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # find club idx of boxes
    pred_classes = outputs["instances"].to("cpu").pred_classes.numpy()
    pred_boxes = outputs["instances"].to("cpu").pred_boxes
    pred_keypoints = outputs2["instances"].pred_keypoints.to("cpu").numpy()
    
    get_club_pos(pred_classes, pred_boxes, pred_keypoints,v)

    out = v.get_output()

    v2 = Visualizer(out.get_image()[:,:,::-1], MetadataCatalog.get(cfg2.DATASETS.TRAIN[0]), scale=1.5)
    v2.draw_instance_predictions(outputs2["instances"].to("cpu"))

    out = v2.get_output()
    image = input_image.copy()

    idx += 1
    res_path = os.path.join(opt.out_dir, 'res_images', f'image-{idx:06d}.jpg')
    cv2.imwrite(res_path,out.get_image()[:, :, ::-1])

os.system(f"ffmpeg -r 25 -i ./{opt.out_dir}/res_images/image-%06d.jpg -vcodec mpeg4 -b:v {bitrate} ./{opt.out_dir}/res_{video_basename}.mp4")