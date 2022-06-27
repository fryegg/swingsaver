from tkinter import N, image_names
from unicodedata import category
import detectron2
import json
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import random
import os
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer


def get_golf_dicts(filenames): # json_dir, imgnames, idx
    dataset_dicts = []
    idx = 0
    for filename in filenames:
        imgnames = os.listdir(os.path.join("./data/golf_image/Association/Female/Swing",filename))
        for imgname in imgnames:
            record = {}
            json_repr = os.path.join("./data/golf_label/Association/Female/Swing",filename,os.path.splitext(imgname)[0]+'.json')
            img_repr = os.path.join("./data/golf_image/Association/Female/Swing",filename,imgname)
            json_file = os.path.join(json_repr)
            idx +=1
            try:
                with open(json_file) as f:
                        imgs_anns = json.load(f)
            except:
                json_file = None
            
            if json_file:
                
                record["file_name"] = img_repr
                record["image_id"] = idx
                record["width"] = imgs_anns["image"]["resolution"][0]
                record["height"] = imgs_anns["image"]["resolution"][1]
                
                
                annos = imgs_anns["annotations"]
                objs = []
                
                for anno in annos:
                    obj = {}
                    bbox_flag = True
                    seg_flag = True
                    # try:
                    #     bbox = anno["box"]
                    # except:
                    #     bbox = []

                    # try:
                    #     poly = anno["polygon"]
                    #     poly = anno["points"]
                    # except:
                    #     poly = []

                    if "box" in anno:
                        obj["bbox"] = anno["box"]
                        obj["bbox_mode"] = BoxMode.XYWH_ABS
                    elif "polygon" in anno:
                        poly_arr = np.array(anno["polygon"]).reshape(-1,2)
                        min_x = min(poly_arr[:,0]) #x
                        max_x = max(poly_arr[:,0]) #x
                        min_y = min(poly_arr[:,1]) #y
                        max_y = max(poly_arr[:,1]) #y
                        obj["bbox"] = [min_x, min_y, max_x-min_x, max_y-min_y]
                        obj["bbox_mode"] = BoxMode.XYWH_ABS
                    elif "points" in anno:
                        bbox_flag = False
                    else:
                        bbox_flag = False
                    

                    # try:
                    #     obj["segmentation"] = anno["polygon"]
                    #     obj["segmentation"] = anno["points"]
                    # except:
                    #     seg_flag = False

                    # category_id: person:1 club:2 ball:3
                    if anno["class"] == "person":
                        category_id = 1
                    elif anno["class"] == "club":
                        category_id = 2
                    elif anno["class"] == "ball":
                        category_id = 3
                    else:
                        pass

                    obj["category_id"] = category_id
                    

                    # obj = {
                    #     "bbox": bbox,
                    #     "bbox_mode": BoxMode.XYXY_ABS,
                    #     "segmentation": poly,
                    #     "category_id": category_id,
                    # }
                    if bbox_flag:
                        objs.append(obj)

                record["annotations"] = objs
                dataset_dicts.append(record)

    return dataset_dicts

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) # "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (Ball, Club, Person)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.OUTPUT_DIR = "./output/"

cfg2 = get_cfg()
cfg2.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg2.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg2.DATALOADER.NUM_WORKERS = 2
cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo

filenames = os.listdir(os.path.join('data/golf_image/Association/Female/Swing',))
train_mode = 0
dataset_name = []
#train

if train_mode==1:
    for d in ['train']:
        DatasetCatalog.register('golf_'+ d, lambda d=d : get_golf_dicts(filenames))
        MetadataCatalog.get('golf_'+ d)
        dataset_dicts = DatasetCatalog.get('golf_'+ d)
    cfg.DATASETS.TRAIN = ("golf_train",)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    #DetectionCheckpointer(trainer.model).load(file_path_or_url)  # load a file, usually from cfg.MODEL.WEIGHTS

    checkpointer = DetectionCheckpointer(trainer.model, save_dir="output")
    checkpointer.save("model_999")  # save to output/model_999.pth

#test
golf_metadata = MetadataCatalog.get("golf_train")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.DATASETS.TEST = ("golf_train", )
predictor = DefaultPredictor(cfg)
predictor2 = DefaultPredictor(cfg2)
dataset_dicts = get_golf_dicts(filenames)


im = cv2.imread("./test/front_driver1-000001.jpg")
outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
outputs2 = predictor2(im)
print(im)
v = Visualizer(im[:, :, ::-1],
                metadata=golf_metadata, 
                scale=0.8, 
)
v.draw_instance_predictions(outputs["instances"].to("cpu"))
#out = v.draw_dataset_dict(d)

head, tail = os.path.split("./test/front_driver1-000001.jpg")

# find club idx of boxes
pred_classes = outputs["instances"].to("cpu").pred_classes.numpy()
pred_boxes = outputs["instances"].to("cpu").pred_boxes
pred_keypoints = outputs2["instances"].pred_keypoints.to("cpu").numpy()

l_wrist = pred_keypoints[:,7,:][0]
r_wrist = pred_keypoints[:,8,:][0]

a = []

for clas, box in zip(pred_classes, pred_boxes):
    if clas == 2:
        x,y,x2,y2 = box.numpy()
        print("box",box.numpy())
        dis1 = np.sqrt((x-l_wrist[0])**2 + (y-l_wrist[1])**2)
        dis2 = np.sqrt((x2-l_wrist[0])**2 + (y2-l_wrist[1])**2)
        dis3 = np.sqrt((x-l_wrist[0])**2 + (y2-l_wrist[1])**2)
        dis4 = np.sqrt((x2-l_wrist[0])**2 + (y-l_wrist[1])**2)
        a = [dis1,dis2,dis3,dis4]
        print(a)
        a_min = min(a)
        a_max = max(a)

        if dis1 == a_min:
            min_ = [x,y]
            print("a")
        elif dis2 == a_min:
            min_ = [x2,y2]
            print("b")
        elif dis3 == a_min:
            min_ = [x,y2]
            print("c")
        elif dis4 == a_min:
            min_ = [x2,y]
            print("d")

        if dis1 == a_max:
            max_ = [x,y]
            print("e")
        elif dis2 == a_max:
            max_ = [x2,y2]
            print("f")
        elif dis3 == a_max:
            max_ = [x,y2]
            print("g")
        elif dis4 == a_max:
            max_ = [x2,y]
            print("h")

# find close with wrist
# "head","uvula","throax","left_shoulder", "right_shoulder","left_elbow", "right_elbow","left_wrist", "right_wrist","pelvis","right_hip", "left_hip","right_knee", "left_knee","right_ankle", "left_ankle"
# pred_keypoint 
print(min_,max_)
v.draw_line([x,x2],[y,y2],color=(1.0, 0, 0))
out = v.get_output()
print(out.get_image()[:, :, ::-1].shape)
cv2.imwrite(tail,out.get_image()[:, :, ::-1])
 

# for j, d in enumerate(random.sample(dataset_dicts, 5)):    
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=golf_metadata, 
#                    scale=0.8, 
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",d)
#     #out = v.draw_dataset_dict(d)
    
#     head, tail = os.path.split(d["file_name"])
#     cv2.imwrite(tail,out.get_image()[:, :, ::-1])