import os
import json
import numpy as np
from detectron2.structures import BoxMode

def get_club_pos(pred_classes, pred_boxes, pred_keypoints, v):
    try:
        l_wrist = pred_keypoints[:,7,:][0]
        r_wrist = pred_keypoints[:,8,:][0]

        a = []

        for clas, box in zip(pred_classes, pred_boxes):
            if clas == 2:
                x,y,x2,y2 = box.numpy()
                dis1 = np.sqrt((x-l_wrist[0])**2 + (y-l_wrist[1])**2)
                dis2 = np.sqrt((x2-l_wrist[0])**2 + (y2-l_wrist[1])**2)
                dis3 = np.sqrt((x-l_wrist[0])**2 + (y2-l_wrist[1])**2)
                dis4 = np.sqrt((x2-l_wrist[0])**2 + (y-l_wrist[1])**2)
                a = [dis1,dis2,dis3,dis4]
                a_min = min(a)
                a_max = max(a)

                if dis1 == a_min:
                    min_ = [x,y]
                elif dis2 == a_min:
                    min_ = [x2,y2]
                elif dis3 == a_min:
                    min_ = [x,y2]
                elif dis4 == a_min:
                    min_ = [x2,y]


                if dis1 == a_max:
                    max_ = [x,y]
                elif dis2 == a_max:
                    max_ = [x2,y2]
                elif dis3 == a_max:
                    max_ = [x,y2]
                elif dis4 == a_max:
                    max_ = [x2,y]
                v.draw_line([x,x2],[y,y2],color=(1.0, 0, 0))
    except:
        pass

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
                    if bbox_flag:
                        objs.append(obj)

                record["annotations"] = objs
                dataset_dicts.append(record)

    return dataset_dicts
