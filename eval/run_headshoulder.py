import cv2
import torch
import numpy as np
import random
import os
import sys
import json
import time
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(cur_dir))
from dgdet.dataset.dataset_test import MyDataset
from dgdet.utils.transform import resize_sample,to_tensor
import cfgs.headshoulder_solver as base_solver
from importlib import import_module


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        return json.JSONEncoder.default(self, obj)



def main(cfg=None):
    if cfg is None:
        cfg = base_solver
    eval_thresh = 0.01
    vis_thresh = 0.5
    testset = cfg.data_config.test_list
    vis_flag = cfg.data_config.vis_flag  # True or False
    for test_data in testset:
        prex = test_data.split('/')[-1].split('.')[0]   # 'test1'   'test2'
        vis_path = cfg.data_config.vis_path+'/'+prex
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        result_json_path = cfg.data_config.js_dir+'/'+'result_'+prex+'.json'
        fjs = open(result_json_path, 'w')

        #define data
        set1 = MyDataset(cfg.data_config,test_data)

        #define model
        # model = cfg.Net(cfg.num_class)
        model = cfg.Net(cfg.num_class, cfg.nms_thres)
        model.eval()
        model.load_state_dict(torch.load(cfg.weight_path))
        model.cuda()

        nb_file = len(set1)
        for idx in range(nb_file):
            img_val = []
            sample = set1[idx]
            ann = sample['annot'].copy()
            img_path = sample['img_path']
            # tmp_name = sample['name']
            # print('-----tmp_name', tmp_name)  # error:_side_1125_images/CDC10_2020-11-25_16-00_18_82.jpg
            name = img_path.split('/')[-1]
            print(idx, name)
            img = sample['img'].copy()
            #preprocessing
            shape = cfg.infer_shape
            tmp = resize_sample(sample,shape)


            scale = tmp['scale']
            tmp = to_tensor(tmp,cfg.imagenet_pretrain)
            data = tmp['img']
            data = torch.unsqueeze(data,dim=0)
            data = data.cuda()
            #forward
            result = model(data)
            scores, classification, transformed_anchors = result
            idxs = np.where(scores.cpu() > eval_thresh)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                score = float(scores[j])
                clf = classification[j]  # 0  int()
                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                bbox = [x1,y1,x2,y2]
                # item = {'confidence': [score], 'data': bbox, 'tagnameid': tag_id[clf]}
                item = {'confidence': [score], 'data': bbox, 'tagnameid': str(int(clf))}
                img_val.append(item)
                if score>=vis_thresh:
                    text_info = str(int(clf))+'_'+str(round(score, 3))
                    cv2.putText(img, text_info, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            if vis_flag:
                save_path = vis_path+'/'+name
                cv2.imwrite(save_path, img)
            dt = {'image': img_path, 'results': img_val}
            fjs.writelines(json.dumps(dt, cls=NumpyEncoder) + '\n')

        


if __name__ == "__main__":
    sovler_path = sys.argv[1]          # cfgs/dangercar_solver.py
    solver = sovler_path.split('/')
    solver = '.'.join(solver)[:-3]
    solver = import_module(solver)   # import cfg.dangercar_solver as cfg
    main(solver)


    
