import os, sys
import cv2, json, time
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from importlib import import_module
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from xml2js_gt import Xml2js_gt

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(cur_dir))




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


def PR_curve(pt, result, labelname='default'):
    pt.axis([0, 1, 0, 1])
    pt.ylabel('Precision  TP/(TP+FP)')
    pt.xlabel('Recall  TP/(TP+FN)')
    pt.xticks(np.arange(0, 1, 0.1))
    pt.yticks(np.arange(0, 1, 0.1))
    precisions = []
    recalls = []

    default_th_pr = []
    for rs in result:
        precisions.append(rs['Precision'])
        recalls.append(rs['Recall'])
        if rs['threshold'] == 0.5:
            default_th_pr = (rs['Precision'], rs['Recall'])

    pt.plot(recalls, precisions, label=labelname)
    pt.plot(recalls, precisions, 'o')
    pt.plot((default_th_pr[1]), (default_th_pr[0]), '*')
    pt.legend()

def parse_gtfile_rec_raw(gt_json, thred=0.01): #all objs in a img
    if gt_json is None: return None
    recs = {}
    labels = {}
    with open(gt_json) as mfile:
        lines = mfile.readlines()
        for idx, line in enumerate(lines):
            sample = json.loads(line.strip())
            img_key = sample['image']
            objects = []
            for result in sample['results']:
                obj_struct = {}
                obj_struct['name'] = result["tagnameid"]
                labels[result["tagnameid"]] = 1
                obj_struct['difficult'] = result["difficult"] #in coco.py, all difficult=0
                obj = result["data"]
                obj_struct['bbox'] = [int(obj[0]), int(obj[1]),
                                    int(obj[2]), int(obj[3])]
                objects.append(obj_struct)
            recs[img_key] = objects
    return recs, labels

def parse_gtfile_rec(gt_ls, thred=0.01): #all objs in a img
    if gt_ls == []: return None
    recs = {}
    labels = {}
    for idx, sample in enumerate(gt_ls):
        img_key = sample['image']
        objects = []
        for result in sample['results']:
            obj_struct = {}
            obj_struct['name'] = result["tagnameid"]
            labels[result["tagnameid"]] = 1
            obj_struct['difficult'] = result["difficult"]  # in coco.py, all difficult=0
            obj = result["data"]
            obj_struct['bbox'] = [int(obj[0]), int(obj[1]),
                                  int(obj[2]), int(obj[3])]
            objects.append(obj_struct)
        recs[img_key] = objects
    return recs, labels

def parse_predfile_rec(pred_file, thred=0.01): #all objs in a img
    if pred_file is None: return None
    recs = {}
    with open(pred_file) as mfile:
        lines = mfile.readlines()
        for idx, line in enumerate(lines):
            sample = json.loads(line.strip())
            img_key = sample['image']
            objects = []
            for result in sample['results']:
                obj_struct = {}
                if result["confidence"][0] < thred: continue
                obj_struct["confidence"] = result["confidence"][0]
                obj_struct['name'] = result["tagnameid"]
                obj = result["data"]
                obj_struct['bbox'] = [int(obj[0]), int(obj[1]),
                                    int(obj[2]), int(obj[3])]
                objects.append(obj_struct)
            recs[img_key] = objects
    return recs


def precision_recal(pred_origin, gt_origin, iou_thres=0.5):
    tp, fp = 0, 0
    p_num = pred_origin.shape[0] if pred_origin.size > 0 else 0
    r_num = gt_origin.shape[0] if gt_origin.size > 0 else 0
    #pred is 0
    if p_num == 0: return tp, fp, p_num, r_num
    #gt is 0
    if r_num == 0: return tp, p_num, p_num, r_num

    BBGT = gt_origin.copy()
    # print(pred_origin, gt_origin)
    for bb in pred_origin:
        if BBGT.size == 0:
            fp += 1
            continue
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                (BBGT[:, 2] - BBGT[:, 0]) *
                (BBGT[:, 3] - BBGT[:, 1]) - inters)
        overlaps = inters / uni
        # print(overlaps,  'debug')
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > iou_thres:
            BBGT = np.delete(BBGT, [jmax], axis=0)
            tp += 1
        else:
            fp += 1
    return tp, fp, p_num, r_num

prob_thres = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.9, 0.922, 0.94, 0.96, 0.980, 0.985, 0.990, 0.995, 0.999]
def voc_eval(imagenames, gt_recs, pred_recs, classname, ovthresh=0.5, use_07_metric=True):
    # extract gt objects for this class
    class_gt_recs = {}
    npos = 0
    for imagename in imagenames:
        if imagename not in gt_recs.keys(): assert 1==0, "error, gts has't imagename: " + imagename
        R = [obj for obj in gt_recs[imagename] if obj['name'] == classname[1]]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        # print(imagename, bbox, sum(~difficult), len(imagenames))
        npos = npos + sum(~difficult)
        class_gt_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # extract pred objects for this class
    class_pred_recs = {}
    npred = 0
    for imagename in imagenames:
        if imagename not in pred_recs.keys(): #has't preds
            class_pred_recs[imagename] = {'bbox': np.array([]), 'confs': np.array([])}
            continue
        R = [obj for obj in pred_recs[imagename] if obj['name'] == classname[1]]
        bbox = np.array([x['bbox'] for x in R])
        confs = np.array([x['confidence'] for x in R])
        npred = npred + len(bbox)
        class_pred_recs[imagename] = {'bbox': bbox,
                                      'confs': confs}
    # print(classname[1], class_pred_recs)
    
    pred_nums = np.zeros((len(prob_thres), ), dtype=np.int32)
    gt_nums = np.zeros((len(prob_thres), ), dtype=np.int32)
    tp_nums = np.zeros((len(prob_thres), ), dtype=np.int32)
    fp_nums = np.zeros((len(prob_thres), ), dtype=np.int32)
    
    for idx, imagename in enumerate(imagenames):
        mask = class_gt_recs[imagename]['difficult'] == 0
        if not mask.any(): gt = np.array([])
        else: gt = class_gt_recs[imagename]['bbox'][mask, :]

        pred_box, confs = class_pred_recs[imagename]['bbox'], class_pred_recs[imagename]['confs']
        #print(pred_box)
        for k in range(len(prob_thres)):
            # print k
            if len(pred_box) > 0:
                I = confs > prob_thres[k]
                pred_k = pred_box[I, :]
            else:
                pred_k = np.array([]) #pred_ori, gt_ori, gt_side_ori, thres
            
            tp, fp, pn, rn = precision_recal(pred_k, gt, iou_thres=ovthresh)
            # if prob_thres[k] == 0.4: print(pred_k, gt, pred_k.size, tp, fp, pn, rn)

            tp_nums[k] += tp
            fp_nums[k] += fp
            pred_nums[k] += pn
            gt_nums[k] += rn
        
        if idx % 100 == 0:
            print("processed {:d}/{:d}".format(idx, len(imagenames)))
    
    result = []
    for k in range(len(prob_thres)):
        if pred_nums[k] == 0:  #pred_num = 0
            Recall = 0.0
            Precision = 1.0
        elif gt_nums[k] == 0: #gt_num = 0
            Recall = 1.0
            Precision = 0.0
        else:
            Precision = tp_nums[k] * 1.0 / pred_nums[k]
            Recall = tp_nums[k] * 1.0 / gt_nums[k]

        brs = {'threshold': prob_thres[k], 'TP': tp_nums[k], 'FP': fp_nums[k], 'TN': -1, 'FN': gt_nums[k]-tp_nums[k], 'TPR': -1, 'TNR': -1, 'FPR': -1,
               'Recall': Recall, 'Precision': Precision, 'Accuracy': -1}
        brs = {'threshold': prob_thres[k], 'Recall': Recall, 'Precision': Precision}
        result.append(brs)

    return result


#use_07 is unuse
def do_python_eval(nargs, use_07=False):
    timeArray = time.localtime(int(time.time()))
    otherStyleTime = time.strftime("%Y%m%d%H%M%S", timeArray)

    # gt_set = nargs.gt_files[0]
    # pred_set = nargs.score_files[0]
    # rt= args.rt_files[0]

    # gt_set = nargs.data_config.gt_js
    tagnames = nargs.data_config.tagnames
    test_root_dir = nargs.data_config.root_dir
    testset = nargs.data_config.test_list
    for test_data in testset:
        gt_set = Xml2js_gt(test_root_dir, test_data, tagnames)   #    generate gt_ls
        prex = test_data.split('/')[-1].split('.')[0]  # 'test1'   'test2'
        pred_set = nargs.data_config.js_dir + '/' + 'result_' + prex + '.json'
        rt = nargs.data_config.js_dir + '/' + 'eval_' + prex + '.json'
        PR_curve_path = nargs.data_config.js_dir

        aps = []
        res = []
        # read_json_files
        imagenames = []
        res_array = []
        gt_recs, labelmap = parse_gtfile_rec(gt_set)
        for i, item in enumerate(gt_recs.keys()):
            imagenames.append(item)
        pred_recs = parse_predfile_rec(pred_set, thred=0.01)

        # linename = os.path.splitext(os.path.basename(gt_set))[0]
        linename = prex
        for i, cls in enumerate(labelmap.keys()):
            linename += '_' + str(cls)
            result = voc_eval(imagenames, gt_recs, pred_recs, (i+1, cls), ovthresh=0.5, use_07_metric=use_07)
            item = {"linename": linename, "result":result, "eval_type":"map"}
            res_array.append(item)
            save_plot = True
            # if nargs.save_plot:
            if save_plot:
                plt.figure(figsize=(8, 8))
                plt.title(linename)
                plt.grid(True)
                ucid = linename
                PR_curve(plt, result, labelname=ucid)
                nsave_figname = linename + '.png'
                plt.savefig(PR_curve_path + '/'+nsave_figname)
                plt.close()
            linename = prex

        with open(rt, "w") as dump_fs:
            # json.dump(item, dump_fs, cls=NumpyEncoder, ensure_ascii=False,indent=1)
            json.dump(res_array, dump_fs, cls=NumpyEncoder, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    sovler_path = sys.argv[1]  # cfgs/headshoulder_solver.py
    solver = sovler_path.split('/')
    solver = '.'.join(solver)[:-3]
    solver = import_module(solver)  # import cfg.headshoulder_solver as cfg
    do_python_eval(solver)

