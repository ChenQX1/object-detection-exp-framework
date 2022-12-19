import os, sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path

root = '$HOME/face_det'
DATA_LIST_PATH = "$HOME/face_det/testset0915/pair.lst"

DET_FILE = "result/result_tiny.txt"
#DET_FILE = "result/result1305_192x108.txt"
ROOT_PATH = str(Path("./").absolute())


def get_gt_file(pred_file, thred=0.3):
    if pred_file is None: return None
    pred_dict = {}
    lines = open(pred_file, 'r').readlines()
    for line in lines:
        split_item = line.strip().split()
        image_path = split_item[0]  # os.path.basename()
        # print(image_path, split_item)
        if len(split_item) < 3: continue
        boxes = np.array(split_item[1:]).reshape((-1, 5))
        for _item in boxes:
            score = float(_item[0])
            if score < thred: continue
            # image key first occur
            if not image_path in pred_dict.keys():
                pred_dict[image_path] = list()
            box = list(map(float, _item[1:]))
            box.append(score)
            pred_dict[image_path].append(box)  # list(map(int, box))
    # print(pred_dict)
    return pred_dict


def parse_file_rec(pred_file, thred=0.5):  # all objs in a img
    pred_dict = get_gt_file(pred_file, thred=thred)
    recs = {}
    for image_path in pred_dict.keys():
        targets = pred_dict[image_path]
        objects = []
        for obj in targets:
            if int(obj[-1]) == -1:  # no target form [-1]*6
                # print('Warning: no gts')
                break
            obj_struct = {}
            obj_struct['name'] = 'face'
            obj_struct['difficult'] = 0  # in coco.py, all difficult=0
            obj_struct['cls_id'] = int(obj[-1])  # start from 0
            obj_struct['bbox'] = [int(obj[0]), int(obj[1]),
                                  int(obj[2]), int(obj[3])]
            objects.append(obj_struct)
        recs[image_path] = objects

    return recs


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    #print(filename)
    # filename = Path(filename).name
    # filename = str(Path("./Annotations")/filename)
    filename = '{}/{}'.format(root,filename)
    tree = ET.parse(filename)
    objects = []

    sz = tree.find('size')
    width = float(sz.find('width').text)
    height = float(sz.find('height').text)
    ratio = 1.
    # if height > 360 or width > 640:
    #     ratio = np.min(np.array([360, 640]).astype(np.float) / np.array([height, width]))

    for obj in tree.findall('object'):
        obj_struct = {}
        name = obj.find('name').text.lower().strip()
        if 'palm' not in name:
            name = name.replace('fjs_', '')
        if 'face' != name:
            print(name, 'not exist...')
            continue
        # else: name = name.replace('poi_', '')

        obj_struct['name'] = name
        obj_struct['pose'] = obj.find('pose').text if (obj.find('pose')) else -1
        obj_struct['truncated'] = int(obj.find('truncated').text) if (obj.find('truncated')) else -1
        obj_struct['difficult'] = int(obj.find('difficult').text) if (obj.find('difficult')) else 0
        bbox = obj.find('bndbox')

        obj_struct['bbox'] = [int(float(bbox.find('xmin').text) * ratio),
                              int(float(bbox.find('ymin').text) * ratio),
                              int(float(bbox.find('xmax').text) * ratio),
                              int(float(bbox.find('ymax').text) * ratio)]
        h = int(float(bbox.find('ymax').text) * ratio) - int(float(bbox.find('ymin').text) * ratio)
        w = int(float(bbox.find('xmax').text) * ratio) - int(float(bbox.find('xmin').text) * ratio)
        # if max(h, w) < 27:
        #     # print('filter small box:', max(h, w))
        #     continue
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:  #############invalid value encountered in greater_equal
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(classname, ovthresh=0.5, use_07_metric=True):
    # read list of images
    with open(DATA_LIST_PATH, 'r') as f:
        lines = f.readlines()
    img_xml = [x.strip().split() for x in lines]
    recs = {}
    imagenames = []

    for i, item in enumerate(img_xml):
        imagename, xml_path = item

        imagenames.append(imagename)
        recs[imagename] = parse_rec(xml_path)
        if i % 100 == 0:
            print('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(img_xml)))
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        if imagename not in recs.keys(): continue
        R = [obj for obj in recs[imagename] if obj['name'] == classname[1]]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    with open(DET_FILE, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        # TODO
        image_ids, confidence, BB = [], [], []
        splitlines = [x.strip().split(' ') for x in lines]
        for x in splitlines:
            image_ids.append(x[0])
            confidence.append(float(x[1]))  # 2
            BB.append([float(z) for z in x[2:]])  # 3
        confidence = np.array(confidence)
        BB = np.array(BB)
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            if image_ids[d] not in class_recs.keys():  # hasn't gts
                # print(image_ids[d], 'has not gts')
                fp[d] = 1.
                continue
            R = class_recs[image_ids[d]]  # gt
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
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
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

            # print('ssss', sorted_scores[d])
        saved_list = []
        error_cnt = [0, 0]  # miss or error det
        for d in range(nd):  # have processed
            if image_ids[d] in saved_list: continue
            error = False
            if image_ids[d] not in class_recs.keys():  # hasn't gts
                error = True
            if not error:
                R = class_recs[image_ids[d]]  # gt
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:  # no det
                    for m in R['det']:
                        if not m:  # error det box
                            error = True
                            error_cnt[0] += 1
                            # break
            if fp[d] == 1: error_cnt[1] += 1

        print("miss or error det", error_cnt)
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap, -sorted_scores


def do_python_eval(use_07=True):
    aps = []
    labelmap = ['face']
    for i, cls in enumerate(labelmap):
        rec, prec, ap, soted_scores = voc_eval((i + 1, cls), ovthresh=0.5, use_07_metric=use_07)
        for i,r in enumerate(rec):
            if prec[i]<=0.99:
                print("pre {:.4f} recall:{:.4f}, thresh:{:.4f}".format(prec[i], r, soted_scores[i]))
                break
        # for i,r in enumerate(rec):
        #     if soted_scores[i]<=0.7:
        #         print("pre {:.4f} recall:{:.4f}, thresh:{:.4f}".format(prec[i], r, soted_scores[i]))
        #         break
        aps += [ap]

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    return rec, prec, ap, soted_scores,aps

def plot_prcurve(all_res):
    for res in all_res:
        rec, prec, ap = res
        plt.plot(rec, prec, lw=2, label='PR of {} (area = {:.4f})'.format("face", ap[0]))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.legend()
    plt.show()

def one_exp():
    all_res = []
    rec, prec, ap, soted_scores,aps = do_python_eval()
    all_res.append([rec, prec,aps])
    #plot_prcurve(all_res)

def multi_exp():
    if len(sys.argv)>1:
        ROOT_PATH = sys.argv[1]
        DATA_LIST_PATH = sys.argv[2]
        DET_FILE = sys.argv[3]
    exp_list = ["./result1305_1.txt"]
    all_res = []
    for exp in exp_list:
        DET_FILE = exp
        rec, prec, ap, soted_scores,aps = do_python_eval()
        all_res.append([rec, prec,aps])
    # plot_prcurve(all_res)

if __name__ == "__main__":
    one_exp()
