import os, sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def convert_temp(origin_frame):
    return  origin_frame.astype(np.float32) / 10.0 - 273.2

def calc_minmax(origin_frame):
    h, w = origin_frame.shape
    frame4hist = origin_frame.astype(np.int32) / 10
    hist, bin_edges = np.histogram(frame4hist, bins=10000, range=(0,10000))
    thdratio = 0.02
    minv, maxv = -1, -1
    for i, hist_num in enumerate(hist):
        ratio = hist_num * 100.0 / (h*w)
        if minv < 0 and ratio >= thdratio:
            minv = i * 10
        if ratio >= thdratio:
            maxv = i * 10
    return minv, maxv

def gen_gray_map(origin_frame):
    h, w = origin_frame.shape
    minv, maxv = calc_minmax(origin_frame)
    v = (origin_frame.astype(np.float32) - minv) / float(maxv - minv + 1e-8)
    v = np.clip(v, 0, 1) * 255
    char_v = v.astype(np.uint8)
    char_v = np.zeros((h, w, 3), dtype=np.uint8)
    char_v[:,:,0] = v.astype(np.uint8).copy()
    char_v[:,:,1] = v.astype(np.uint8).copy()
    char_v[:,:,2] = v.astype(np.uint8).copy()
    return char_v

def gen_color_map(origin_frame):
    fminv, fmaxv = 0.0, 40.0
    temp_map = convert_temp(origin_frame)
    v = (temp_map - fminv) / (fmaxv - fminv)
    v = (np.clip(v, 0, 1) ** 1.8) * 255
    v_char = v.astype(np.uint8)
    
    v_smooth = cv2.bilateralFilter(v_char, 10, 20, 5)
    v_hot = cv2.applyColorMap(v_smooth, cv2.COLORMAP_JET)
    return v_hot

def visual_temp_map(origin_frame): #param: gray img
    gray_map = gen_gray_map(origin_frame)
    color_map = gen_color_map(origin_frame)
    
    weight = 0.25
    return gray_map #* weight + color_map * (1 - weight)

#process xml gt file and pred file
def get_gt_file(pred_file, thred=0.3):
    if pred_file is None: return None
    pred_dict = {}
    lines = open(pred_file, 'r').readlines()
    for line in lines:
        split_item = line.strip().split()
        image_path = split_item[0] #os.path.basename()
        # print(image_path, split_item)
        if len(split_item) < 3: continue
        boxes = np.array(split_item[1:]).reshape((-1,5))
        for _item in boxes:
            score = float(_item[0])
            if score < thred: continue
            #image key first occur
            if not image_path in pred_dict.keys():
                pred_dict[image_path] = list()
            box = list(map(float, _item[1:]))
            # box[2] += box[0]
            # box[3] += box[1]
            # print(box)
            box.append(score)
            pred_dict[image_path].append(box) #list(map(int, box))
    # print(pred_dict)
    return pred_dict

def parse_file_rec(pred_file, thred=0.5): #all objs in a img
    pred_dict = get_gt_file(pred_file, thred=thred)
    recs = {}
    for image_path in pred_dict.keys():
        targets = pred_dict[image_path]
        objects = []
        for obj in targets:
            if int(obj[-1]) == -1:  #no target form [-1]*6
                #print('Warning: no gts')
                break
            obj_struct = {}
            obj_struct['name'] = 'face'
            obj_struct['difficult'] = 0 #in coco.py, all difficult=0
            obj_struct['cls_id'] = int(obj[-1])  #start from 0
            obj_struct['bbox'] = [int(obj[0]), int(obj[1]),
                                int(obj[2]), int(obj[3])]
            objects.append(obj_struct)
        recs[image_path] = objects

    return recs

def show_results(image, targets=None, preds=None, resize_hw=None, img_name=None):
    im_copy = image.copy()
    h, w, c = image.shape
    if resize_hw is None: resize_hw = (h, w)
    image = cv2.resize(image, (resize_hw[1], resize_hw[0]))

    if not (targets is None or targets.shape[0] == 0):
        gt = targets[:, :4]
        label = 0 #targets[:, 4]
        # score = targets[:, 4]
        ###Plot the boxes
        for i in range(len(gt)):
            xmin = int(round(gt[i][0]) / w * resize_hw[1])
            ymin = int(round(gt[i][1]) / h * resize_hw[0])
            xmax = int(round(gt[i][2]) / w * resize_hw[1])
            ymax = int(round(gt[i][3]) / h * resize_hw[0])
            
            coords = (xmin, ymin), xmax-xmin, ymax-ymin
            color = (0,255,0) #colors[label[i]]
            # print(label[i], 'label', class_dict)
            display_txt = 'gt_{}'.format(int(label))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=2)

            tx, ty = xmin, ymin
            if ymin < 15: ty = ymax + 8
            cv2.putText(image,'{}'.format(display_txt), (tx-10, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

    # h, w, c = 360, 640, 3 #image.shape
    if not (preds is None or preds.shape[0] == 0):
        # print(preds.shape, 'ssssss')
        gt = preds[:, :4]
        label = 1 #preds[:, 4]
        score = preds[:, 4]
        
        ###Plot the boxes
        colors = [(0,0,255),(255,0,0),(0,255,255),(128,0,128),(128,128,0),(255,165,0),(192,14,235),]

        for i in range(len(gt)):
            xmin = int(round(gt[i][0]) / w * resize_hw[1])
            ymin = int(round(gt[i][1]) / h * resize_hw[0])
            xmax = int(round(gt[i][2]) / w * resize_hw[1])
            ymax = int(round(gt[i][3]) / h * resize_hw[0])
            
            coords = (xmin, ymin), xmax-xmin, ymax-ymin
            color = colors[1]
            # print(label[i], 'label', class_dict)
            # display_txt = [key for (key, value) in class_dict.items() if value == label[i]][0].replace('_', '')+'_{:.3f}'.format(score[i])
            display_txt = 'f_{:.3f}'.format(score[i]*-1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=2)

            tx, ty = xmin, ymin
            if ymin < 15: ty = ymax + 8
            cv2.putText(image,'{}'.format(display_txt), (tx-10, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

            # expand_ratio = 0.125
            # roih, roiw = ymax - ymin, xmax - xmin
            # if min(roih, roiw) <= 0: continue
            # xmin = max(0, xmin - roiw*expand_ratio)
            # ymin = max(0, ymin - roih*expand_ratio)
            # xmax = min(w, xmax + roiw*expand_ratio)
            # ymax = min(h, ymax + roih*expand_ratio)
            # roi = im_copy[int(ymin):int(ymax), int(xmin):int(xmax)]
            # if min(roi.shape) == 0: continue
            # cv2.imwrite('tmp/{}_{}'.format(i, img_name), roi)
    
    return image
#####################


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
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
            if np.sum(rec >= t) == 0:   #############invalid value encountered in greater_equal
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

detfile = None
pred_file = None
def voc_eval(data_root, test_set, results_set, classname, ovthresh=0.5, use_07_metric=True):
    # read list of images
    with open(os.path.join(data_root, test_set[0], test_set[1]), 'r') as f:
        lines = f.readlines()
    img_xml = [x.strip().split() for x in lines]

    recs = {}
    imagenames = []
    if len(img_xml[0]) == 2 and img_xml[0][1][-4:] == '.xml':
        for i, item in enumerate(img_xml):
            imagename, xml_path = item

            imagenames.append(imagename)
            recs[imagename] = parse_rec(os.path.join(data_root, xml_path))
            if i % 500 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(img_xml))) 
    else:
        #'id_cropimgs_results_pf.lst' #'det_test_face_1301.txt' #'no_face_results_pf.lst' #'image_lyy_results_pf.lst' #'car_results_pf.lst' #
        # pred_file = 'big_face_results1303.txt' #'mask1_results/det_results_1301.txt'
        recs = parse_file_rec(pred_file, thred=0.4)
        # imagenames = recs.keys()
        for i, item in enumerate(img_xml):
            imagenames.append(item[0])

    # extract gt objects for this class
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
    # print(class_recs)
    # NOTE read dets
    # detfile = 'results_hi35/det_test_{}.txt'.format(classname[1])
    #'face_det_mask200128_test_results/det_results_int8.txt' #'face_det_mask200128_5k_1303.txt' #'mask1_results/det_results.txt' #'big_face_results/det_results.txt' #'big_face_results/det_results.txt' #
    # 'face_det_mask200128_test_results/det_results_int8v2.txt' #'face_det_mask200131_allcover_test_results/det_results.txt'
    # detfile = 'results/det_test_face_1305_int8_2.txt' #'big_face_results/det_results.txt' #
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        #TODO
        image_ids, confidence, BB = [], [], []
        splitlines = [x.strip().split(' ') for x in lines]
        for x in splitlines:
            # if len(x) < 3: continue
            # if int(x[1]) != classname[0]: continue
            # if float(x[1]) < 0.15: continue
            # if float(x[2]) < 0.15: continue
            image_ids.append(x[0])
            confidence.append(float(x[1])) #2
            BB.append([float(z) for z in x[2:]]) #3
        confidence = np.array(confidence)
        BB = np.array(BB)
        # print(BB)
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            if image_ids[d] not in class_recs.keys():    #hasn't gts
                # print(image_ids[d], 'has not gts')
                fp[d] = 1.
                continue
            R = class_recs[image_ids[d]] #gt
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # print(BBGT, bb)
                # exit(0)
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
        save_diff = False
        saved_list = []
        error_cnt = [0, 0] #miss or error det
        for d in range(nd): #have processed
            if image_ids[d] in saved_list: continue
            error = False
            if image_ids[d] not in class_recs.keys():    #hasn't gts
                error = True
            if not error:
                R = class_recs[image_ids[d]] #gt
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:   #no det
                    for m in R['det']:
                        if not m:   #error det box
                            error = True
                            error_cnt[0] += 1
                            # break
            if fp[d] == 1: error_cnt[1] += 1

            if (fp[d] == 1 or error):
                saved_list.append(image_ids[d])
                if not save_diff: continue

                select_idx = []
                for ii, x in enumerate(image_ids):
                    if x == image_ids[d]: select_idx.append(ii)
                preds = np.column_stack((BB[select_idx, :], sorted_scores[select_idx]))
                targets = class_recs[image_ids[d]]['bbox'] if image_ids[d] in class_recs.keys() else None
                
                if image_ids[d][-3:] == 'npy':
                    img = np.load(os.path.join(data_root, image_ids[d]))
                    # image = np.repeat(img[..., np.newaxis], 3, 2)
                    image = visual_temp_map(img)
                else:
                    image = cv2.imread(os.path.join(data_root, image_ids[d]))
                    #print(data_root, image_ids[d])
                if image is None: print("Unable to read", os.path.join(data_root, image_ids[d]))

                img_path_split = image_ids[d].split('/')
                img_name = os.path.basename(image_ids[d])
                # print(image_ids[d])
                vis_img = show_results(image, targets, preds, resize_hw=None, img_name=img_name)
                # vis_img = cv2.resize(vis_img, (640, 360))
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30] #default 95
                result, encimg = cv2.imencode('.jpg', vis_img, encode_param)
                if result == False: print('could not encode image!')
                vis_img = cv2.imdecode(encimg, 1)
                cv2.imwrite(data_root+'/diff/'+img_path_split[-2]+'_'+img_name+'.jpg', vis_img)
                # return
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

    return rec, prec, ap


def do_python_eval(data_root, test_set=['', 'test.lst'], results_set='results', use_07=True):
    aps = []
    res = []
    labelmap = ['face',] #['fire',] #['part_cover','all_cover','lp', 'nolp', 'dirty_cover', 'other_cover', 'blur', 'light']
    # labelmap = ('poi_water','poi_phone','poi_palm','poi_face')

    for i, cls in enumerate(labelmap):
        rec, prec, ap = voc_eval(data_root, test_set, results_set, (i+1, cls), ovthresh=0.5, use_07_metric=use_07)
        aps += [ap]
        print("thres=0.4", rec[-4], prec[-4], rec[4], prec[4])
        # plt.plot(rec, prec, lw=2, label='PR of {} (area = {:.4f})'.format(cls, ap))
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.legend()
        # plt.show()
    print('Mean AP = {:.4f}'.format(np.mean(aps)))

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('Usage: ' + sys.argv[0] + ' <data_root> <test_list> <pred_file> <detfile>')
        data_root = './'
        test_list = 'sndg/sntest.txt' #'image_kunming_bank_imgs_test.lst' #'bank_badcase.lst' #'test_4k.lst' #'track_face_det_error_imgs.lst'
        pred_file = None #'test_4k_det_test_face.txt' 
        detfile = 'results/det_test_face_65000v0_87.8.txt' #'test_tmp.txt' #'test_4k_results/det_results.txt'
    else:
        data_root = sys.argv[1]
        test_list = sys.argv[2]
        pred_file = sys.argv[3]
        detfile = sys.argv[4]
        print('inited by sys.argv...')

    # data_root = './'    #('face_det_mask', 'tmp') #
    test_set = ('', test_list)
    results_set = 'results'

    # data_root = '/home/maolei/data/tmp/face_det_vedio/'
    # test_set = ('', 'DMS_0614_test.lst')

    # data_root = '/home/maolei/data/face_det'
    # test_set = ('sndg', 'sntest.txt')
    # results_set = 'results'

    do_python_eval(data_root, test_set, results_set)
