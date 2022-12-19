import os
import xml
import xml.dom.minidom
import xml.etree.ElementTree as ET
import os

def open_list(root_dir=None,train_list=None):
    result = []
    for item in train_list:
        # open each list
        fr = open('{}/{}'.format(root_dir,item)).readlines()
        # encode json
        for count,line in enumerate(fr):
            img_path,ann_path = line.strip().split(' ')
            if ann_path=='-1':
                continue
            img_path = '{}/{}'.format(root_dir,img_path)
            ann_path = '{}/{}'.format(root_dir,ann_path)
            #if os.path.exists(img_path) and os.path.exists(ann_path):
            info = {}
            info['img_path'] = img_path
            info['ann_path'] = ann_path
            info['org_list'] = item
            result.append(info)
            # if count==100:
            #     break
    print('total trian data : {}'.format(len(result)))
    return result

def read_xml(xml_file,min_box=5):
    tree = ET.parse(xml_file)
    target = tree.getroot()
    boxs = []
    for obj in target.iter('object'):
        name = obj.find('name').text
        if 'face' not in name:
            continue
        #print(name)
        bbox = obj.find('bndbox')
        x0 = float(bbox.find('xmin').text)
        y0 = float(bbox.find('ymin').text)
        x1 = float(bbox.find('xmax').text)
        y1 = float(bbox.find('ymax').text)
        if (x1-x0)<min_box or (y1-y0)<min_box:
            continue
        boxs.append([x0,y0,x1,y1])
    return boxs

def read_testset(root,f):
    result = []
    if root is None:
        root = '$HOME/face_det'
    fr = open('{}/{}'.format(root,f)).readlines()
    for line in fr:
        img_path,ann_path = line.strip().split(' ')
        if ann_path=='-1':
            continue
        img_path = '{}/{}'.format(root,img_path)
        ann_path = '{}/{}'.format(root,ann_path)
        if os.path.exists(ann_path):
            info = {
                'img_path':img_path,
                'ann_path':ann_path
            }
            result.append(info)
    return result







        
        
