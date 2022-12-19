import json,  os, time
import numpy as np
import sys
import argparse
import xml.etree.ElementTree as ET
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(cur_dir))

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

def parsing_xmlfile(xml_path, tag_id):
    ls = []
    xml_file = ET.parse(xml_path).getroot()
    for obj in xml_file.iter('object'):
        cls_name = obj.find('name').text.strip().lower()
        class_id = None
        #id = None
        for item in tag_id.keys():
            if item in cls_name:
                class_id = tag_id[item]
        if class_id == None:
            continue
        '''
        for idx, key in enumerate(tag_id.keys()):
            if key in cls_name:
                class_id = idx
                break
        

        
        if class_id is None:
            #print(key,cls_name)
            #print(xml_path)
            print('-----class_name parsing .xml file is not found in tagnames, '
                    'please check again!')
            exit(0)
        '''
        xml_box = obj.find('bndbox')
        xmin = float(xml_box.find('xmin').text)
        ymin = float(xml_box.find('ymin').text)
        xmax = float(xml_box.find('xmax').text)
        ymax = float(xml_box.find('ymax').text)
        # tmp = [int(xmin), int(ymin), int(xmax), int(ymax), tag_id[cls_name]]
        tmp = [int(xmin), int(ymin), int(xmax), int(ymax), class_id]
        ls.append(tmp)
    return ls


def Xml2js_gt(test_root_dir, test_list, tagnames):
    test_ls = test_root_dir+'/'+test_list
    img_xml_path = open(test_ls, 'r').readlines()
    gt_ls = []
    for line in img_xml_path:
        line_info = line.strip('\n').split(' ')
        print(line_info)
        img_path = test_root_dir+'/'+line_info[0]
        xml_path = test_root_dir+'/'+line_info[1]
        if os.path.exists(xml_path) is False:
            continue
        obj_box = parsing_xmlfile(xml_path, tagnames)
        img_val = []
        for info in obj_box:
            item = {'evaltype': 'map', 'data':info[:4], 'tagnameid': str(info[4]), 'difficult': '0'}
            img_val.append(item)
        dt = {'image': img_path, 'results': img_val}
        gt_ls.append(dt)
    return gt_ls


# if __name__ == "__main__":
#     sovler_path = sys.argv[1]  # cfgs/dangercar_solver.py
#     solver = sovler_path.split('/')
#     solver = '.'.join(solver)[:-3]
#     solver = import_module(solver)  # import cfg.dangercar_solver as cfg
#     Xml2js_gt(solver)







