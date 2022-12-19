import xml.etree.ElementTree as ET
import numpy as np

class Filter(object):
    def __init__(self,maps,min_box):
        self.maps = maps
        self.min_box = min_box

    def read_xml(self,xml_path):
        tree = ET.parse(xml_path)
        target = tree.getroot()
        annots = []
        for obj in target.iter('object'):
            name = obj.find('name').text
            id = None
            for item in self.maps.keys():
                if item in name:
                    id = self.maps[item]+1
            if id is None:
                #raise ValueError("ground truth not in filter maps!")
                continue
            #print(name)
            bbox = obj.find('bndbox')
            x0 = float(bbox.find('xmin').text)
            y0 = float(bbox.find('ymin').text)
            x1 = float(bbox.find('xmax').text)
            y1 = float(bbox.find('ymax').text)
            if (x1-x0)<self.min_box or (y1-y0)<self.min_box:
                continue
            #box = [x0,y0,x1,y1]
            annot = [x0,y0,x1,y1,id]
            annots.append(annot)
        if len(annots)==0:
            annots = np.zeros(shape=(0,5))
        else:
            annots = np.array(annots)
        return annots