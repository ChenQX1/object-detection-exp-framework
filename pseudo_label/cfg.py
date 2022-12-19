import os
from dgdet.detector.resnet import RetinaNet

'''
cv2: BGR
mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
'''
Net = RetinaNet

weight_path = 'weight/resnet50_dcn.pth'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

thresh = 0.7

infer_shape = [800,800]
step = 400
#scale = [0.4,0.6,0.8,1]
scale = [0.6,0.8,1,1.4]

gpus = [1,2,3,4]

annot_save_dir = '$HOME/face_det/pseudo_label'