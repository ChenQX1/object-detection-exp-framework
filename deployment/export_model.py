import torch
import onnx
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.dirname(cur_dir))
from pth2caffe.convert import export_caffe
from pth2caffe.utils import save_net
from pth2caffe.convert import onnx2caffe
from deployment.ssd import get_priorbox,add_ssd_layers
import google.protobuf.text_format
from cfgs import base_solver
from importlib import import_module

def convert(cfg,save_dir='tmp'):
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    # load network
    if cfg is None:
        cfg = base_solver
    net = cfg.Net(cfg.num_class)
    net.load_state_dict(torch.load(cfg.weight_path))
    net.deploy()
    
    # convert caffe model in deploy mode
    h,w = cfg.infer_shape
    export_caffe(net,(3,h,w),save_dir)

    # gen post-process layer
    onnx_model = onnx.load('{}/net.onnx'.format(save_dir))
    output = [x.name for x in onnx_model.graph.output]
    print(output)
    nb_level = len(output)//2
    clf_blobs = output[0:nb_level]
    reg_blobs = output[nb_level:]
    priorboxs = get_priorbox(net.anchors)
    net_params = add_ssd_layers(clf_blobs,reg_blobs,priorboxs,cfg.num_class)
    
    # rewrite prototxt
    with open('{}/net.prototxt'.format(save_dir), 'a') as f:
        f.write(google.protobuf.text_format.MessageToString(net_params))


if __name__ == "__main__":
    cfg_path = sys.argv[-1]
    cfg = cfg_path.split('/')
    cfg = '.'.join(cfg)[:-3]
    cfg = import_module(cfg)
    convert(cfg)