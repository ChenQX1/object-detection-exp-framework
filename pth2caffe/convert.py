import onnx
import numpy as np
import caffe
from caffe.proto import caffe_pb2 as pb2
import pth2caffe.simplifier as sim
from pth2caffe.operators import DataLayer
from pth2caffe.register import convert_layer
from pth2caffe.weight import convert_weight
from pth2caffe.utils import infer_shape
from pth2caffe.utils import proto2np,save_net,get_input_shape
import google.protobuf.text_format
import os


def simplifier(onnx_model):
    onnx_model = infer_shape(onnx_model)
    model = sim.optimize_graph(onnx_model)
    nodes = list(model.graph.node)
    initializer = model.graph.initializer
    initializer = proto2np(initializer)
    #nodes = sim.optimize_resize(nodes,initializer)
    nodes = sim.optimize_shuffle(nodes,initializer)
    nodes = sim.optimize_softmax(nodes,initializer)
    nodes = sim.optimize_view(nodes,initializer)
    #onnx_model.graph.node.clear()
    return nodes,initializer

def convert_NetParameter(onnx_model):
    nodes,initializers = simplifier(onnx_model)
    #print(nodes)
    #nodes = rename_node2(nodes)
    shape = get_input_shape(onnx_model)
    #---------init---------
    net_param = pb2.NetParameter()
    #------convert op----------
    layers = []
    data_layer = DataLayer(shape)
    layers.append(data_layer)
    for node in nodes:
        layer = convert_layer(node,initializers)
        if layer is None:
            continue
        if isinstance(layer,list):
            layers.extend(layer)
            continue
        layers.append(layer)
    net_param.layer.extend(layers)
    return net_param

def cp_weight(caffe_net,onnx_model):
    nodes,initializers = simplifier(onnx_model)
    #nodes = rename_node2(nodes)
    #------copy weights------
    for node in nodes:
        params = convert_weight(node,initializers)
        if params is None:
            continue
        #node to many layers
        if isinstance(params,list):
            for item in params:
                layer = item['name']
                data = item['data']
                for idx,w in enumerate(data):
                    np.copyto(caffe_net.params[layer][idx].data,w)
            continue
        #node to 1 layer
        layer = params['name']
        data = params['data']
        for idx,w in enumerate(data):
            np.copyto(caffe_net.params[layer][idx].data,w)
    return caffe_net

def onnx2caffe(onnx_model,save_dir='tmp'):
    '''
    save prototxt and caffemodel
    '''
    print('gen prototxt...')
    net_param = convert_NetParameter(onnx_model)
    with open('{}/net.prototxt'.format(save_dir), 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(net_param))
    #return None
    print('copy weight...')
    caffe_net = caffe.Net('{}/net.prototxt'.format(save_dir),caffe.TEST)
    caffe_net = cp_weight(caffe_net,onnx_model)
    caffe_net.save('{}/net.caffemodel'.format(save_dir))
    return caffe_net

def export_caffe(net,shape=(3,224,224),save_dir='tmp'):
    onnx_path = '{}/net.onnx'.format(save_dir)
    save_net(net,shape,onnx_path)
    onnx_model = onnx.load(onnx_path)
    if os.path.exists(save_dir) is False:
        os.system('mkdir {}'.format(save_dir))
    onnx2caffe(onnx_model,save_dir)

def main():
    onnx_model = onnx.load('hrnet.onnx')
    #onnx_model = onnx.load('onnx_model/resize.onnx')
    #node = onnx_model.graph.node 
    #print(node)
    #net_parm = convert_NetParameter(onnx_model)
    #print(net_parm)
    #nodes = simplifier_node(onnx_model)
    #print(nodes)
    onnx2caffe(onnx_model,save_dir='xx')


if __name__ == "__main__":
    main()




