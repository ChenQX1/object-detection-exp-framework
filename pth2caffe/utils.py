import torch
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import shape_inference
from caffe.proto import caffe_pb2 as pb2
from pth2caffe.rename import rename

def save_net(net,shape,onnx_path):
    print('gen onnx model...')
    c,h,w = shape
    dump = torch.randn(size=(1,c,h,w))
    torch.onnx.export(net,dump,onnx_path,opset_version=11,input_names=['data'],keep_initializers_as_inputs=True)
    # rename onnx model
    onnx_model = onnx.load(onnx_path)
    onnx_model = rename(onnx_model)
    # save onnx model
    onnx.save(onnx_model,onnx_path)

def proto2np(initializer):
    result = {}
    for item in initializer:
        name = item.name
        data = numpy_helper.to_array(item)
        #print(name)
        #print(data.shape)
        result[name] = data
    return result

def proto2dict(proto):
    result = {}
    for item in proto:
        name = item.name
        result[name] = item
    return result

def get_input_shape(onnx_model):
    inputs = onnx_model.graph.input
    shape = inputs[0].type.tensor_type.shape
    result = []
    for item in shape.dim:
        result.append(item.dim_value)
    return result


'''
add shape to attribute
'''
require_shapes = ['Resize','Reshape','ReduceMean']
def infer_shape(onnx_model):
    onnx_model = shape_inference.infer_shapes(onnx_model,check_type=True)
    blob_shapes = proto2dict(onnx_model.graph.value_info)
    output_shapes = proto2dict(onnx_model.graph.output)
    blob_shapes.update(output_shapes)
    #print(blob_shapes.keys())
    for item in onnx_model.graph.node:
        if item.op_type in require_shapes:
            #------get input shape------------
            input = item.input[0]
            shape = blob_shapes[input]
            #print(shape)
            proto = shape.type.tensor_type.shape.dim
            value = []
            for dim in proto:
                value.append(dim.dim_value)
            #print(value)
            shape_in = helper.make_attribute(key='shape_in',value=value)
            item.attribute.extend([shape_in]) #add shape to attrs

            #-------get output shape----------
            output = item.output[0]
            #print(output)
            shape = blob_shapes[output]
            proto = shape.type.tensor_type.shape.dim
            value = []
            for dim in proto:
                value.append(dim.dim_value)
            shape_out = helper.make_attribute(key='shape_out',value=value)
            item.attribute.extend([shape_out]) #add shape to attrs
            #print(item)
    return onnx_model

'''
only support Slice on axes=1
'''
def delete_slice(onnx_nodes):
    for node in onnx_nodes[:]:
        if node.op_type == 'Slice':
            attr = proto2dict(node.attribute)
            if attr['axes'].ints[0]!=1:
                onnx_nodes.remove(node)
            elif attr['axes'].ints[0]==1:
                input = int(node.input[0])
                #print(input)
                node.input[0] = str(input-1)
    return onnx_nodes


def main():
    onnx_model = onnx.load('onnx_model/GasStationStaff_nv1.onnx')
    onnx_model = infer_shape(onnx_model)
    onnx.save(onnx_model,'onnx_model/xxx_s.onnx')
    # node = onnx_model.graph.node
    # for item in node:
    #     if item.op_type=='Resize':
    #         #print(item)
    #         pass

if __name__ == "__main__":
    main()
