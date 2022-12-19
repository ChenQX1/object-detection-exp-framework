from numpy.core.fromnumeric import shape
import onnx
from caffe.proto import caffe_pb2 as pb2
from pth2caffe.utils import proto2dict

'''
convert onnx node to caffe layer
'''

def DataLayer(shape):
    #-----init layer---------
    layer = pb2.LayerParameter()
    layer.type = 'Input'
    layer.top.extend(['data'])
    layer.name = 'data'
    input_shape = pb2.BlobShape()
    input_shape.dim.extend(shape)
    layer.input_param.shape.extend([input_shape])
    return layer

def _Conv(node,initializers):
    #-----init layer------
    layer = pb2.LayerParameter()
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])
    layer.name = node.name

    #-----get param-------
    w = initializers[node.input[1]]
    if node.op_type=='Conv':
        layer.type = 'Convolution'
        layer.convolution_param.num_output = w.shape[0]
    elif node.op_type=='ConvTranspose':
        layer.type = 'Deconvolution'
        layer.convolution_param.num_output = w.shape[1]
    if len(node.input)==2:
        layer.convolution_param.bias_term = False
    attrs = proto2dict(node.attribute)
    attr = attrs['kernel_shape']
    if attr.ints[0]==attr.ints[1]:
        layer.convolution_param.kernel_size.extend([attr.ints[0]])
    else:
        layer.convolution_param.kernel_h = attr.ints[0]
        layer.convolution_param.kernel_w = attr.ints[1]
    attr = attrs['strides']
    if attr.ints[0]==attr.ints[1]:
        layer.convolution_param.stride.extend([attr.ints[0]])
    else:    
        layer.convolution_param.stride_h = attr.ints[0]
        layer.convolution_param.stride_w = attr.ints[1]
    attr = attrs['pads']
    if attr.ints[0]==attr.ints[1]:
        layer.convolution_param.pad.extend([attr.ints[0]])
    else:
        layer.convolution_param.pad_h = attr.ints[0]
        layer.convolution_param.pad_w = attr.ints[1]
    attr = attrs['group']
    layer.convolution_param.group = attr.i
    attr = attrs['dilations']
    layer.convolution_param.dilation.extend([attr.ints[0]])
    return layer


def _BatchNormalization(node,initializers):
    #-----init layer norm------
    layer1 = pb2.LayerParameter()
    layer1.type = 'BatchNorm'
    layer1.name = "{}_norm".format(node.name)
    layer1.bottom.extend([node.input[0]])
    layer1.top.extend(node.output)
    #-----get param-------
    attrs = proto2dict(node.attribute)
    
    #-----init layer scale-------
    layer2 = pb2.LayerParameter()
    layer2.type = 'Scale'
    layer2.name = "{}_scale".format(node.name)
    layer2.bottom.extend(node.output)
    layer2.top.extend(node.output)
    layer2.scale_param.bias_term = True
    return [layer1,layer2]


def _Pool(node,initializers):
    #-----init layer------
    layer = pb2.LayerParameter()
    layer.type = 'Pooling'
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])
    layer.name = node.name
    
    #-----get param-------
    if node.op_type=='MaxPool':
        layer.pooling_param.pool = pb2.PoolingParameter.MAX
    elif node.op_type=='AveragePool':
        layer.pooling_param.pool = pb2.PoolingParameter.AVE
    for attr in node.attribute:
        if attr.name=='kernel_shape':
            layer.pooling_param.kernel_size = attr.ints[0]
        if attr.name == 'strides':
            layer.pooling_param.stride = attr.ints[0]
        if attr.name=='pads':
            layer.pooling_param.pad = attr.ints[0]
    return layer


def _Relu(node,initializers):
    #-----init layer------
    layer = pb2.LayerParameter()
    layer.type = 'ReLU'
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])
    layer.name = node.name
    return layer

def _Add(node,initializers):
    #-----init layer------
    layer = pb2.LayerParameter()
    layer.type = "Eltwise"
    layer.bottom.extend(node.input)
    layer.top.extend([node.output[0]])
    layer.name = node.name
    return layer

def _Flatten(node,initializers):
    #-----init layer------
    layer = pb2.LayerParameter()
    layer.type = 'Flatten'
    layer.flatten_param.axis = 1
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])
    layer.name = node.name
    return layer

# 没写完呢!!! -_-
def _Reshape(node,initializers):
    #-----init layer-----
    layer = pb2.LayerParameter()
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])
    layer.name = node.output[0]

    #----init params-----
    layer.type = 'Reshape'
    attrs = proto2dict(node.attribute)
    shape = attrs['shape']
    print(shape)
    layer.reshape_param.shape.dim.extend(shape)
    return layer

def _Sigmoid(node,initializers):
    #-----init layer------
    layer = pb2.LayerParameter()
    layer.type = 'Sigmoid'
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])
    return layer

def _Softmax(node,initializers):
    #-----init layer------
    layer = pb2.LayerParameter()
    layer.type = 'Softmax'
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])
    return layer

def _GlobalAveragePool(node,initializers):
    #-----init layer------
    layer = pb2.LayerParameter()
    layer.type = 'Pooling'
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])

    layer.pooling_param.pool = pb2.PoolingParameter.AVE
    layer.pooling_param.global_pooling = True

    return layer

def _Gemm(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.type = 'InnerProduct'
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])

    #-------init params------------
    if len(node.input)==2:
        layer.inner_product_param.bias_term = False
    w = node.input[1]
    w = initializers[w]
    output_num = w.shape[0]
    layer.inner_product_param.num_output = output_num
    return layer

def _Mul(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend(node.input)
    layer.top.extend([node.output[0]])

    #-------init params------------
    layer.type = 'Eltwise'
    layer.eltwise_param.operation = 0

    return layer

# 没写完呢!!! -_-
def _Clip(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])

    #-------init params------------
    layer.type = 'Clip'
    #layer.eltwise_param.operation = 0
    clip_data = node.input[1:]
    thresh = []
    for key in clip_data:
        thresh.append(initializers[key])
    thresh = list(map(float,thresh))
    
    return layer

def _Concat(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend(node.input)
    layer.top.extend([node.output[0]])

    #-------init params------------
    layer.type = 'Concat'
    attr = proto2dict(node.attribute)
    layer.concat_param.axis = attr['axis'].i
    
    return layer

def _Transpose(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend(node.input)
    layer.top.extend([node.output[0]])
    
    #------init params--------
    layer.type = 'Permute'
    attr = proto2dict(node.attribute)
    index = attr['perm'].ints
    #print(index)
    layer.permute_param.order.extend(index)
    return layer

def _ShuffleChannel(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.output[0]
    layer.bottom.extend(node.input)
    layer.top.extend([node.output[0]])

    #------init params--------
    layer.type = 'Permute'
    attr = proto2dict(node.attribute)
    group = attr['group']
    layer.shuffle_channel_param.group = group.i
    return layer

def _Resize(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])

    #------init params--------
    layer.type = 'Interp'
    attr = proto2dict(node.attribute)
    shape = attr['shape']
    #print(shape)
    layer.interp_param.height = shape.ints[3]
    layer.interp_param.width = shape.ints[4]
    return layer

def _UpSampleDeconv(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])

    #------infer param---------
    attr = proto2dict(node.attribute)
    shape_in = attr['shape_in']
    shape_out = attr['shape_out']
    #print(shape_in)
    print(shape_out)
    factor = shape_out.ints[2]//shape_in.ints[2]
    c = shape_out.ints[1]
    k = 2 * factor - factor % 2

    #------init params--------
    import math
    layer.type = 'Deconvolution'
    layer.convolution_param.num_output = c
    layer.convolution_param.group = c
    layer.convolution_param.bias_term = False
    layer.convolution_param.kernel_size.extend([k])
    layer.convolution_param.stride.extend([factor])
    layer.convolution_param.pad.extend([int(math.ceil((factor - 1) / 2.))])
    return layer


# 没写完呢!!! -_-
def _Slice(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.output[0]
    layer.bottom.extend([node.input[0]])
    layer.top.extend([node.output[0]])

    #------init params--------
    layer.type = 'Crop'
    start = initializers[node.input[1]][0]
    end = initializers[node.input[2]][0]
    axis = initializers[node.input[3]][0]
    print(start,end,axis)
    #shape = attr['shape']
    #layer.crop_param.height = shape.ints[3]
    #layer.crop_param.width = shape.ints[4]
    return layer

def _Split(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend(node.output)
    
    #------init params--------
    layer.type = 'Slice'
    attrs = proto2dict(node.attribute)
    axis = attrs['axis'].i
    slice_point = []
    point = 0
    for i in attrs['split'].ints:
        point+=i
        slice_point.append(point)
    slice_point.pop() #len(slice_point)=len(top)-1
    layer.slice_param.axis = axis
    layer.slice_param.slice_point.extend(slice_point)
    return layer

def _LeakyRelu(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend(node.output)

    #------init params--------
    layer.type = 'ReLU'
    attrs = proto2dict(node.attribute)
    alpha = attrs['alpha'].f
    layer.relu_param.negative_slope = alpha
    return layer

# def _ReduceSum(node,initializers):
#     #-------init layer--------
#     layer = pb2.LayerParameter()
#     layer.name = node.output[0]
#     layer.bottom.extend([node.input[0]])
#     layer.top.extend(node.output)

#     #------init params--------
#     layer.type = 'Reduction'
#     attrs = proto2dict(node.attribute)
#     axis = attrs['axes'].ints[0]
#     layer.reduction_param.operation = 1
#     layer.reduction_param.axis = axis
#     return layer

def _ReduceSum(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend(node.output)

    #------init params--------
    layer.type = 'Convolution'
    # attrs = proto2dict(node.attribute)
    # axis = attrs['axes'].ints[0]
    # layer.reduction_param.operation = 1
    # layer.reduction_param.axis = axis
    layer.convolution_param.num_output = 1
    layer.convolution_param.bias_term = False
    layer.convolution_param.kernel_size.extend([1])
    layer.convolution_param.weight_filler.type = 'constant'
    layer.convolution_param.weight_filler.value = 1
    return layer

def _ReduceMean(node,initializers):
    #-------init layer--------
    layer = pb2.LayerParameter()
    layer.name = node.name
    layer.bottom.extend([node.input[0]])
    layer.top.extend(node.output)

    #------init params--------
    layer.type = 'Convolution'
    attrs = proto2dict(node.attribute)
    shape_in = attrs['shape_in']
    #print(shape_in)
    #------build kenerl----------
    c = shape_in.ints[1]
    layer.convolution_param.num_output = 1
    layer.convolution_param.bias_term = False
    layer.convolution_param.kernel_size.extend([1])
    layer.convolution_param.weight_filler.type = 'constant'
    layer.convolution_param.weight_filler.value = 1/c
    return layer

    

