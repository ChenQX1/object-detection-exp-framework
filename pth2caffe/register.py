import onnx
import pth2caffe.operators as op

Operators = {
    'Conv':op._Conv,
    'ConvTranspose':op._Conv,
    'BatchNormalization':op._BatchNormalization,
    'MaxPool':op._Pool,
    'AveragePool':op._Pool,
    'Gemm':op._Gemm,
    'Relu':op._Relu,
    'Add':op._Add,
    'Sigmoid':op._Sigmoid,
    'Softmax':op._Softmax,
    'GlobalAveragePool':op._GlobalAveragePool,
    'Flatten':op._Flatten,
    'Reshape':op._Flatten,
    'Mul':op._Mul,
    'Clip':op._Clip,
    'Concat':op._Concat,
    'Transpose':op._Transpose,
    'ShuffleChannel':op._ShuffleChannel,
    #'Resize':op._Resize,
    'Resize':op._UpSampleDeconv,
    #'Slice':op._Slice,
    'Split':op._Split,
    'LeakyRelu':op._LeakyRelu,
    'ReduceSum':op._ReduceSum,
    'ReduceMean':op._ReduceMean,
}

 
def convert_layer(node,initializers):
    op_type = node.op_type
    layer = None
    if op_type not in Operators.keys():
        print('[warning!] {} is not support!'.format(op_type))
        #print(node)
    else:
        #print('convert {}'.format(op_type))
        layer = Operators[op_type](node,initializers)
    return layer