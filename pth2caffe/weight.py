import onnx
import numpy as np
from pth2caffe.utils import proto2dict

def _Conv(node,initializer):
    #init result
    params = {
        'name':None,
        'data':[]
    }
    name = node.name
    params['name'] = name
    w_name = node.input[1]
    weight = initializer[w_name]
    #print(weight.shape)
    params['data'].append(weight)
    if len(node.input)==3:
        b_name = node.input[2]
        bias = initializer[b_name]
        params['data'].append(bias)
    return params

def _BatchNormalization(node,initializer):
    params_norm = {
        'name':None,
        'data':[]
    }
    params_scale = {
        'name':None,
        'data':[]
    }
    #print(node.input)
    layer_name = node.name
    #print(node.input[1:])
    w,bias,mean,var = node.input[1:]  #weight name
    w = initializer[w]
    bias = initializer[bias]
    mean = initializer[mean]
    var = initializer[var]

    #-----init norm weight----------
    norm_layer = '{}_norm'.format(layer_name)
    params_norm['name'] = norm_layer
    params_norm['data'].append(mean)
    params_norm['data'].append(var)
    params_norm['data'].append(np.array([1.]))

    #-----init scale weight--------
    scale_layer = '{}_scale'.format(layer_name)
    params_scale['name'] = scale_layer
    params_scale['data'].append(w)
    params_scale['data'].append(bias)

    return [params_norm,params_scale]

def _Gemm(node,initializer):
    params_norm = {
        'name':None,
        'data':[]
    }
    w = node.input[1]
    w = initializer[w]
    params_norm['name'] = node.name
    params_norm['data'].append(w)
    if len(node.input)==3:
        bias = node.input[2]
        bias = initializer[bias]
        params_norm['data'].append(bias)
    return params_norm


'''
copy form https://gitlab.deepglint.com/leimao/pytorch2caffe/-/blob/master/ConvertLayer_caffe.py
'''
def FillBilinear(ch, k):
    blob = np.zeros(shape=(ch, 1, k, k))

    """ Create bilinear weights in numpy array """
    bilinear_kernel = np.zeros([k, k], dtype=np.float32)
    scale_factor = (k + 1) // 2
    if k % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(k):
        for y in range(k):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)

    for i in range(ch):
        blob[i, 0, :, :] = bilinear_kernel
    return blob


def _UpSampleDeconv(node,initializer):
    #--------init result--------
    params = {
        'name':None,
        'data':[]
    }
    name = node.name
    params['name'] = name

    #---------infer weight--------
    attr = proto2dict(node.attribute)
    shape_in = attr['shape_in']
    shape_out = attr['shape_out']
    factor = shape_out.ints[2]//shape_in.ints[2]
    c = shape_out.ints[1]
    k = 2 * factor - factor % 2
    w = FillBilinear(c,k)
    params['data'].append(w)
    return params

load_func = {
    'Conv':_Conv,
    'ConvTranspose':_Conv,
    'BatchNormalization':_BatchNormalization,
    'Gemm':_Gemm,
    'Resize':_UpSampleDeconv,
}

def convert_weight(node,initializer):
    op_type = node.op_type
    if op_type not in load_func:
        print('{} weight is not implement!'.format(op_type))
        return None
    else:
        params = load_func[op_type](node,initializer)
        return params
