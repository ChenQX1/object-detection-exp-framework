import caffe
from caffe.proto import caffe_pb2 as pb2

'''
convert pytorch layer to caffe net params
'''

def Permute(bottom,top,order):
    layer = pb2.LayerParameter()
    layer.type = 'Permute'
    layer.name = top
    layer.bottom.extend([bottom])
    layer.top.extend([top])
    # update param
    layer.permute_param.order.extend(order) #type(order) = list
    return layer

def Reshape(bottom,top,shape=[0,-1,1,1]):
    layer = pb2.LayerParameter()
    layer.type = 'Reshape'
    layer.name = top
    layer.bottom.extend([bottom])
    layer.top.extend([top])
    # update param
    layer.reshape_param.shape.dim.extend(shape) #type(shape) = list
    return layer

def Cat(bottoms,top,dim=1):
    layer = pb2.LayerParameter()
    layer.type = 'Concat'
    layer.name = top
    layer.bottom.extend(bottoms)
    layer.top.extend([top])
    # update param
    layer.concat_param.axis = dim
    return layer

def Sigmoid(bottom,top):
    layer = pb2.LayerParameter()
    layer.type = 'Sigmoid'
    layer.name = top
    layer.bottom.extend([bottom])
    layer.top.extend([top])
    return layer

def PriorBox(bottoms,top,stride,anchor):
    layer = pb2.LayerParameter()
    layer.type = 'PriorBox'
    layer.name = top
    layer.bottom.extend(bottoms)
    layer.top.extend([top])
    # update param
    layer.prior_box_param.min_size.extend(anchor)
    layer.prior_box_param.step = stride
    layer.prior_box_param.flip = False
    layer.prior_box_param.clip = False
    layer.prior_box_param.variance.extend([0.1,0.1,0.2,0.2])
    layer.prior_box_param.offset = 0.5
    return layer

'''
bottoms: [regresson_cat, classfication_cat, priorbox_cat]
include : 1-test
code_type : 2-CENTER_SIZE
'''
INF = 100
def DetectionOutput(bottoms,num_class,nms=0.1):
    layer = pb2.LayerParameter()
    layer.type = 'DetectionOutput'
    layer.name = 'detection_out'
    layer.bottom.extend(bottoms)
    layer.top.extend(['detection_out'])
    # update include
    include = layer.include.add()
    include.phase = 1
    # update param
    layer.detection_output_param.num_classes = num_class
    layer.detection_output_param.share_location = True
    layer.detection_output_param.background_label_id = 100
    layer.detection_output_param.code_type = 2
    layer.detection_output_param.keep_top_k = INF
    layer.detection_output_param.confidence_threshold = 0.01
    # update nms_param
    layer.detection_output_param.nms_param.nms_threshold = nms
    layer.detection_output_param.nms_param.top_k = 200
    return layer


def main():
    net = DetectionOutput(bottoms=['cat1','cat2','cat3'],top='output',num_class=2,nms=0.1)
    print(net)

if __name__ == "__main__":
    main()
    