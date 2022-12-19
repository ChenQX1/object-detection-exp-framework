import deployment.operators as opt
from dgdet.src.utils.anchor import Anchors
from caffe.proto import caffe_pb2 as pb2

def get_priorbox(anchor):
    feat_p = anchor.pyramid_levels
    strides = [2**x for x in feat_p]
    sizes = anchor.sizes
    priorboxs = {}
    for (stride,size) in zip(strides,sizes):
        if stride not in priorboxs.keys():
            priorboxs[stride] = [size]
            continue
        priorboxs[stride].append(size)
    return priorboxs


def add_ssd_layers(clf_blobs,reg_blobs,priorboxs=None,num_class=1):
    layers = []
    # add clf layers
    bottoms_clf = []
    for blob in clf_blobs:
        permute = opt.Permute(blob,'permute_'+blob,order=[0,2,3,1])
        layers.append(permute)
        reshape = opt.Reshape('permute_'+blob,'reshape_'+blob,shape=[0,-1,1,1])
        layers.append(reshape)
        bottoms_clf.append('reshape_'+blob)
    cat1 = opt.Cat(bottoms_clf,'cat1',dim=1)
    layers.append(cat1)
    sigmoid = opt.Sigmoid('cat1','sigmoid')
    layers.append(sigmoid)

    # add reg layers
    bottoms_reg = []
    for blob in reg_blobs:
        permute = opt.Permute(blob,'permute_'+blob,order=[0,2,3,1])
        layers.append(permute)
        reshape = opt.Reshape('permute_'+blob,'reshape_'+blob,shape=[0,-1,1,1])
        layers.append(reshape)
        bottoms_reg.append('reshape_'+blob)
    cat2 = opt.Cat(bottoms_reg,'cat2',dim=1)
    layers.append(cat2)

    # add priobox
    bottoms_prioboxs =[]
    for idx,(stride,size) in enumerate(priorboxs.items()):
        blob = clf_blobs[idx]
        blobs = [blob,'data']
        priorbox = opt.PriorBox(blobs,'priorbox_'+blob,stride,size)
        bottoms_prioboxs.append('priorbox_'+blob)
        layers.append(priorbox)
    cat3 = opt.Cat(bottoms_prioboxs,'cat3',dim=2)
    layers.append(cat3)
    
    blobs = ['cat2','sigmoid','cat3']
    output = opt.DetectionOutput(blobs,num_class)
    layers.append(output)
    net_param = pb2.NetParameter()
    net_param.layer.extend(layers)
    return net_param

# test for priorbox
def main():
    anchor = Anchors(
            pyramid_levels = [3,4,5,6,7],
            sizes = [16,32,64,128,256],
            scales=[1],
        )
    priorbox = get_priorbox(anchor)
    print(priorbox)

# test for add layers
def main2():
    clf_blobs = ['1','2','3','4','5']
    reg_blobs = ['6','7','8','9','10']
    anchor = Anchors(
            pyramid_levels = [3,4,5,6,7],
            sizes = [16,32,64,128,256],
            scales=[1],
        )
    priorbox = get_priorbox(anchor)
    layers = add_ssd_layers(clf_blobs,reg_blobs,priorbox,1)
    print(layers)

if __name__ == "__main__":
    main2()