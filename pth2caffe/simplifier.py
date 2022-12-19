import onnx
from onnx import helper
from onnx import shape_inference
from onnx import optimizer
from pth2caffe.graph2 import Finder
from pth2caffe.graph2 import delateSubGraph,replaceSubGraph
from pth2caffe.utils import proto2np,infer_shape


def show_pass():
    all_passes = optimizer.get_available_passes()
    print("Available optimization passes:")
    for p in all_passes:
        print(p)

def optimize_graph(onnx_model):
    passes = [
        'extract_constant_to_initializer',
        'fuse_bn_into_conv',
        'eliminate_nop_dropout',
        'eliminate_identity',
        ]
    onnx_model = optimizer.optimize(onnx_model,passes)
    return onnx_model

def optimize_resize(nodes,initializer):
    finder = Finder(nodes,initializer)
    querys = [
        ['Shape','Gather','Cast','Mul','Cast','Floor','Unsqueeze'],
        ['Shape','Slice'],
        ['Concat','Cast','Concat'],
    ]
    for query in querys:
        sub_graphs = finder.findSubGraph(query)
        nodes = delateSubGraph(nodes,sub_graphs)
    return nodes

def optimize_shuffle(nodes,initializer):
    finder = Finder(nodes,initializer)
    query = ['Reshape','Transpose','Reshape']
    sub_graphs = finder.findSubGraph(query)
    #replace sub graph with new nodes
    new_nodes = []
    for item in sub_graphs:
        input = item[0].input[0]
        shape = item[0].input[1]
        bs,group,channel,h,w = initializer[shape]
        output = item[-1].output
        #------make new ShuffleChannel node---------
        node = helper.make_node(
            inputs=input,
            outputs=output,
            op_type='ShuffleChannel',
            group = group,
        )
        new_nodes.append(node)
    nodes = replaceSubGraph(nodes,sub_graphs,new_nodes)
    return nodes

def optimize_view(nodes,initializer):
    finder = Finder(nodes,initializer)
    query = ['Shape','Gather','Unsqueeze','Concat']
    sub_graphs = finder.findSubGraph(query)
    nodes = delateSubGraph(nodes,sub_graphs)
    return nodes

def optimize_softmax(nodes,initializer):
    finder = Finder(nodes,initializer)
    query = ['Exp','ReduceSum','Div']
    sub_graphs = finder.findSubGraph(query)
    new_nodes = []
    for item in sub_graphs:
        input = item[0].input
        output = item[-1].output
        #------make new sotfmax node---------
        node = helper.make_node(
            inputs=input,
            outputs=output,
            op_type='Softmax',
            axes=1,                 #AttributeType 
        )
        new_nodes.append(node)
    nodes = replaceSubGraph(nodes,sub_graphs,new_nodes)
    return nodes

def main():
    model = onnx.load('onnx_model/slice.onnx')
    model = optimize_graph(model)
    model = infer_shape(model)
    nodes = list(model.graph.node)
    initializer = model.graph.initializer
    initializer = proto2np(initializer)
    for item in nodes:
        for idx,k in enumerate(item.input):
            if idx>0:
                n = initializer[k]
                print(n[0])
        #         if n==9223372036854775807:
        #             n = -1
        #         print(n)    
        # print(item)
    #nodes = optimize_resize(nodes,initializer)
    #nodes = optimize_shuffle(nodes,initializer)
    #nodes = optimize_view(nodes,initializer)
    #nodes = optimize_softmax(nodes,initializer)


if __name__ == "__main__":
    main()

