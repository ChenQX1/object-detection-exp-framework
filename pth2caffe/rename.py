import onnx

def rename(onnx_model):
    nodes = onnx_model.graph.node
    output = [x.name for x in onnx_model.graph.output]
    mapping = {}
    # obtation layer name
    for item in nodes:
        outs = item.output
        for tmp in outs:
            if tmp in output:
                mapping[tmp] = item.name
    # rename blob
    for item in nodes:
        # rename input
        for i in range(len(item.input)):
            blob_name = item.input[i]
            if blob_name in mapping.keys():
                item.input[i] = mapping[blob_name]
        # rename output
        for i in range(len(item.output)):
            blob_name = item.output[i]
            if blob_name in mapping.keys():
                item.output[i] = mapping[blob_name]
    # reanme onnxmodel
    for item in onnx_model.graph.output:
        item.name = mapping[item.name]
    
    #outs = [x.name for x in onnx_model.graph.output]
    #print(outs)
    return onnx_model


def main():
    onnx_model = onnx.load('tmp/net.onnx')
    onnx_model = rename(onnx_model)
    onnx_model = onnx.save(onnx_model,'tmp/rename.onnx')


if __name__ == "__main__":
    main()




