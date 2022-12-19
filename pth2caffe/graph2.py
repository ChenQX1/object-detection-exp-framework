import onnx

class Finder(object):
    def __init__(self,nodes,initializer):
        '''
        nodes: list
        initializer: dict
        '''
        if isinstance(nodes,list) is False:
            raise ValueError('nodes type is wrong !')
        if isinstance(initializer,dict) is False:
            raise ValueError('initializer type is wrong !')
        self.onnx_nodes = nodes
        self.const_name = list(initializer.keys())
        self.blobs = []
        self.tree = {}
        self.edges = {}
        self._init_blobs()
        self.sub_graphs = []

    def _init_blobs(self):
        #update blobs
        for node in self.onnx_nodes:
            for item in node.input:
                if item in self.const_name:
                    continue
                self.blobs.append(item)
            for item in node.output:
                self.blobs.append(item)
        self.blobs = list(set(self.blobs))
        #update tree
        for item in self.blobs:
            self.tree[item] = []
        for node in self.onnx_nodes:
            inputs = node.input
            outputs = node.output
            outputs = list(outputs)
            for item in inputs:
                if item in self.const_name:
                    continue
                self.tree[item].extend(outputs)
        for (k,v) in self.tree.items():
            v = list(set(v))
        #update edge dict
        for item in self.onnx_nodes:
            for b in item.input:
                for e in item.output:
                    self.edges['{}->{}'.format(b,e)] = item
        #print(self.edges.keys())

    #add sub route into self.sub_graphs
    def _find(self,route,query_ops):
        if len(query_ops)==0:
            self.sub_graphs.append(route)
            return
        start = route[-1]
        child = self.tree[start]
        for item in child:
            op = self.edges['{}->{}'.format(start,item)]
            if op.op_type==query_ops[0]:
                q = query_ops.copy()
                r = route.copy()
                q.pop(0)
                r.append(item)
                self._find(r,q)
    
    def _blob2nodes(self,route):
        result = []
        for i in range(len(route)-1):
            s = route[i]
            e = route[i+1]
            node = self.edges['{}->{}'.format(s,e)]
            result.append(node)
        return result

    def findSubGraph(self,query):
        self.sub_graphs = []
        for blob in self.blobs:
            route = [blob]
            self._find(route,query.copy())
        results = []
        for route in self.sub_graphs:
            nodes = self._blob2nodes(route)
            results.append(nodes)
        return results

def delateSubGraph(nodes,sub_nodes):
    union = []
    for item in sub_nodes:
        union.extend(item)
    result = []
    for node in nodes:
        if node not in union:
            result.append(node)
    return result

def replaceSubGraph(nodes,sub_graphs,new_nodes):
    result = []
    for item in nodes:
        replace = False
        for drop,new_node in zip(sub_graphs,new_nodes):
            if item in drop:
                replace = True
            if (item in drop) and (new_node not in result):
                result.append(new_node)
        if replace is False:
            result.append(item)
    return result

