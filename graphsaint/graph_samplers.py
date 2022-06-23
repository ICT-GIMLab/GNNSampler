from graphsaint.globals import *
import numpy as np
import scipy.sparse
import time
import math
import pdb
from math import ceil
import graphsaint.cython_sampler as cy
#from neigh_analysis import *
from line_profiler import LineProfiler

class GraphSampler:
    
    def __init__(self, adj_train, node_train, size_subgraph, args_preproc):

        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)
        self.size_subgraph = size_subgraph
        self.name_sampler = 'None'
        self.node_subgraph = None
        self.preproc(**args_preproc)

    def preproc(self, **kwargs):
        pass

    def par_sample(self, stage, **kwargs):
        return self.cy_sampler.par_sample()

    def _helper_extract_subgraph(self, node_ids):

        node_ids = np.unique(node_ids)
        node_ids.sort()
        orig2subg = {n: i for i, n in enumerate(node_ids)}
        n = node_ids.size
        indptr = np.zeros(node_ids.size + 1)
        indices = []
        subg_edge_index = []
        subg_nodes = node_ids
        #counter = 0
        for nid in node_ids:
            idx_s, idx_e = self.adj_train.indptr[nid], self.adj_train.indptr[nid + 1]
            neighs = self.adj_train.indices[idx_s : idx_e]
            for i_n, n in enumerate(neighs):
                #counter = counter + 1
                if n in orig2subg:                    
                    indices.append(orig2subg[n])
                    indptr[orig2subg[nid] + 1] += 1
                    subg_edge_index.append(idx_s + i_n)
        indptr = indptr.cumsum().astype(np.int64)
        indices = np.array(indices)
        subg_edge_index = np.array(subg_edge_index)
        data = np.ones(indices.size)
        assert indptr[-1] == indices.size == subg_edge_index.size
        return indptr, indices, data, subg_nodes, subg_edge_index

# --------------------------------------------------------------------
# [BELOW] python wrapper for parallel samplers implemented with Cython
# --------------------------------------------------------------------

class NodeSamplingVanillaPython(GraphSampler):

    """
    This class is just to showcase how you can write the graph sampler in pure python.
    The simplest and most basic sampler: just pick nodes uniformly at random and return the
    node-induced subgraph.
    """
    
    def __init__(self, adj_train, node_train, size_subgraph):
        super().__init__(adj_train, node_train, size_subgraph, {})
        self.n = 0
        self.s = 0

    def par_sample(self, stage, **kwargs):
        node_ids = np.random.choice(self.node_train, self.size_subgraph)
        ret = self._helper_extract_subgraph(node_ids)
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret

    def preproc(self):
        self.weight_time = 0
        self.dataset_name = "vanilla"

    def configReturn(self):
        return self.n, self.s, self.weight_time, self.dataset_name

class LocalityBasedNodeSampling(GraphSampler):

    """
    This class is implemented using locality that explored in the adj_train 
    to optimize the sampling process.
    Inputs:
        adj_train       scipy sparse CSR matrix of the training graph           
        node_train      1D np array storing the indices of the training nodes   
        size_subgraph   int, the (estimated) number of nodes in the subgraph    
        load_weight     bool, the flag of whether to load the weight that are pre-computed
    """
    
    def __init__(self, adj_train, node_train, size_subgraph, load_weight, dataset):
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)
        self.size_subgraph = size_subgraph
        self.load_weight = load_weight
        self.dataset = dataset
        self.weight = None
        self.preproc()
        
        
    def par_sample(self, stage, **kwargs):
        if self.load_weight:
            weight = np.load("./precomputed_weight/weight_" + self.dataset +".npy")
            node_ids = np.random.choice(self.node_train, self.size_subgraph, p = weight)
        else:
            node_ids = np.random.choice(self.node_train, self.size_subgraph, p = self.weight) 
            
        ret = self._helper_extract_subgraph(node_ids) 
        ret = list(ret)
        for i in range(len(ret)):
            ret[i] = [ret[i]]
        return ret

    def preproc(self):
        self.n = 4
        self.s = 0.775
        if self.load_weight:
            self.weight_time = 0
            self.dataset_name = self.dataset
            print("Sampling weight requires no extra computation")
        else:
            print('#'*60)
            print("#"*14,"Computing sampling weight ...","#"*15)
            print('#'*60)
            t0 = time.time()
            self.dist_locality, self.dataset_name = adj_train_analysis(self.adj_train, self.n, self.s)
            self.weight = self.dist_locality[self.node_train]/sum(self.dist_locality[self.node_train])
            print('\033[1;33;44m total sampled number', len(np.nonzero(self.weight)[0]), '\033[0m')
            print("total sampled number ", len(np.nonzero(self.weight)[0]))
            t1 = time.time()
            print('\033[1;33;44m weight computation time', t1-t0, '\033[0m')
            self.weight_time = t1-t0
            np.save(self.dataset + str(self.n) + "_" + str(self.s) + ".npy", self.weight)
            print("Weight Saved!")
    
    def configReturn(self):
        return self.n, self.s, self.weight_time, self.dataset_name
