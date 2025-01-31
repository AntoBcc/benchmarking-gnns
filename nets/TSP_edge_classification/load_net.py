"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.TSP_edge_classification.gated_gcn_net import GatedGCNNet
from nets.TSP_edge_classification.simplegated_gcn_net import SimpleGatedGCNNet
from nets.TSP_edge_classification.gcn_net import GCNNet
from nets.TSP_edge_classification.gat_net import GATNet
from nets.TSP_edge_classification.graphsage_net import GraphSageNet
from nets.TSP_edge_classification.gin_net import GINNet
from nets.TSP_edge_classification.mo_net import MoNet as MoNet_
from nets.TSP_edge_classification.mlp_net import MLPNet
from nets.TSP_edge_classification.ring_gnn_net import RingGNNNet
from nets.TSP_edge_classification.three_wl_gnn_net import ThreeWLGNNNet

from nets.TSP_edge_classification.gated_gcn_net_hybrid import GatedGCNNet as GatedGCNNet_hybrid


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def SimpleGatedGCN(net_params):
    return SimpleGatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def RingGNN(net_params):
    return RingGNNNet(net_params)

def ThreeWLGNN(net_params):
    return ThreeWLGNNNet(net_params)

def GatedGCN_hybrid(net_params):
    return GatedGCNNet_hybrid(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'SimpleGatedGCN': SimpleGatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'MLP': MLP,
        'RingGNN': RingGNN,
        '3WLGNN': ThreeWLGNN,
        'GatedGCN-hybrid' : GatedGCNNet_hybrid
    }
    
    return models[MODEL_NAME](net_params)