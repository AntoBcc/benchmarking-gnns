import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.simplegated_gcn_layer import SimpleGatedGCNLayer
from layers.mlp_readout_layer import MLPReadout

class SimpleGatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.layer_type = {
            "edgereprfeat": SimpleGatedGCNLayer,
        }.get(net_params['layer_type'], SimpleGatedGCNLayer)
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ self.layer_type(hidden_dim, hidden_dim, dropout,
                                                      self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(self.layer_type(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(4*out_dim, n_classes)
        
    def forward(self, g, h, e):
        # Check that edges in the batched graph are properly ordered
        u, v = g.edges()
        bnumedges = g.batch_num_edges()
        revmap = []
        for i, ne in enumerate(bnumedges):
            start = torch.sum(bnumedges[:i])
            end = torch.sum(bnumedges[:i+1])
            m = end - start  
            # assert (u[start:(start+m//2)] == v[(start+m//2):end]).all()
            # assert (v[start:(start+m//2)] == u[(start+m//2):end]).all()
            revmap += [torch.arange(start + m//2, end).to(self.device),
                       torch.arange(start, start + m//2).to(self.device)]
        revmap = torch.cat(revmap)
        # assert (u[revmap] == v).all()
        # assert (v[revmap] == u).all()

        h = self.embedding_h(h.float())
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        e = self.embedding_e(e.float())
        u = torch.zeros_like(e).to(self.device)
        
        for conv in self.layers:
            h, e, u = conv(g, h, e, u, revmap)
        g.ndata['h'] = h
        g.edata['e'] = e
        g.edata['u'] = u
        g.edata['erev'] = e[revmap]
        g.edata['urev'] = u[revmap]
        
        def _edge_feat(edges):
            e = edges.data['e']
            erev  = edges.data['erev']
            e = torch.cat([e, erev, edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}
        g.apply_edges(_edge_feat)
        
        return g.edata['e']
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss
    