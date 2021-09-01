import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class SimpleGatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.U = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        self.edge_mlp = nn.Sequential(
                            nn.ReLU(),
                            # nn.Linear(output_dim, output_dim),
                            # nn.ReLU(),
                            nn.Linear(output_dim, output_dim))
        self.edge_linear = nn.Linear(input_dim, output_dim)

    def forward(self, g, h, e, u, revmap):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e
        g.edata['u']  = u
        g.edata['urev'] = u[revmap]
        g.edata['Urev'] = self.U(g.edata['urev']) 
        g.edata['Ce'] = self.C(e)
        g.ndata['deg'] = g.in_degrees().reshape(-1, 1)

        # g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))     # original GatedGCN
        # g.edata['e'] = g.edata['DEh'] + g.edata['Ce']    # original GatedGCN
            
        g.apply_edges(self.compute_e)
        
        # g.edata['sigma'] = torch.sigmoid(g.edata['e'])                      # original GatedGCN
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))     # original GatedGCN
        
        g.update_all(self.compute_u, fn.sum("u", "h"))
        
        # g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))    # original GatedGCN
        # g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6) # original GatedGCN
         
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        u = g.edata['u'] # result of graph convolution
        
        if self.batch_norm:
            # pass
            h = self.bn_node_h(h) # batch normalization  # original GatedGCN
            # e = self.bn_node_e(e) # batch normalization  # original GatedGCN
        
        # h = F.relu(h) # original GatedGCN
        # e = F.relu(e) # # original GatedGCN
        if self.residual:
            # pass
            h = h_in + h # residual connection # original GatedGCN
            # e = e_in + e # residual connection # original GatedGCN
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e, u

    def compute_e(self, edges):
        # Dh = self.D(edges.src['h'] - edges.data['urev'])
        Dh = edges.src['Dh']
        e = self.edge_mlp(Dh + edges.dst["Eh"] +  edges.data["Ce"])
        # e = self.edge_mlp(Dh + edges.dst["Eh"] + edges.data["Ce"] + edges.data["Urev"])
        sigma = torch.sigmoid(e)
        e = edges.data['e'] + self.bn_node_e(e)
        # e = edges.data['e'] + F.relu(self.bn_node_e(e)) 

        # print(e.norm() * edges.dst["deg"].float().mean() / len(e))
        return {"e": e, "sigma": sigma}
    

    def compute_u(self, edges):
        # Bh = self.B(edges.src["h"] - edges.data["urev"])
        Bh = edges.src["Bh"]
        # return {"u": Bh * edges.data["sigma"] / edges.dst["sum_sigma"]}
        return {"u": edges.data['e']}
        # return {"u": edges.data['e'] * Bh}
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

