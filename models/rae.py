import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_diffusion.graph_utils import build_topk_adjacency, normalize_adjacency


class GraphConv(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, Wn):
        h = torch.bmm(Wn, H)
        h = self.lin(h)
        h = self.norm(h)
        h = F.silu(h)
        return self.drop(h)


class GraphEncoder(nn.Module):
    def __init__(self, V=116, hidden=256, layers=3, topk=20, dropout=0.1):
        super().__init__()
        self.topk = topk
        self.hidden = hidden
        self.in_proj = nn.Sequential(
            nn.Linear(V, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.gconvs = nn.ModuleList([GraphConv(hidden, dropout=dropout) for _ in range(layers)])

    def forward(self, fc):
        B, V, _ = fc.shape
        W = build_topk_adjacency(fc, topk=self.topk, use_abs=True).to(fc.device)
        Wn = normalize_adjacency(W)
        H = self.in_proj(fc.reshape(B * V, V)).view(B, V, self.hidden)
        for gconv in self.gconvs:
            H = H + gconv(H, Wn)
        return H


class TopoGuidedDecoder(nn.Module):
    def __init__(self, K=64, topo_dim=50, use_film=True, use_topology_conditioning=True):
        super().__init__()
        self.K = K
        self.use_film = use_film
        self.use_topology_conditioning = use_topology_conditioning
        if use_topology_conditioning:
            self.topo_mlp = nn.Sequential(nn.Linear(topo_dim, 128), nn.SiLU(), nn.Linear(128, 256))
            cond_out = K * 2 if use_film else K
            self.cond_proj = nn.Linear(256, cond_out)
        else:
            self.topo_mlp = None
            self.cond_proj = None
        self.dec_proj = nn.Linear(K, K, bias=False)

    def forward(self, z, mu_g):
        if self.use_topology_conditioning:
            if mu_g is None:
                raise ValueError("mu_g is required when topology conditioning is enabled")
            cond_params = self.cond_proj(self.topo_mlp(mu_g))
            if self.use_film:
                gamma, beta = cond_params.chunk(2, dim=-1)
                z_cond = z * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            else:
                z_cond = z + cond_params.unsqueeze(1)
        else:
            z_cond = z

        z_proj = self.dec_proj(z_cond)
        recon = torch.tanh(torch.matmul(z_proj, z_proj.transpose(1, 2)) / math.sqrt(self.K))
        return recon - torch.diag_embed(torch.diagonal(recon, dim1=1, dim2=2))


class GraphRAE(nn.Module):
    def __init__(self, V=116, K=64, hidden=256, layers=3, topk=20, dropout=0.1,
                 topo_dim=50, use_film=True, use_topology_conditioning=True):
        super().__init__()
        self.V = V
        self.K = K
        self.encoder = GraphEncoder(V=V, hidden=hidden, layers=layers, topk=topk, dropout=dropout)
        self.to_z = nn.Linear(hidden, K)
        self.z_norm = nn.LayerNorm(K)
        self.decoder = TopoGuidedDecoder(
            K=K, topo_dim=topo_dim, use_film=use_film,
            use_topology_conditioning=use_topology_conditioning,
        )

    def decode(self, z, mu_g):
        return self.decoder(z, mu_g)

    def forward(self, fc, mu_g):
        z = self.z_norm(self.to_z(self.encoder(fc)))
        return self.decode(z, mu_g), z


def fc_recon_loss(recon, target):
    mask = 1.0 - torch.eye(recon.size(1), device=recon.device).unsqueeze(0)
    return F.mse_loss(recon * mask, target * mask)
