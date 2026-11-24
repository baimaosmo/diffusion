import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def build_topk_adjacency(fc, topk=20, use_abs=True):
    w = fc.detach()
    w = w - torch.diag_embed(torch.diagonal(w, dim1=1, dim2=2))
    if use_abs:
        w = w.abs()

    _, V, _ = w.shape
    _, idx = torch.topk(w, k=min(topk, V - 1), dim=-1, largest=True, sorted=False)

    mask = torch.zeros_like(w)
    mask.scatter_(-1, idx, 1.0)
    mask = ((mask + mask.transpose(1, 2)) > 0).float()
    return w * mask


def normalize_adjacency(W, eps=1e-6):
    deg = W.sum(dim=-1).clamp(min=eps)
    inv_sqrt = torch.rsqrt(deg)
    return inv_sqrt.unsqueeze(2) * W * inv_sqrt.unsqueeze(1)


def generate_ba_mask(V, m_attach=5):
    m_safe = max(1, min(m_attach, V - 1))
    G = nx.barabasi_albert_graph(n=V, m=m_safe, seed=None)
    mask_np = nx.to_numpy_array(G, dtype=np.float32)
    return torch.from_numpy(mask_np)


def flatten_upper(fc, k=1):
    V = fc.size(1)
    idx = torch.triu_indices(V, V, offset=k, device=fc.device)
    return fc[:, idx[0], idx[1]]


def repair_and_filter(real_flat, fake_flat, top_k_ratio=1.0):
    if torch.isnan(fake_flat).any():
        return real_flat[:10]

    fake_flat = torch.clamp(fake_flat, -1.0, 1.0)
    real_mean = real_flat.mean(dim=0, keepdim=True)
    sim = F.cosine_similarity(fake_flat, real_mean + 1e-6, dim=1)
    k = min(len(fake_flat), max(1, int(len(fake_flat) * top_k_ratio)))
    _, top_idx = torch.topk(sim, k=k)
    return fake_flat[top_idx]
