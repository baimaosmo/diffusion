import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentNormalizer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, z):
        with torch.no_grad():
            self.mean = z.mean(dim=(0, 1), keepdim=True)
            self.std = z.std(dim=(0, 1), keepdim=True).clamp(min=self.eps)

    def transform(self, z):
        return (z - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResGCNBlock(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.lin = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj, gamma=None, beta=None):
        h = torch.matmul(adj, x)
        h = self.lin(h)
        h = self.norm(h)
        if gamma is not None and beta is not None:
            h = h * (1 + gamma) + beta
        return x + self.drop(self.act(h))


class ConditionalDiffusionGCN(nn.Module):
    def __init__(self, V=116, K=64, hidden=256, depth=6, num_classes=2, dropout=0.1,
                 topo_dim=50, use_film=True, use_topology_conditioning=True):
        super().__init__()
        self.use_film = use_film
        self.use_topology_conditioning = use_topology_conditioning
        self.in_proj = nn.Linear(K, hidden)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.label_emb = nn.Embedding(num_classes, hidden)
        self.topo_mlp = (nn.Sequential(nn.Linear(topo_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                         if use_topology_conditioning else None)
        cond_out = hidden * 2 if use_film else hidden
        self.cond_proj = nn.Linear(hidden, cond_out)
        self.adj_param = nn.Parameter(torch.randn(V, V))
        self.layers = nn.ModuleList([ResGCNBlock(hidden, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, K)

    def get_normalized_adj(self):
        A = torch.sigmoid(self.adj_param + self.adj_param.transpose(0, 1))
        return A / A.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def forward(self, x, t, y, mu_g):
        h = self.in_proj(x)
        cond = self.time_mlp(t)[:, None, :] + self.label_emb(y)[:, None, :]
        if self.use_topology_conditioning:
            if mu_g is None:
                raise ValueError("mu_g is required when topology conditioning is enabled")
            cond = cond + self.topo_mlp(mu_g)[:, None, :]
        cond_params = self.cond_proj(cond)
        adj = self.get_normalized_adj()
        if self.use_film:
            gamma, beta = cond_params.chunk(2, dim=-1)
            for layer in self.layers:
                h = layer(h, adj, gamma, beta)
        else:
            h = h + cond_params
            for layer in self.layers:
                h = layer(h, adj)
        return self.out_proj(self.out_norm(h))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, num_heads=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * ff_mult, hidden),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, gamma=None, beta=None):
        h = self.norm1(x)
        if gamma is not None and beta is not None:
            h = h * (1 + gamma) + beta
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)

        h = self.norm2(x)
        if gamma is not None and beta is not None:
            h = h * (1 + gamma) + beta
        return x + self.drop2(self.ff(h))


class ConditionalDiffusionTransformer(nn.Module):
    def __init__(
        self,
        V=116,
        K=64,
        hidden=256,
        depth=6,
        num_classes=2,
        dropout=0.1,
        topo_dim=50,
        use_film=True,
        use_topology_conditioning=True,
        num_heads=4,
    ):
        super().__init__()
        self.use_film = use_film
        self.use_topology_conditioning = use_topology_conditioning
        self.in_proj = nn.Linear(K, hidden)
        self.node_emb = nn.Parameter(torch.randn(1, V, hidden) * 0.02)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.label_emb = nn.Embedding(num_classes, hidden)
        self.topo_mlp = (nn.Sequential(nn.Linear(topo_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                         if use_topology_conditioning else None)
        cond_out = hidden * 2 if use_film else hidden
        self.cond_proj = nn.Linear(hidden, cond_out)
        self.layers = nn.ModuleList([TransformerBlock(hidden, num_heads=num_heads, dropout=dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, K)

    def forward(self, x, t, y, mu_g):
        h = self.in_proj(x) + self.node_emb[:, : x.size(1), :]
        cond = self.time_mlp(t)[:, None, :] + self.label_emb(y)[:, None, :]
        if self.use_topology_conditioning:
            if mu_g is None:
                raise ValueError("mu_g is required when topology conditioning is enabled")
            cond = cond + self.topo_mlp(mu_g)[:, None, :]
        cond_params = self.cond_proj(cond)
        if self.use_film:
            gamma, beta = cond_params.chunk(2, dim=-1)
            for layer in self.layers:
                h = layer(h, gamma, beta)
        else:
            h = h + cond_params
            for layer in self.layers:
                h = layer(h)
        return self.out_proj(self.out_norm(h))


class LatentDDPM:
    def __init__(self, model, T=1000, device=None):
        self.model = model
        self.T = T
        self.device = device or next(model.parameters()).device

        betas = torch.linspace(1e-4, 0.02, T, device=self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_om = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_var = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-8)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_ac, t, x0.shape) * x0 + extract(self.sqrt_om, t, x0.shape) * noise, noise

    @torch.no_grad()
    def p_sample(self, x_t, t, y, mu_g):
        eps = torch.clamp(self.model(x_t, t, y, mu_g), -2.0, 2.0)
        alpha_t = extract(self.alphas, t, x_t.shape)
        alpha_bar = extract(self.alphas_cumprod, t, x_t.shape)
        beta_t = extract(self.betas, t, x_t.shape)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar + 1e-6)) * eps)
        var = extract(self.posterior_var, t, x_t.shape)
        nonzero = (t != 0).float().view(-1, 1, 1)
        return mean + nonzero * torch.sqrt(var + 1e-6) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, n, V, K, y, mu_g):
        x = torch.randn(n, V, K, device=self.device)
        for i in range(self.T - 1, -1, -1):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            x = torch.clamp(self.p_sample(x, t, y, mu_g), -4.0, 4.0)
        return x


def extract(a, t, x_shape):
    out = a.gather(0, t)
    while len(out.shape) < len(x_shape):
        out = out.unsqueeze(-1)
    return out
