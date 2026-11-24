import numpy as np
import torch

from brain_diffusion.graph_utils import build_topk_adjacency

try:
    import ripser
except ImportError:
    ripser = None


def extract_persistence_landscape(fc_matrix, resolution=50):
    if ripser is None:
        print("[Warning] 'ripser' is not installed. Persistence features will be zero vectors.")
        return torch.zeros(resolution, dtype=torch.float32)

    try:
        dist_mat = 1.0 - fc_matrix.cpu().numpy()
        np.fill_diagonal(dist_mat, 0)

        dgms = ripser.ripser(dist_mat, distance_matrix=True, maxdim=1)["dgms"]
        H0, H1 = dgms[0], dgms[1]
        H0 = H0[H0[:, 1] < np.inf]

        lifetimes_0 = H0[:, 1] - H0[:, 0]
        lifetimes_1 = H1[:, 1] - H1[:, 0] if len(H1) > 0 else np.zeros(1)
        all_life = np.concatenate([lifetimes_0, lifetimes_1])
        all_life = np.sort(all_life)[::-1]

        mu_G = np.zeros(resolution, dtype=np.float32)
        length = min(resolution, len(all_life))
        mu_G[:length] = all_life[:length]
        return torch.from_numpy(mu_G)
    except Exception:
        return torch.zeros(resolution, dtype=torch.float32)


def extract_avg_clustering_feature(fc_matrix, resolution=50, topk=20):
    try:
        _, adj = _build_binary_graph(fc_matrix, topk=topk)
        avg_clustering = _clustering_coefficients(adj).mean(dim=1)
        return _pack_features(avg_clustering[:, None], resolution, squeeze=fc_matrix.dim() == 2)
    except Exception:
        return torch.zeros(resolution, dtype=torch.float32)


def extract_strength_stats_feature(fc_matrix, resolution=50, topk=20):
    try:
        weighted, adj = _build_binary_graph(fc_matrix, topk=topk)
        strength = weighted.sum(dim=-1)
        fc_abs = weighted[weighted > 0]
        if fc_abs.numel() == 0:
            return torch.zeros(resolution, dtype=torch.float32)

        entropy = _safe_entropy(strength)
        values = torch.stack(
            [
                strength.mean(dim=1),
                strength.std(dim=1),
                strength.max(dim=1).values,
                strength.min(dim=1).values,
                torch.quantile(strength, 0.25, dim=1),
                torch.quantile(strength, 0.50, dim=1),
                torch.quantile(strength, 0.75, dim=1),
                adj.sum(dim=(1, 2)) / (adj.size(1) * (adj.size(1) - 1.0)),
                weighted.mean(dim=(1, 2)),
                entropy,
            ],
            dim=1,
        )
        return _pack_features(values, resolution, squeeze=fc_matrix.dim() == 2)
    except Exception:
        return torch.zeros(resolution, dtype=torch.float32)


def extract_efficiency_stats_feature(fc_matrix, resolution=50, topk=20):
    try:
        weighted, adj = _build_binary_graph(fc_matrix, topk=topk)
        dist = _shortest_path_distances(adj)
        finite = torch.isfinite(dist)
        off_diag = ~torch.eye(adj.size(1), dtype=torch.bool, device=adj.device).unsqueeze(0)
        valid = finite & off_diag
        inv_dist = torch.where(valid, 1.0 / dist.clamp(min=1.0), torch.zeros_like(dist))
        global_eff = inv_dist.sum(dim=(1, 2)) / (adj.size(1) * (adj.size(1) - 1.0))
        path_sum = torch.where(valid, dist, torch.zeros_like(dist)).sum(dim=(1, 2))
        path_count = valid.sum(dim=(1, 2)).clamp(min=1)
        char_path = path_sum / path_count
        degree = adj.sum(dim=-1)
        strength = weighted.sum(dim=-1)
        clustering = _clustering_coefficients(adj)

        values = torch.stack(
            [
                global_eff,
                char_path,
                clustering.mean(dim=1),
                clustering.std(dim=1),
                degree.mean(dim=1),
                degree.std(dim=1),
                strength.mean(dim=1),
                strength.std(dim=1),
                _degree_assortativity(adj, degree),
                _spectral_gap(adj),
            ],
            dim=1,
        )
        return _pack_features(values, resolution, squeeze=fc_matrix.dim() == 2)
    except Exception:
        return torch.zeros(resolution, dtype=torch.float32)


def extract_laplacian_spectrum_feature(fc_matrix, resolution=50, topk=20):
    try:
        weighted, _ = _build_binary_graph(fc_matrix, topk=topk)
        degree = weighted.sum(dim=-1)
        lap = torch.diag_embed(degree) - weighted
        eigvals = torch.linalg.eigvalsh(lap).clamp(min=0.0)
        eigvals = eigvals[:, : min(resolution, eigvals.size(1))]
        return _pack_features(eigvals, resolution, squeeze=fc_matrix.dim() == 2)
    except Exception:
        return torch.zeros(resolution, dtype=torch.float32)


def extract_graph_stats_feature(fc_matrix, resolution=50, topk=20):
    try:
        weighted, adj = _build_binary_graph(fc_matrix, topk=topk)
        degree = adj.sum(dim=-1)
        strength = weighted.sum(dim=-1)
        clustering = _clustering_coefficients(adj)
        dist = _shortest_path_distances(adj)
        finite = torch.isfinite(dist)
        off_diag = ~torch.eye(adj.size(1), dtype=torch.bool, device=adj.device).unsqueeze(0)
        valid = finite & off_diag
        inv_dist = torch.where(valid, 1.0 / dist.clamp(min=1.0), torch.zeros_like(dist))
        global_eff = inv_dist.sum(dim=(1, 2)) / (adj.size(1) * (adj.size(1) - 1.0))
        path_sum = torch.where(valid, dist, torch.zeros_like(dist)).sum(dim=(1, 2))
        path_count = valid.sum(dim=(1, 2)).clamp(min=1)
        char_path = path_sum / path_count

        values = torch.stack(
            [
                adj.sum(dim=(1, 2)) / (adj.size(1) * (adj.size(1) - 1.0)),
                clustering.mean(dim=1),
                degree.mean(dim=1),
                degree.std(dim=1),
                strength.mean(dim=1),
                strength.std(dim=1),
                global_eff,
                char_path,
                _degree_assortativity(adj, degree),
                _spectral_gap(adj),
            ],
            dim=1,
        )
        return _pack_features(values, resolution, squeeze=fc_matrix.dim() == 2)
    except Exception:
        return torch.zeros(resolution, dtype=torch.float32)


def extract_topology_feature(fc_matrix, feature_type="persistence", resolution=50, topk=20):
    if feature_type == "none":
        return torch.zeros(resolution, dtype=torch.float32)
    if feature_type == "persistence":
        return extract_persistence_landscape(fc_matrix, resolution=resolution)
    if feature_type == "avg_clustering":
        return extract_avg_clustering_feature(fc_matrix, resolution=resolution, topk=topk)
    if feature_type == "strength_stats":
        return extract_strength_stats_feature(fc_matrix, resolution=resolution, topk=topk)
    if feature_type == "efficiency_stats":
        return extract_efficiency_stats_feature(fc_matrix, resolution=resolution, topk=topk)
    if feature_type == "laplacian_spectrum":
        return extract_laplacian_spectrum_feature(fc_matrix, resolution=resolution, topk=topk)
    if feature_type == "graph_stats":
        return extract_graph_stats_feature(fc_matrix, resolution=resolution, topk=topk)
    choices = "'none', 'persistence', 'avg_clustering', 'strength_stats', 'efficiency_stats', 'laplacian_spectrum', 'graph_stats'"
    raise ValueError(f"Unsupported topology feature type: {feature_type}. Choose one of {choices}.")


def _build_binary_graph(fc_matrix, topk=20):
    fc_batch = fc_matrix.unsqueeze(0) if fc_matrix.dim() == 2 else fc_matrix
    weighted = build_topk_adjacency(fc_batch, topk=topk, use_abs=True).float()
    weighted = weighted - torch.diag_embed(torch.diagonal(weighted, dim1=1, dim2=2))
    adj = (weighted > 0).float()
    return weighted, adj


def _clustering_coefficients(adj):
    deg = adj.sum(dim=-1)
    closed_triplets = torch.diagonal(torch.bmm(torch.bmm(adj, adj), adj), dim1=1, dim2=2)
    denom = deg * (deg - 1.0)
    return torch.where(denom > 0, closed_triplets / denom.clamp(min=1.0), torch.zeros_like(deg))


def _shortest_path_distances(adj):
    B, V, _ = adj.shape
    inf = torch.tensor(float("inf"), device=adj.device)
    dist = torch.where(adj > 0, torch.ones_like(adj), inf.expand_as(adj))
    idx = torch.arange(V, device=adj.device)
    dist[:, idx, idx] = 0.0
    for k in range(V):
        dist = torch.minimum(dist, dist[:, :, k].unsqueeze(2) + dist[:, k, :].unsqueeze(1))
    return dist


def _degree_assortativity(adj, degree):
    values = []
    for b in range(adj.size(0)):
        edges = torch.triu(adj[b], diagonal=1) > 0
        src, dst = edges.nonzero(as_tuple=True)
        if src.numel() < 2:
            values.append(torch.tensor(0.0, device=adj.device))
            continue
        x = degree[b, src]
        y = degree[b, dst]
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp(min=1e-8))
        values.append((x * y).sum() / denom)
    return torch.stack(values)


def _spectral_gap(adj):
    degree = adj.sum(dim=-1)
    lap = torch.diag_embed(degree) - adj
    eigvals = torch.linalg.eigvalsh(lap).clamp(min=0.0)
    if eigvals.size(1) < 2:
        return torch.zeros(adj.size(0), device=adj.device)
    return eigvals[:, 1]


def _safe_entropy(values):
    probs = values / values.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1)


def _pack_features(values, resolution, squeeze=False):
    if values.dim() == 1:
        values = values.unsqueeze(1)
    feature = torch.zeros((values.size(0), resolution), dtype=torch.float32, device=values.device)
    length = min(resolution, values.size(1))
    feature[:, :length] = values[:, :length].float()
    feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    return feature.squeeze(0).cpu() if squeeze else feature.cpu()
