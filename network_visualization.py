import datetime
import json
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from brain_diffusion.graph_utils import flatten_upper


def create_network_run_dir(root_dir, seed):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(root_dir, f"run_{timestamp}_seed{seed}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


@torch.no_grad()
def select_best_networks(X_tr, y_tr, FC0, FC1, source):
    """Select one candidate per class using training-class centroid similarity."""
    real_flat = flatten_upper(X_tr)
    records = {}
    for class_id, candidates in ((0, FC0), (1, FC1)):
        if candidates is None or len(candidates) == 0:
            continue
        class_real = real_flat[y_tr == class_id]
        if len(class_real) == 0:
            continue

        candidate_flat = flatten_upper(candidates)
        class_centroid = class_real.mean(dim=0, keepdim=True)
        similarities = F.cosine_similarity(candidate_flat, class_centroid, dim=1)
        best_index = int(torch.argmax(similarities).item())
        best_flat = candidate_flat[best_index]
        mse = F.mse_loss(best_flat, class_centroid.squeeze(0)).item()
        records[class_id] = {
            "class_id": class_id,
            "candidate_index": best_index,
            "cosine_similarity": float(similarities[best_index].item()),
            "mse_to_real_centroid": float(mse),
            "source": source,
            "matrix": candidates[best_index].detach().cpu().float().clone(),
        }
    return records


def update_global_best(global_best, fold_records, fold):
    for class_id, record in fold_records.items():
        candidate = dict(record)
        candidate["fold"] = int(fold)
        current = global_best.get(class_id)
        if current is None or candidate["cosine_similarity"] > current["cosine_similarity"]:
            global_best[class_id] = candidate


def save_network_records(records, output_dir, scope, top_edges):
    saved = []
    scope_dir = os.path.join(output_dir, scope)
    for class_id, record in sorted(records.items()):
        class_dir = os.path.join(scope_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        matrix = record["matrix"].detach().cpu().numpy().astype(np.float32)

        csv_path = os.path.join(class_dir, "best_fc.csv")
        npy_path = os.path.join(class_dir, "best_fc.npy")
        edge_path = os.path.join(class_dir, "best_fc.edge")
        sparse_edge_path = os.path.join(class_dir, f"best_fc_top{int(top_edges)}.edge")
        png_path = os.path.join(class_dir, "best_fc_visualization.png")
        metadata_path = os.path.join(class_dir, "metadata.json")

        np.savetxt(csv_path, matrix, delimiter=",", fmt="%.8f")
        np.save(npy_path, matrix)
        np.savetxt(edge_path, matrix, delimiter="\t", fmt="%.8f")
        sparse_matrix = _keep_strongest_edges(matrix, top_edges)
        np.savetxt(sparse_edge_path, sparse_matrix, delimiter="\t", fmt="%.8f")
        _plot_network(matrix, record, png_path, top_edges)

        metadata = {
            key: value
            for key, value in record.items()
            if key != "matrix"
        }
        metadata.update(
            {
                "shape": list(matrix.shape),
                "selection_criterion": "maximum cosine similarity to the real training-class centroid",
                "visualized_top_edges": int(top_edges),
                "csv_path": csv_path,
                "npy_path": npy_path,
                "brainnet_edge_path": edge_path,
                "brainnet_sparse_edge_path": sparse_edge_path,
                "visualization_path": png_path,
            }
        )
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        saved.append(metadata)
    return saved


def save_run_summary(output_dir, fold_metadata, global_metadata, cfg):
    summary = {
        "selection_criterion": "maximum cosine similarity to the real training-class centroid",
        "module_switches": {
            "autoencoder": bool(cfg.use_autoencoder),
            "diffusion": bool(cfg.use_diffusion),
            "gating": bool(cfg.use_gating),
        },
        "seed": int(cfg.seed),
        "fold_best": fold_metadata,
        "global_best": global_metadata,
    }
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_brainnet_instructions(output_dir, cfg)
    return path


def _keep_strongest_edges(matrix, top_edges):
    matrix = np.asarray(matrix, dtype=np.float32)
    V = matrix.shape[0]
    upper_i, upper_j = np.triu_indices(V, k=1)
    weights = matrix[upper_i, upper_j]
    nonzero = np.flatnonzero(np.abs(weights) > 0)
    sparse = np.zeros_like(matrix)
    if len(nonzero):
        edge_count = min(max(1, int(top_edges)), len(nonzero))
        selected = nonzero[np.argsort(np.abs(weights[nonzero]))[-edge_count:]]
        sparse[upper_i[selected], upper_j[selected]] = weights[selected]
        sparse[upper_j[selected], upper_i[selected]] = weights[selected]
    return sparse


def _write_brainnet_instructions(output_dir, cfg):
    path = os.path.join(output_dir, "BRAINNET_VIEWER_README.txt")
    text = f"""BrainNet Viewer files

1. best_fc.edge is the complete weighted FC matrix in ASCII tab-separated format.
2. best_fc_top{int(cfg.network_vis_top_edges)}.edge contains only the strongest
   {int(cfg.network_vis_top_edges)} undirected connections by absolute weight.
3. Load an atlas .node file whose node count and ROI order exactly match the
   {int(cfg.V)} x {int(cfg.V)} FC matrix. Do not use an AAL90 node file for AAL116 data.
4. In BrainNet Viewer load:
   Surface: BrainMesh_ICBM152.nv
   Node:    your matching atlas .node file
   Edge:    best_fc.edge or best_fc_top{int(cfg.network_vis_top_edges)}.edge

The CSV file stores the same matrix but is not the recommended direct input
because BrainNet Viewer identifies and parses edge files by the .edge format.
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _plot_network(matrix, record, output_path, top_edges):
    cache_dir = os.path.join(tempfile.gettempdir(), "brain_diffusion_matplotlib")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    V = matrix.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    max_abs = max(float(np.max(np.abs(matrix))), 1e-6)
    heatmap = axes[0].imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    axes[0].set_title(f"Class {record['class_id']} best generated FC")
    axes[0].set_xlabel("ROI")
    axes[0].set_ylabel("ROI")
    fig.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04, label="Connection weight")

    upper_i, upper_j = np.triu_indices(V, k=1)
    weights = matrix[upper_i, upper_j]
    nonzero = np.flatnonzero(np.abs(weights) > 0)
    if len(nonzero):
        edge_count = min(max(1, int(top_edges)), len(nonzero))
        selected = nonzero[np.argsort(np.abs(weights[nonzero]))[-edge_count:]]
    else:
        selected = np.array([], dtype=np.int64)

    graph = nx.Graph()
    graph.add_nodes_from(range(V))
    for index in selected:
        graph.add_edge(int(upper_i[index]), int(upper_j[index]), weight=float(weights[index]))

    positions = nx.circular_layout(graph)
    degrees = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float32)
    node_sizes = 18.0 + 5.0 * degrees
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        node_color=degrees,
        cmap=plt.cm.viridis,
        linewidths=0.2,
        edgecolors="black",
        ax=axes[1],
    )
    if graph.number_of_edges():
        edge_list = list(graph.edges())
        edge_weights = np.array([graph[u][v]["weight"] for u, v in edge_list])
        edge_widths = 0.3 + 2.2 * np.abs(edge_weights) / max_abs
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edge_list,
            width=edge_widths,
            edge_color=edge_weights,
            edge_cmap=plt.cm.coolwarm,
            edge_vmin=-max_abs,
            edge_vmax=max_abs,
            alpha=0.75,
            ax=axes[1],
        )
    axes[1].set_title(f"Top {len(selected)} strongest edges (circular ROI layout)")
    axes[1].set_axis_off()

    score = record["cosine_similarity"]
    mse = record["mse_to_real_centroid"]
    fig.suptitle(f"Similarity={score:.4f} | MSE={mse:.6f} | source={record['source']}")
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
