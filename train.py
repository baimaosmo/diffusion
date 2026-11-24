import datetime
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from brain_diffusion.graph_utils import flatten_upper, generate_ba_mask, repair_and_filter
from brain_diffusion.fc_export import FCExporter
from brain_diffusion.models import (
    ClusterGatedMLP,
    ConditionalDiffusionGCN,
    ConditionalDiffusionTransformer,
    GraphRAE,
    LatentDDPM,
    LatentNormalizer,
    PlainMLP,
)
from brain_diffusion.models.rae import fc_recon_loss
from brain_diffusion.network_visualization import (
    create_network_run_dir,
    save_network_records,
    save_run_summary,
    select_best_networks,
    update_global_best,
)
from brain_diffusion.topology import extract_topology_feature
from brain_diffusion.sample_diagnostics import SampleDiagnostics


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(dataset, cfg):
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    input_fc = torch.stack([dataset[i][0] for i in range(len(dataset))])
    all_fc = apply_fc_binarization(input_fc, cfg)
    all_y = np.array([int(dataset[i][1].item()) for i in range(len(dataset))])
    skf = StratifiedKFold(n_splits=cfg.k, shuffle=True, random_state=cfg.seed)
    D = cfg.V * (cfg.V - 1) // 2
    diagnostics = SampleDiagnostics(dataset, cfg) if cfg.save_sample_diagnostics else None
    fc_exporter = None
    if cfg.save_all_fc:
        fc_exporter = FCExporter(cfg, input_fc, all_fc, all_y, getattr(dataset, "ids", None))
        print(f">>> [FC Export] All FCs and split metadata: {fc_exporter.run_dir}")

    topology_conditioning = cfg.topo_feature_type != "none"
    print(
        ">>> [Module Switches] "
        f"Autoencoder={'ON' if cfg.use_autoencoder else 'OFF'}, "
        f"Diffusion={'ON' if cfg.use_diffusion else 'OFF'}, "
        f"Gating={'ON' if cfg.use_gating else 'OFF'}, "
        f"TopologyConditioning={'ON' if topology_conditioning else 'OFF'}"
    )

    network_run_dir = None
    global_best_networks = {}
    fold_network_metadata = []
    if cfg.save_best_networks:
        if cfg.network_vis_top_edges < 1:
            raise ValueError("network_vis_top_edges must be at least 1")
        network_run_dir = create_network_run_dir(cfg.best_network_dir, cfg.seed)
        print(f">>> [Network Output] Saving best generated networks to: {network_run_dir}")

    acc_scores, auc_scores, f1_scores = [], [], []
    prec_scores, rec_scores, spec_scores = [], [], []
    gen_mse_scores = []

    if topology_conditioning:
        print(f">>> [Pre-Extraction] {cfg.topo_feature_type} features for all subjects...")
        mu_G_all = [
            extract_topology_feature(
                fc,
                feature_type=cfg.topo_feature_type,
                resolution=cfg.topo_resolution,
                topk=cfg.topk,
            )
            for fc in all_fc
        ]
        mu_G_all = torch.stack(mu_G_all).to(device)
    else:
        print(">>> [Pre-Extraction] OFF: no persistent-homology or topology features are computed")
        # Keep the existing call signatures and batching path; models ignore
        # this placeholder because their topology branches are absent.
        mu_G_all = torch.zeros((len(all_fc), cfg.topo_resolution), dtype=all_fc.dtype, device=device)

    for fold, (tr, va) in enumerate(skf.split(all_fc, all_y)):
        print(f"\n========== Fold {fold + 1}/{cfg.k} ==========")
        X_tr = all_fc[tr].to(device)
        y_tr = torch.tensor(all_y[tr], dtype=torch.long, device=device)
        X_va = all_fc[va].to(device)
        y_va = torch.tensor(all_y[va], dtype=torch.long, device=device)
        mu_G_tr = mu_G_all[tr]
        export_callback = None
        if fc_exporter is not None:
            fc_exporter.begin_fold(fold + 1, tr, va)
            export_callback = partial(fc_exporter.save_generated, fold=fold + 1)

        rae = None
        Z_tr_raw = X_tr
        if cfg.use_autoencoder:
            rae = train_rae(X_tr, mu_G_tr, cfg, device)
            with torch.no_grad():
                _, Z_tr_raw = rae(X_tr, mu_G_tr)
        else:
            print(">>> [Step 1] Autoencoder OFF: using FC matrices as diffusion states")

        if cfg.use_diffusion:
            norm = LatentNormalizer()
            norm.fit(Z_tr_raw)
            Z_tr_n = norm.transform(Z_tr_raw)
            latent_dim = Z_tr_n.size(-1)
            _, ddpm = train_diffusion(Z_tr_n, y_tr, mu_G_tr, cfg, device, latent_dim)
            X_tr_flat, X_va_flat, X0_keep, X1_keep, fold_gen_mse, fold_best_networks = generate_augmented_features(
                X_tr, y_tr, X_va, mu_G_tr, rae, norm, ddpm, cfg, device, latent_dim,
                export_callback=export_callback,
            )
        elif rae is not None:
            print(">>> [Step 2] Diffusion OFF: using Graph-RAE reconstructions for augmentation")
            X_tr_flat, X_va_flat, X0_keep, X1_keep, fold_gen_mse, fold_best_networks = generate_reconstructed_features(
                X_tr, y_tr, X_va, mu_G_tr, rae, export_callback=export_callback
            )
        else:
            print(">>> [Steps 1-3] Autoencoder and diffusion OFF: real FC samples only")
            X_tr_flat, X_va_flat, X0_keep, X1_keep, fold_gen_mse, fold_best_networks = prepare_real_only_features(
                X_tr, X_va
            )

        if fold_gen_mse is not None:
            gen_mse_scores.append(fold_gen_mse)
        if network_run_dir and fold_best_networks:
            for record in fold_best_networks.values():
                record["fold"] = fold + 1
            fold_network_metadata.extend(
                save_network_records(
                    fold_best_networks,
                    network_run_dir,
                    scope=f"fold_{fold + 1:02d}",
                    top_edges=cfg.network_vis_top_edges,
                )
            )
            update_global_best(global_best_networks, fold_best_networks, fold + 1)

        classifier_result = train_classifier(
            X_tr_flat, X_va_flat, X0_keep, X1_keep, y_tr, y_va, D, cfg, device,
            return_diagnostics=diagnostics is not None,
        )
        if diagnostics is not None:
            metrics, details = classifier_result
            diagnostics.record_fold(
                fold + 1, tr, va, X_tr_flat.detach().cpu().numpy(), X_va_flat.detach().cpu().numpy(),
                all_y[tr], all_y[va], details,
            )
        else:
            metrics = classifier_result
        acc, auc, f1, prec, rec, spec = metrics
        print(
            f"Fold {fold + 1} Result: Acc={acc:.3f}, AUC={auc:.3f}, "
            f"F1={f1:.3f}, Rec(Sens)={rec:.3f}, Spec={spec:.3f}"
        )

        acc_scores.append(min(acc, 1.0))
        auc_scores.append(auc)
        f1_scores.append(f1)
        prec_scores.append(prec)
        rec_scores.append(rec)
        spec_scores.append(spec)

    if network_run_dir:
        global_metadata = save_network_records(
            global_best_networks,
            network_run_dir,
            scope="global_best",
            top_edges=cfg.network_vis_top_edges,
        )
        summary_path = save_run_summary(
            network_run_dir,
            fold_network_metadata,
            global_metadata,
            cfg,
        )
        if global_best_networks:
            print(f">>> [Network Output] Global best networks and summary saved: {summary_path}")
        else:
            print(">>> [Network Output] No generated/reconstructed networks were available to save.")

    if fc_exporter is not None:
        fc_exporter.finish()
        print(f">>> [FC Export] Plot manifests: {fc_exporter.run_dir}/manifest_class{{0,1}}.json")
    if diagnostics is not None:
        diagnostics.finish()
    print_final_results(acc_scores, auc_scores, f1_scores, rec_scores, spec_scores, gen_mse_scores)
    write_log(cfg, acc_scores, auc_scores, f1_scores, rec_scores, spec_scores, gen_mse_scores)
    return summarize_results(acc_scores, auc_scores, f1_scores, prec_scores, rec_scores, spec_scores, gen_mse_scores)


def apply_fc_binarization(all_fc, cfg):
    """Optionally convert continuous FC values to binary edges for one run."""
    if not cfg.fc_binarize:
        print(">>> [FC Input] Continuous weighted FC (binarization disabled)")
        return all_fc

    threshold = float(cfg.fc_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"fc_threshold must be in (0, 1], got {threshold}")

    if cfg.fc_threshold_mode == "positive":
        binary = (all_fc >= threshold).to(all_fc.dtype)
        rule = f"r >= {threshold:g}"
    elif cfg.fc_threshold_mode == "absolute":
        binary = (all_fc.abs() >= threshold).to(all_fc.dtype)
        rule = f"|r| >= {threshold:g}"
    else:
        raise ValueError(
            f"Unsupported fc_threshold_mode={cfg.fc_threshold_mode!r}; "
            "choose 'positive' or 'absolute'."
        )

    binary = torch.maximum(binary, binary.transpose(1, 2))
    binary = binary - torch.diag_embed(torch.diagonal(binary, dim1=1, dim2=2))
    V = binary.size(1)
    upper = torch.triu_indices(V, V, offset=1, device=binary.device)
    density = binary[:, upper[0], upper[1]].mean().item()
    print(
        f">>> [FC Input] Binary FC enabled: {rule}; "
        f"retained undirected-edge density={density:.2%}"
    )
    return binary


def train_rae(X_tr, mu_G_tr, cfg, device):
    print(">>> [Step 1] Training Graph-RAE")
    rae = GraphRAE(
        V=cfg.V,
        K=cfg.K,
        hidden=cfg.enc_hidden,
        layers=cfg.enc_layers,
        topk=cfg.topk,
        dropout=cfg.enc_dropout,
        topo_dim=cfg.topo_resolution,
        use_film=cfg.use_film,
        use_topology_conditioning=cfg.topo_feature_type != "none",
    ).to(device)
    opt_ae = torch.optim.Adam(rae.parameters(), lr=1e-3)
    ae_loader = DataLoader(TensorDataset(X_tr, mu_G_tr), batch_size=cfg.ae_bs, shuffle=True)

    rae.train()
    for ep in range(cfg.ae_epochs):
        total_loss = 0.0
        ba_mask = generate_ba_mask(cfg.V, m_attach=cfg.ba_m).to(device)
        non_ba_mask = 1.0 - (ba_mask > 0).float()

        for fc, mu_g in ae_loader:
            recon, z = rae(fc, mu_g)

            r_loss = fc_recon_loss(recon, fc)
            reg_loss = torch.mean(z**2)
            topo_loss = torch.mean((recon * non_ba_mask) ** 2)

            if cfg.topo_feature_type == "none":
                con_loss = r_loss.new_zeros(())
            else:
                recon_norm = F.normalize(recon.view(recon.size(0), -1), p=2, dim=1)
                mu_g_norm = F.normalize(mu_g, p=2, dim=1)
                con_loss = F.mse_loss(
                    torch.matmul(recon_norm, recon_norm.T),
                    torch.matmul(mu_g_norm, mu_g_norm.T),
                )

            loss = r_loss + cfg.lambda_reg * reg_loss + cfg.lambda_topo * topo_loss + cfg.lambda_con * con_loss
            opt_ae.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rae.parameters(), 1.0)
            opt_ae.step()
            total_loss += loss.item()

        if (ep + 1) % 50 == 0:
            print(f"RAE Ep {ep + 1:03d} | Loss={total_loss / len(ae_loader):.5f}")

    rae.eval()
    return rae


def train_diffusion(Z_tr_n, y_tr, mu_G_tr, cfg, device, latent_dim):
    space_name = "latent" if cfg.use_autoencoder else "FC"
    guidance = "topology-guided" if cfg.topo_feature_type != "none" else "class-conditioned w/o-PH"
    print(f">>> [Step 2] Training {guidance} {space_name} diffusion ({cfg.diff_backbone})...")
    model_cls = get_diffusion_model_cls(cfg.diff_backbone)
    cdt = model_cls(
        V=cfg.V,
        K=latent_dim,
        hidden=256,
        depth=6,
        num_classes=2,
        dropout=0.1,
        topo_dim=cfg.topo_resolution,
        use_film=cfg.use_film,
        use_topology_conditioning=cfg.topo_feature_type != "none",
    ).to(device)
    ddpm = LatentDDPM(cdt, T=cfg.T, device=device)
    opt_diff = torch.optim.AdamW(cdt.parameters(), lr=2e-4, weight_decay=1e-4)
    diff_loader = DataLoader(TensorDataset(Z_tr_n, y_tr, mu_G_tr), batch_size=cfg.diff_bs, shuffle=True)

    cdt.train()
    for ep in range(cfg.diff_epochs):
        last_loss = None
        for z, y, mu_g in diff_loader:
            t = torch.randint(0, ddpm.T, (z.size(0),), device=device).long()
            zt, noise = ddpm.q_sample(z, t)
            loss = F.mse_loss(cdt(zt, t, y, mu_g), noise)
            opt_diff.zero_grad()
            loss.backward()
            opt_diff.step()
            last_loss = loss.item()

        if (ep + 1) % 100 == 0:
            print(f"    Diff Ep {ep + 1}/{cfg.diff_epochs} | Loss={last_loss:.6f}")

    cdt.eval()
    return cdt, ddpm


def get_diffusion_model_cls(backbone):
    if backbone == "gcn":
        return ConditionalDiffusionGCN
    if backbone == "transformer":
        return ConditionalDiffusionTransformer
    raise ValueError(f"Unsupported diffusion backbone: {backbone}. Choose 'gcn' or 'transformer'.")


@torch.no_grad()
def generate_augmented_features(X_tr, y_tr, X_va, mu_G_tr, rae, norm, ddpm, cfg, device, latent_dim,
                                export_callback=None):
    generation_mode = "Dual-topology guided" if cfg.topo_feature_type != "none" else "Class-conditioned w/o-PH"
    print(f">>> [Step 3] {generation_mode} generation...")
    y0 = torch.zeros(cfg.gen_per_class, dtype=torch.long, device=device)
    y1 = torch.ones(cfg.gen_per_class, dtype=torch.long, device=device)

    mu_g_mean_0 = mu_G_tr[y_tr == 0].mean(dim=0).unsqueeze(0).repeat(cfg.gen_per_class, 1)
    mu_g_mean_1 = mu_G_tr[y_tr == 1].mean(dim=0).unsqueeze(0).repeat(cfg.gen_per_class, 1)

    Z0 = norm.inverse(ddpm.sample(cfg.gen_per_class, cfg.V, latent_dim, y0, mu_g_mean_0))
    Z1 = norm.inverse(ddpm.sample(cfg.gen_per_class, cfg.V, latent_dim, y1, mu_g_mean_1))

    if rae is not None:
        FC0 = rae.decode(Z0, mu_g_mean_0)
        FC1 = rae.decode(Z1, mu_g_mean_1)
    else:
        FC0 = postprocess_direct_fc(Z0, cfg)
        FC1 = postprocess_direct_fc(Z1, cfg)

    source = "latent_diffusion" if rae is not None else "direct_fc_diffusion"
    return package_augmented_features(X_tr, y_tr, X_va, FC0, FC1, source, export_callback=export_callback)


@torch.no_grad()
def generate_reconstructed_features(X_tr, y_tr, X_va, mu_G_tr, rae, export_callback=None):
    reconstructed, _ = rae(X_tr, mu_G_tr)
    FC0 = reconstructed[y_tr == 0]
    FC1 = reconstructed[y_tr == 1]
    return package_augmented_features(X_tr, y_tr, X_va, FC0, FC1, "autoencoder_reconstruction",
                                      export_callback=export_callback)


def prepare_real_only_features(X_tr, X_va):
    X_tr_flat = flatten_upper(X_tr)
    X_va_flat = flatten_upper(X_va)
    empty = X_tr_flat.new_empty((0, X_tr_flat.size(1)))
    return X_tr_flat, X_va_flat, empty, empty.clone(), None, {}


def postprocess_direct_fc(fc, cfg):
    lower = 0.0 if cfg.fc_binarize else -1.0
    fc = torch.clamp(fc, lower, 1.0)
    fc = (fc + fc.transpose(1, 2)) / 2.0
    return fc - torch.diag_embed(torch.diagonal(fc, dim1=1, dim2=2))


def package_augmented_features(X_tr, y_tr, X_va, FC0, FC1, source, export_callback=None):

    # Preserve the exact model outputs before ranking, filtering or repair.
    if export_callback is not None:
        export_callback(FC0, FC1, source)

    X_tr_flat = flatten_upper(X_tr)
    X_va_flat = flatten_upper(X_va)
    X0_flat = flatten_upper(FC0)
    X1_flat = flatten_upper(FC1)

    gen_mse_0 = F.mse_loss(X0_flat.mean(dim=0), X_tr_flat[y_tr == 0].mean(dim=0)).item()
    gen_mse_1 = F.mse_loss(X1_flat.mean(dim=0), X_tr_flat[y_tr == 1].mean(dim=0)).item()
    fold_gen_mse = (gen_mse_0 + gen_mse_1) / 2.0
    print(f"    [Generation Quality] Gen-MSE: {fold_gen_mse:.5f}")

    X0_keep = repair_and_filter(X_tr_flat[y_tr == 0], X0_flat, top_k_ratio=1.0)
    X1_keep = repair_and_filter(X_tr_flat[y_tr == 1], X1_flat, top_k_ratio=1.0)
    best_networks = select_best_networks(X_tr, y_tr, FC0, FC1, source)
    return X_tr_flat, X_va_flat, X0_keep, X1_keep, fold_gen_mse, best_networks


def train_classifier(X_tr_flat, X_va_flat, X0_keep, X1_keep, y_tr, y_va, D, cfg, device,
                     return_diagnostics=False):
    X_train_final = torch.cat([X_tr_flat, X0_keep, X1_keep], dim=0)
    y_train_final = torch.cat(
        [
            y_tr,
            torch.zeros(len(X0_keep), device=device),
            torch.ones(len(X1_keep), device=device),
        ],
        dim=0,
    ).long()

    val_probs = None
    if cfg.use_gating:
        print(">>> [Step 4] Clustering & training gated MLP...")
        num_clusters = min(cfg.num_clusters, len(X_tr_flat))
        kmeans = KMeans(n_clusters=num_clusters, random_state=cfg.seed, n_init=10).fit(
            X_tr_flat.cpu().numpy()
        )
        train_probs = get_cluster_probs(X_train_final, kmeans)
        val_probs = get_cluster_probs(X_va_flat, kmeans)
        mlp = ClusterGatedMLP(input_dim=D, num_clusters=num_clusters).to(device)
        train_dataset = TensorDataset(X_train_final, train_probs, y_train_final)
    else:
        print(">>> [Step 4] Gating OFF: training plain MLP without K-means gate...")
        mlp = PlainMLP(input_dim=D).to(device)
        train_dataset = TensorDataset(X_train_final, y_train_final)

    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    batch_size = min(32, len(train_dataset))
    drop_last = len(train_dataset) >= 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )

    n0, n1 = (y_train_final == 0).sum().item(), (y_train_final == 1).sum().item()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, n0 / max(1, n1)], device=device))

    best_acc = best_auc = best_f1 = best_prec = best_rec = best_spec = 0.0
    tail_predictions = []
    for ep in range(80):
        mlp.train()
        for batch in train_loader:
            if cfg.use_gating:
                x, cluster_probs, y = batch
                logits = mlp(x, cluster_probs)
            else:
                x, y = batch
                logits = mlp(x)
            opt_mlp.zero_grad()
            criterion(logits, y).backward()
            opt_mlp.step()

        evaluation = evaluate_classifier(
            mlp, X_va_flat, val_probs, y_va, use_gating=cfg.use_gating,
            return_predictions=return_diagnostics,
        )
        if return_diagnostics:
            metrics, predicted_labels, prob_class1 = evaluation
            if ep >= 80 - min(80, cfg.diagnostic_tail_epochs):
                tail_predictions.append(predicted_labels.copy())
        else:
            metrics = evaluation
        acc, auc, f1, prec, rec, spec = metrics
        if acc > best_acc:
            best_acc, best_auc = acc, auc
            best_f1, best_prec, best_rec, best_spec = f1, prec, rec, spec

        if (ep + 1) % 20 == 0:
            print(f"    MLP Ep {ep + 1:02d} | Val Acc={acc:.3f} AUC={auc:.3f} F1={f1:.3f} Spec={spec:.3f}")

    result = (best_acc, best_auc, best_f1, best_prec, best_rec, best_spec)
    if return_diagnostics:
        details = {
            "final_epoch": 80,
            "final_predicted_label": predicted_labels,
            "final_prob_class1": prob_class1,
            "tail_epochs": len(tail_predictions),
            "tail_error_rate": (np.stack(tail_predictions) != y_va.cpu().numpy()).mean(axis=0),
        }
        return result, details
    return result


def get_cluster_probs(X_tensor, kmeans_model, temp=1.0):
    dists = torch.from_numpy(kmeans_model.transform(X_tensor.cpu().numpy())).to(X_tensor.device)
    return F.softmax(-dists / temp, dim=1)


@torch.no_grad()
def evaluate_classifier(mlp, X_va_flat, val_probs, y_va, use_gating=True, return_predictions=False):
    mlp.eval()
    val_logits = mlp(X_va_flat, val_probs) if use_gating else mlp(X_va_flat)
    val_probs_soft = torch.softmax(val_logits, dim=1)[:, 1]
    val_pred = val_logits.argmax(1)
    # val_pred = (val_probs_soft >= 0.6).long()

    y_va_np = y_va.cpu().numpy()
    val_pred_np = val_pred.cpu().numpy()
    val_probs_np = val_probs_soft.cpu().numpy()
    acc = (val_pred_np == y_va_np).mean()

    try:
        if len(np.unique(y_va_np)) > 1:
            auc = roc_auc_score(y_va_np, val_probs_np)
            f1 = f1_score(y_va_np, val_pred_np, zero_division=0)
            prec = precision_score(y_va_np, val_pred_np, zero_division=0)
            rec = recall_score(y_va_np, val_pred_np, zero_division=0)
            tn, fp, _, _ = confusion_matrix(y_va_np, val_pred_np).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            auc, f1, prec, rec, spec = 0.5, 0.0, 0.0, 0.0, 0.0
    except Exception:
        auc, f1, prec, rec, spec = 0.5, 0.0, 0.0, 0.0, 0.0

    metrics = (acc, auc, f1, prec, rec, spec)
    if return_predictions:
        return metrics, val_pred_np, val_probs_np
    return metrics


def print_final_results(acc_scores, auc_scores, f1_scores, rec_scores, spec_scores, gen_mse_scores):
    results = summarize_results(acc_scores, auc_scores, f1_scores, [], rec_scores, spec_scores, gen_mse_scores)
    print("\n=== FINAL RESULTS (Dual Topology Locking) ===")
    print(f"Mean Acc : {results['acc_mean']:.4f} +/- {results['acc_std']:.4f}")
    print(f"Mean AUC : {results['auc_mean']:.4f} +/- {results['auc_std']:.4f}")
    print(f"Mean F1  : {results['f1_mean']:.4f} +/- {results['f1_std']:.4f}")
    print(f"Mean Sens: {results['sens_mean']:.4f} (Recall/Sensitivity)")
    print(f"Mean Spec: {results['spec_mean']:.4f} (Specificity)")
    if results["gen_mse_mean"] is None:
        print("Gen-MSE  : N/A (no generated or reconstructed samples)")
    else:
        print(f"Gen-MSE  : {results['gen_mse_mean']:.6f} (Lower is better generator)")


def summarize_results(acc_scores, auc_scores, f1_scores, prec_scores, rec_scores, spec_scores, gen_mse_scores):
    def mean(values):
        return float(np.mean(values)) if values else 0.0

    def std(values):
        return float(np.std(values)) if values else 0.0

    return {
        "acc_mean": mean(acc_scores),
        "acc_std": std(acc_scores),
        "auc_mean": mean(auc_scores),
        "auc_std": std(auc_scores),
        "f1_mean": mean(f1_scores),
        "f1_std": std(f1_scores),
        "prec_mean": mean(prec_scores),
        "prec_std": std(prec_scores),
        "sens_mean": mean(rec_scores),
        "sens_std": std(rec_scores),
        "spec_mean": mean(spec_scores),
        "spec_std": std(spec_scores),
        "gen_mse_mean": mean(gen_mse_scores) if gen_mse_scores else None,
        "gen_mse_std": std(gen_mse_scores) if gen_mse_scores else None,
    }


def write_log(cfg, acc_scores, auc_scores, f1_scores, rec_scores, spec_scores, gen_mse_scores):
    if not cfg.log_path:
        return

    log_dir = os.path.dirname(cfg.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gen_mse_text = f"{np.mean(gen_mse_scores):.5f}" if gen_mse_scores else "N/A"
    log_line = (
        f"[{now_time}] "
        f"Set: [TopoCon={cfg.lambda_con}, TopoRes={cfg.topo_resolution}, BA={cfg.ba_m}, "
        f"TopoFeature={cfg.topo_feature_type}, "
        f"TopoConditioning={'off' if cfg.topo_feature_type == 'none' else 'on'}, "
        f"TopoConEffective={0.0 if cfg.topo_feature_type == 'none' else cfg.lambda_con}, "
        f"FiLM={'on' if cfg.use_film else 'off'}, "
        f"DiffBackbone={cfg.diff_backbone}, "
        f"FCBinary={'on' if cfg.fc_binarize else 'off'}, "
        f"FCThreshold={cfg.fc_threshold}, FCThresholdMode={cfg.fc_threshold_mode}, "
        f"Autoencoder={'on' if cfg.use_autoencoder else 'off'}, "
        f"Diffusion={'on' if cfg.use_diffusion else 'off'}, "
        f"Gating={'on' if cfg.use_gating else 'off'}] | "
        f"Results: Acc={np.mean(acc_scores):.3f}, AUC={np.mean(auc_scores):.3f}, "
        f"F1={np.mean(f1_scores):.3f}, Sens={np.mean(rec_scores):.3f}, Spec={np.mean(spec_scores):.3f}, "
        f"Gen-MSE={gen_mse_text}\n"
    )
    with open(cfg.log_path, "a", encoding="utf-8") as f:
        f.write(log_line)
