from dataclasses import dataclass


@dataclass
class PipelineConfig:
    fc_dir: str = "/root/autodl-fs/data/Directly_use/norm_bold_matrix"
    label_csv: str | None = None
    fc_pattern: str | None = None
    log_path: str = "runs/kki_results.log"
    data_format: str = "adni_bold"  # "fc", "adhd200_aal", "adni_bold", or legacy "andi_bold"
    bold_pattern: str = "norm_bold_matrix_sample_{index}.csv"
    bold_orientation: str = "roi_by_time"  # "auto", "time_by_roi", or "roi_by_time"
    fc_cache_dir: str | None = None
    # Original ADNI sample IDs, not zero-based dataset indices. () restores all samples.
    adni_exclude_sample_ids: tuple[int, ...] = (4,)
    fc_binarize: bool = False
    fc_threshold: float = 0.5
    fc_threshold_mode: str = "positive"  # "positive" or "absolute"
    adhd_site: str = "NYU"
    adhd_tc_prefix: str = "sfnwmrda"
    adhd_rest: str = "first"

    k: int = 5
    V: int = 116
    K: int = 64

    ae_epochs: int = 400
    ae_bs: int = 32
    diff_epochs: int = 400
    diff_bs: int = 32
    T: int = 1000
    gen_per_class: int = 100

    ba_m: int = 50
    lambda_topo: float = 0.2
    lambda_reg: float = 1e-5
    lambda_con: float = 0.2

    num_clusters: int = 10
    topk: int = 70
    enc_hidden: int = 128
    enc_layers: int = 3
    enc_dropout: float = 0.2
    topo_resolution: int = 10
    # "none" is the clean w/o-PH ablation: no topology feature extraction,
    # conditioning branch, or topology-feature similarity loss.
    topo_feature_type: str = "persistence"  # "persistent_homology", "graphlet", or "none"
    use_film: bool = True
    diff_backbone: str = "gcn"  # "gcn" or "transformer"
    use_autoencoder: bool = True
    use_diffusion: bool = True
    use_gating: bool = True
    save_best_networks: bool = True
    best_network_dir: str = "runs/best_generated_networks"
    network_vis_top_edges: int = 10
    save_all_fc: bool = True
    fc_export_dir: str = "runs/fc_exports"
    experiment_name: str = "Model"

    save_sample_diagnostics: bool = True
    diagnostic_dir: str = "/root/runs/sample_diagnostics"
    diagnostic_mad_threshold: float = 3.5
    diagnostic_tail_epochs: int = 10

    seed: int = 45
