import argparse
import os

from brain_diffusion.config import PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Topology-guided brain FC diffusion pipeline")
    parser.add_argument("--fc-dir", default=PipelineConfig.fc_dir)
    parser.add_argument("--label-csv", default=PipelineConfig.label_csv)
    parser.add_argument("--fc-pattern", default=PipelineConfig.fc_pattern)
    parser.add_argument("--log-path", default=PipelineConfig.log_path)
    parser.add_argument(
        "--data-format",
        choices=["fc", "adhd200_aal", "adni_bold", "andi_bold"],
        default=PipelineConfig.data_format,
    )
    parser.add_argument("--bold-pattern", default=PipelineConfig.bold_pattern)
    parser.add_argument(
        "--bold-orientation",
        choices=["auto", "time_by_roi", "roi_by_time"],
        default=PipelineConfig.bold_orientation,
    )
    parser.add_argument("--fc-cache-dir", default=PipelineConfig.fc_cache_dir)
    parser.add_argument(
        "--adni-exclude-sample-ids", nargs="*", type=int,
        default=PipelineConfig.adni_exclude_sample_ids,
        help="ADNI only: skip original sample IDs (default: 4). Pass with no IDs to keep all samples. "
             "--audit-only still checks every original file.",
    )
    diagnostic_group = parser.add_mutually_exclusive_group()
    diagnostic_group.add_argument("--sample-diagnostics", dest="save_sample_diagnostics", action="store_true",
                                  help="Save raw quality checks and fixed-final-epoch held-out sample predictions.")
    diagnostic_group.add_argument("--disable-sample-diagnostics", dest="save_sample_diagnostics", action="store_false")
    parser.set_defaults(save_sample_diagnostics=PipelineConfig.save_sample_diagnostics)
    parser.add_argument("--diagnostic-dir", default=PipelineConfig.diagnostic_dir)
    parser.add_argument("--diagnostic-mad-threshold", type=float, default=PipelineConfig.diagnostic_mad_threshold)
    parser.add_argument("--diagnostic-tail-epochs", type=int, default=PipelineConfig.diagnostic_tail_epochs)
    parser.add_argument("--audit-only", action="store_true",
                        help="Scan every raw BOLD/FC file and save QC reports, including unreadable files; skip training.")
    binarize_group = parser.add_mutually_exclusive_group()
    binarize_group.add_argument(
        "--fc-binarize",
        dest="fc_binarize",
        action="store_true",
        help="Binarize FC matrices at training time.",
    )
    binarize_group.add_argument(
        "--disable-fc-binarize",
        dest="fc_binarize",
        action="store_false",
        help="Keep continuous Pearson FC values (default).",
    )
    parser.set_defaults(fc_binarize=PipelineConfig.fc_binarize)
    parser.add_argument("--fc-threshold", type=float, default=PipelineConfig.fc_threshold)
    parser.add_argument(
        "--fc-threshold-mode",
        choices=["positive", "absolute"],
        default=PipelineConfig.fc_threshold_mode,
        help="positive: r >= threshold; absolute: |r| >= threshold.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Load/convert the dataset and optionally save FC files, then exit before training.",
    )
    parser.add_argument("--adhd-site", default=PipelineConfig.adhd_site)
    parser.add_argument("--adhd-tc-prefix", default=PipelineConfig.adhd_tc_prefix)
    parser.add_argument("--adhd-rest", default=PipelineConfig.adhd_rest)
    parser.add_argument("--k", type=int, default=PipelineConfig.k)
    parser.add_argument("--V", type=int, default=PipelineConfig.V)
    parser.add_argument("--K", type=int, default=PipelineConfig.K)
    parser.add_argument("--ae-epochs", type=int, default=PipelineConfig.ae_epochs)
    parser.add_argument("--ae-bs", type=int, default=PipelineConfig.ae_bs)
    parser.add_argument("--diff-epochs", type=int, default=PipelineConfig.diff_epochs)
    parser.add_argument("--diff-bs", type=int, default=PipelineConfig.diff_bs)
    parser.add_argument("--T", type=int, default=PipelineConfig.T)
    parser.add_argument("--topo-resolution", type=int, default=PipelineConfig.topo_resolution)
    parser.add_argument("--gen-per-class", type=int, default=PipelineConfig.gen_per_class)
    parser.add_argument("--ba-m", type=int, default=PipelineConfig.ba_m)
    parser.add_argument("--lambda-topo", type=float, default=PipelineConfig.lambda_topo)
    parser.add_argument("--lambda-reg", type=float, default=PipelineConfig.lambda_reg)
    parser.add_argument("--lambda-con", type=float, default=PipelineConfig.lambda_con)
    parser.add_argument("--num-clusters", type=int, default=PipelineConfig.num_clusters)
    parser.add_argument("--topk", type=int, default=PipelineConfig.topk)
    parser.add_argument("--enc-hidden", type=int, default=PipelineConfig.enc_hidden)
    parser.add_argument("--enc-layers", type=int, default=PipelineConfig.enc_layers)
    parser.add_argument("--enc-dropout", type=float, default=PipelineConfig.enc_dropout)
    topology_feature_group = parser.add_mutually_exclusive_group()
    topology_feature_group.add_argument(
        "--topo-feature-type",
        choices=["none", "persistence", "avg_clustering", "strength_stats", "efficiency_stats", "laplacian_spectrum", "graph_stats"],
        default=PipelineConfig.topo_feature_type,
    )
    topology_feature_group.add_argument(
        "--disable-persistence", "--disable-topology-conditioning",
        dest="topo_feature_type", action="store_const", const="none",
        help="Clean w/o-PH ablation: skip topology features, their conditioning branches, and feature-similarity loss.",
    )
    parser.add_argument("--diff-backbone", choices=["gcn", "transformer"], default=PipelineConfig.diff_backbone)
    film_group = parser.add_mutually_exclusive_group()
    film_group.add_argument("--use-film", dest="use_film", action="store_true", help="Enable FiLM modulation.")
    film_group.add_argument("--disable-film", dest="use_film", action="store_false", help="Disable FiLM modulation for ablation.")
    parser.set_defaults(use_film=PipelineConfig.use_film)

    autoencoder_group = parser.add_mutually_exclusive_group()
    autoencoder_group.add_argument("--use-autoencoder", dest="use_autoencoder", action="store_true")
    autoencoder_group.add_argument(
        "--disable-autoencoder",
        dest="use_autoencoder",
        action="store_false",
        help="Ablation: diffuse directly in FC space without Graph-RAE.",
    )
    parser.set_defaults(use_autoencoder=PipelineConfig.use_autoencoder)

    diffusion_group = parser.add_mutually_exclusive_group()
    diffusion_group.add_argument("--use-diffusion", dest="use_diffusion", action="store_true")
    diffusion_group.add_argument(
        "--disable-diffusion",
        dest="use_diffusion",
        action="store_false",
        help="Ablation: skip DDPM and augment with autoencoder reconstructions only.",
    )
    parser.set_defaults(use_diffusion=PipelineConfig.use_diffusion)

    gating_group = parser.add_mutually_exclusive_group()
    gating_group.add_argument("--use-gating", dest="use_gating", action="store_true")
    gating_group.add_argument(
        "--disable-gating",
        dest="use_gating",
        action="store_false",
        help="Ablation: replace ClusterGatedMLP with a plain MLP.",
    )
    parser.set_defaults(use_gating=PipelineConfig.use_gating)

    save_network_group = parser.add_mutually_exclusive_group()
    save_network_group.add_argument(
        "--save-best-networks",
        dest="save_best_networks",
        action="store_true",
        help="Save the best generated FC per class/fold and visualizations.",
    )
    save_network_group.add_argument(
        "--disable-save-best-networks",
        dest="save_best_networks",
        action="store_false",
    )
    parser.set_defaults(save_best_networks=PipelineConfig.save_best_networks)
    parser.add_argument("--best-network-dir", default=PipelineConfig.best_network_dir)
    fc_export_group = parser.add_mutually_exclusive_group()
    fc_export_group.add_argument("--save-all-fc", dest="save_all_fc", action="store_true",
                                 help="Export all generated FCs and class-specific train/held-out FCs (default).")
    fc_export_group.add_argument("--disable-save-all-fc", dest="save_all_fc", action="store_false")
    parser.set_defaults(save_all_fc=PipelineConfig.save_all_fc)
    parser.add_argument("--fc-export-dir", default=PipelineConfig.fc_export_dir)
    parser.add_argument("--experiment-name", default=PipelineConfig.experiment_name,
                        help="Label in topology plots, e.g. Full or w/o BA; does not change model settings.")
    parser.add_argument(
        "--network-vis-top-edges",
        type=int,
        default=PipelineConfig.network_vis_top_edges,
        help="Number of strongest absolute-weight edges drawn in the network graph.",
    )
    parser.add_argument("--seed", type=int, default=PipelineConfig.seed)
    return parser.parse_args()


def resolve_label_csv(args):
    if args.label_csv:
        return os.path.abspath(os.path.expanduser(args.label_csv))
    if args.data_format == "adhd200_aal":
        return None

    data_dir = os.path.abspath(os.path.expanduser(args.fc_dir))
    search_dirs = [data_dir]
    if os.path.basename(data_dir) in {"norm_bold_matrix", "pearson_fc", "FC"}:
        search_dirs.append(os.path.dirname(data_dir))
    names = ["label_2.csv", "labels.csv"] if args.data_format in {"adni_bold", "andi_bold"} else ["labels.csv"]
    for directory in search_dirs:
        for name in names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                print(f"Detected label file: {candidate}")
                return candidate
    return os.path.join(data_dir, names[0])


def main():
    args = parse_args()
    label_csv = resolve_label_csv(args)

    cfg = PipelineConfig(
        fc_dir=args.fc_dir,
        label_csv=label_csv,
        fc_pattern=args.fc_pattern,
        log_path=args.log_path,
        data_format=args.data_format,
        bold_pattern=args.bold_pattern,
        bold_orientation=args.bold_orientation,
        fc_cache_dir=args.fc_cache_dir,
        adni_exclude_sample_ids=tuple(args.adni_exclude_sample_ids),
        fc_binarize=args.fc_binarize,
        fc_threshold=args.fc_threshold,
        fc_threshold_mode=args.fc_threshold_mode,
        adhd_site=args.adhd_site,
        adhd_tc_prefix=args.adhd_tc_prefix,
        adhd_rest=args.adhd_rest,
        k=args.k,
        V=args.V,
        K=args.K,
        ae_epochs=args.ae_epochs,
        ae_bs=args.ae_bs,
        diff_epochs=args.diff_epochs,
        diff_bs=args.diff_bs,
        T=args.T,
        topo_resolution=args.topo_resolution,
        gen_per_class=args.gen_per_class,
        ba_m=args.ba_m,
        lambda_topo=args.lambda_topo,
        lambda_reg=args.lambda_reg,
        lambda_con=args.lambda_con,
        num_clusters=args.num_clusters,
        topk=args.topk,
        enc_hidden=args.enc_hidden,
        enc_layers=args.enc_layers,
        enc_dropout=args.enc_dropout,
        topo_feature_type=args.topo_feature_type,
        diff_backbone=args.diff_backbone,
        use_film=args.use_film,
        use_autoencoder=args.use_autoencoder,
        use_diffusion=args.use_diffusion,
        use_gating=args.use_gating,
        save_best_networks=args.save_best_networks,
        best_network_dir=args.best_network_dir,
        network_vis_top_edges=args.network_vis_top_edges,
        save_all_fc=args.save_all_fc,
        fc_export_dir=args.fc_export_dir,
        experiment_name=args.experiment_name,
        save_sample_diagnostics=args.save_sample_diagnostics,
        diagnostic_dir=args.diagnostic_dir,
        diagnostic_mad_threshold=args.diagnostic_mad_threshold,
        diagnostic_tail_epochs=args.diagnostic_tail_epochs,
        seed=args.seed,
    )

    if cfg.data_format in {"fc", "adni_bold", "andi_bold"} and not os.path.exists(label_csv):
        print(f"[Warning] {label_csv} not found.")
        return

    if args.audit_only:
        from brain_diffusion.sample_diagnostics import audit_inputs

        audit_inputs(cfg)
        return

    from brain_diffusion.data import ADHD200AALDataset, ANDIBOLDDataset, FCMatDataset
    from brain_diffusion.train import run_pipeline

    if cfg.data_format == "adhd200_aal":
        dataset = ADHD200AALDataset(
            cfg.fc_dir,
            site=cfg.adhd_site,
            phenotypic_csv=args.label_csv,
            V=cfg.V,
            tc_prefix=cfg.adhd_tc_prefix,
            rest=cfg.adhd_rest,
        )
    elif cfg.data_format in {"adni_bold", "andi_bold"}:
        dataset = ANDIBOLDDataset(
            cfg.fc_dir,
            label_csv,
            V=cfg.V,
            bold_pattern=cfg.bold_pattern,
            orientation=cfg.bold_orientation,
            fc_cache_dir=cfg.fc_cache_dir,
            excluded_sample_ids=cfg.adni_exclude_sample_ids,
        )
    else:
        dataset = FCMatDataset(cfg.fc_dir, label_csv, V=cfg.V, fc_pattern=cfg.fc_pattern)
    if args.prepare_only:
        print("Dataset preparation completed; --prepare-only requested, skipping training.")
        return
    run_pipeline(dataset, cfg)


if __name__ == "__main__":
    main()
