import argparse
import csv
import itertools
import json
import os
import random
import time
from dataclasses import asdict, replace

from brain_diffusion.config import PipelineConfig


SEARCH_SPACE = {
    "lambda_con": [0.1, 0.3, 0.5],
    "lambda_topo": [0.05, 0.1, 0.2],
    "ba_m": [5, 20, 70],
    "topo_resolution": [10, 20, 50],
    "num_clusters": [6, 8, 10],
    "topk": [20, 50, 115],
    "enc_hidden": [128, 256],
    "enc_layers": [2, 3],
    "enc_dropout": [0.1, 0.2, 0.3],
    "gen_per_class": [100, 200],
}


FAST_SEARCH_SPACE = {
    "lambda_con": [0.1, 0.3, 0.5],
    "lambda_topo": [0.05, 0.1],
    "ba_m": [20, 70],
    "topo_resolution": [10, 20],
    "num_clusters": [8, 10],
    "topk": [50, 115],
    "enc_hidden": [128],
    "enc_layers": [2, 3],
    "enc_dropout": [0.2],
    "gen_per_class": [100],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Automatic hyperparameter search for brain diffusion pipeline")
    parser.add_argument("--fc-dir", default=PipelineConfig.fc_dir)
    parser.add_argument("--label-csv", default=PipelineConfig.label_csv)
    parser.add_argument("--fc-pattern", default=PipelineConfig.fc_pattern)
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
    binarize_group = parser.add_mutually_exclusive_group()
    binarize_group.add_argument("--fc-binarize", dest="fc_binarize", action="store_true")
    binarize_group.add_argument("--disable-fc-binarize", dest="fc_binarize", action="store_false")
    parser.set_defaults(fc_binarize=PipelineConfig.fc_binarize)
    parser.add_argument("--fc-threshold", type=float, default=PipelineConfig.fc_threshold)
    parser.add_argument(
        "--fc-threshold-mode",
        choices=["positive", "absolute"],
        default=PipelineConfig.fc_threshold_mode,
    )
    parser.add_argument(
        "--fc-threshold-grid",
        type=float,
        nargs="+",
        default=None,
        help="Thresholds to search, for example: 0.2 0.3 0.4 0.5. Requires --fc-binarize.",
    )
    parser.add_argument("--adhd-site", default=PipelineConfig.adhd_site)
    parser.add_argument("--adhd-tc-prefix", default=PipelineConfig.adhd_tc_prefix)
    parser.add_argument("--adhd-rest", default=PipelineConfig.adhd_rest)
    parser.add_argument("--out-dir", default="tuning_runs")
    parser.add_argument("--save-all-fc", action=argparse.BooleanOptionalAction, default=PipelineConfig.save_all_fc,
                        help="Export all FC batches per trial; --no-save-all-fc disables this.")
    parser.add_argument("--mode", choices=["random", "grid"], default="random")
    parser.add_argument("--space", choices=["fast", "full"], default="fast")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--score", choices=["auc_mean", "acc_mean", "f1_mean"], default="auc_mean")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--ae-epochs", type=int, default=80)
    parser.add_argument("--diff-epochs", type=int, default=80)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--ae-bs", type=int, default=32)
    parser.add_argument("--diff-bs", type=int, default=32)
    parser.add_argument("--V", type=int, default=PipelineConfig.V)
    topology_feature_group = parser.add_mutually_exclusive_group()
    topology_feature_group.add_argument(
        "--topo-feature-type",
        choices=["none", "persistence", "avg_clustering", "strength_stats", "efficiency_stats", "laplacian_spectrum", "graph_stats"],
        default=PipelineConfig.topo_feature_type,
    )
    topology_feature_group.add_argument(
        "--disable-persistence", "--disable-topology-conditioning",
        dest="topo_feature_type", action="store_const", const="none",
        help="Tune the clean w/o-PH model without topology-feature conditioning or similarity loss.",
    )
    parser.add_argument("--diff-backbone", choices=["gcn", "transformer"], default=PipelineConfig.diff_backbone)
    film_group = parser.add_mutually_exclusive_group()
    film_group.add_argument("--use-film", dest="use_film", action="store_true", help="Enable FiLM modulation.")
    film_group.add_argument("--disable-film", dest="use_film", action="store_false", help="Disable FiLM modulation for ablation runs.")
    parser.set_defaults(use_film=PipelineConfig.use_film)

    autoencoder_group = parser.add_mutually_exclusive_group()
    autoencoder_group.add_argument("--use-autoencoder", dest="use_autoencoder", action="store_true")
    autoencoder_group.add_argument("--disable-autoencoder", dest="use_autoencoder", action="store_false")
    parser.set_defaults(use_autoencoder=PipelineConfig.use_autoencoder)

    diffusion_group = parser.add_mutually_exclusive_group()
    diffusion_group.add_argument("--use-diffusion", dest="use_diffusion", action="store_true")
    diffusion_group.add_argument("--disable-diffusion", dest="use_diffusion", action="store_false")
    parser.set_defaults(use_diffusion=PipelineConfig.use_diffusion)

    gating_group = parser.add_mutually_exclusive_group()
    gating_group.add_argument("--use-gating", dest="use_gating", action="store_true")
    gating_group.add_argument("--disable-gating", dest="use_gating", action="store_false")
    parser.set_defaults(use_gating=PipelineConfig.use_gating)
    parser.add_argument(
        "--save-best-networks",
        action="store_true",
        default=False,
        help="Save visualizations for every tuning trial (disabled by default).",
    )
    parser.add_argument("--best-network-dir", default=PipelineConfig.best_network_dir)
    parser.add_argument("--network-vis-top-edges", type=int, default=PipelineConfig.network_vis_top_edges)
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
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    label_csv = resolve_label_csv(args)
    if args.data_format in {"fc", "adni_bold", "andi_bold"} and not os.path.exists(label_csv):
        raise FileNotFoundError(f"{label_csv} not found")

    from brain_diffusion.data import ADHD200AALDataset, ANDIBOLDDataset, FCMatDataset
    from brain_diffusion.train import run_pipeline

    if args.data_format == "adhd200_aal":
        dataset = ADHD200AALDataset(
            args.fc_dir,
            site=args.adhd_site,
            phenotypic_csv=args.label_csv,
            V=args.V,
            tc_prefix=args.adhd_tc_prefix,
            rest=args.adhd_rest,
        )
    elif args.data_format in {"adni_bold", "andi_bold"}:
        dataset = ANDIBOLDDataset(
            args.fc_dir,
            label_csv,
            V=args.V,
            bold_pattern=args.bold_pattern,
            orientation=args.bold_orientation,
            fc_cache_dir=args.fc_cache_dir,
        )
    else:
        dataset = FCMatDataset(args.fc_dir, label_csv, V=args.V, fc_pattern=args.fc_pattern)
    space = dict(FAST_SEARCH_SPACE if args.space == "fast" else SEARCH_SPACE)
    if args.fc_threshold_grid:
        if not args.fc_binarize:
            raise ValueError("--fc-threshold-grid requires --fc-binarize")
        if any(not 0.0 < value <= 1.0 for value in args.fc_threshold_grid):
            raise ValueError("Every --fc-threshold-grid value must be in (0, 1]")
        space["fc_threshold"] = args.fc_threshold_grid
    candidates = build_candidates(space, args.mode, args.trials)

    csv_path = os.path.join(args.out_dir, "results.csv")
    best_path = os.path.join(args.out_dir, "best_config.json")
    best_score = float("-inf")
    best_record = None

    for trial_idx, params in enumerate(candidates, start=1):
        effective_params = {
            "fc_binarize": args.fc_binarize,
            "fc_threshold": args.fc_threshold,
            "fc_threshold_mode": args.fc_threshold_mode,
            "use_autoencoder": args.use_autoencoder,
            "use_diffusion": args.use_diffusion,
            "use_gating": args.use_gating,
            "save_best_networks": args.save_best_networks,
            "best_network_dir": os.path.join(args.best_network_dir, f"trial_{trial_idx:03d}"),
            "network_vis_top_edges": args.network_vis_top_edges,
            **params,
        }
        print(f"\n========== Trial {trial_idx}/{len(candidates)} ==========")
        print(json.dumps(effective_params, ensure_ascii=False, indent=2))

        trial_log = os.path.join(args.out_dir, f"trial_{trial_idx:03d}.log")
        cfg = replace(
            PipelineConfig(),
            fc_dir=args.fc_dir,
            label_csv=label_csv,
            fc_pattern=args.fc_pattern,
            data_format=args.data_format,
            bold_pattern=args.bold_pattern,
            bold_orientation=args.bold_orientation,
            fc_cache_dir=args.fc_cache_dir,
            adhd_site=args.adhd_site,
            adhd_tc_prefix=args.adhd_tc_prefix,
            adhd_rest=args.adhd_rest,
            log_path=trial_log,
            save_all_fc=args.save_all_fc,
            fc_export_dir=os.path.join(args.out_dir, "fc_exports"),
            experiment_name=f"trial_{trial_idx:03d}",
            k=args.k,
            V=args.V,
            ae_epochs=args.ae_epochs,
            diff_epochs=args.diff_epochs,
            T=args.T,
            ae_bs=args.ae_bs,
            diff_bs=args.diff_bs,
            topo_feature_type=args.topo_feature_type,
            diff_backbone=args.diff_backbone,
            use_film=args.use_film,
            seed=args.seed + trial_idx,
            **effective_params,
        )

        start_time = time.time()
        try:
            metrics = run_pipeline(dataset, cfg)
            status = "ok"
            error = ""
        except Exception as exc:
            metrics = {}
            status = "failed"
            error = repr(exc)
            print(f"[Trial failed] {error}")

        elapsed_sec = time.time() - start_time
        record = {
            "trial": trial_idx,
            "status": status,
            "elapsed_sec": round(elapsed_sec, 2),
            "error": error,
            **effective_params,
            **metrics,
        }
        append_csv(csv_path, record)

        score_value = record.get(args.score)
        if status == "ok" and score_value is not None and score_value > best_score:
            best_score = score_value
            best_record = record
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump({"score": args.score, "record": best_record, "config": asdict(cfg)}, f, ensure_ascii=False, indent=2)
            print(f"[Best updated] {args.score}={best_score:.4f}")

    print("\n========== Tuning Finished ==========")
    print(f"Results CSV: {csv_path}")
    print(f"Best config: {best_path}")
    if best_record:
        print(json.dumps(best_record, ensure_ascii=False, indent=2))


def build_candidates(space, mode, trials):
    keys = list(space)
    if mode == "grid":
        combos = [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]
        return combos[:trials] if trials else combos

    seen = set()
    candidates = []
    max_unique = 1
    for values in space.values():
        max_unique *= len(values)
    target = min(trials, max_unique)

    while len(candidates) < target:
        params = {key: random.choice(values) for key, values in space.items()}
        sig = tuple((key, params[key]) for key in keys)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(params)
    return candidates


def append_csv(path, record):
    file_exists = os.path.exists(path)
    existing_fields = []
    rewrote_file = False
    if file_exists:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_fields = next(reader, [])

    fields = list(dict.fromkeys(existing_fields + list(record.keys())))
    rows = []
    if file_exists and existing_fields != fields:
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    if rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        rewrote_file = True

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists and not rewrote_file:
            writer.writeheader()
        writer.writerow(record)


if __name__ == "__main__":
    main()
