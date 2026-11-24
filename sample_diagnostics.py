"""Non-destructive input QC and held-out sample diagnostics.

Quality flags and confident errors are review candidates, not proof of bad labels.
No sample is removed and no diagnostic statistic is fed back into training.
"""

import csv
import datetime
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np


def bold_quality(raw, time_by_roi):
    raw = np.asarray(raw, dtype=np.float64)
    ts = np.asarray(time_by_roi, dtype=np.float64)
    std = ts.std(axis=0)
    median = np.median(ts, axis=0)
    scale = 1.4826 * np.median(np.abs(ts - median), axis=0)
    usable = scale > 1e-8
    z = np.abs(ts[:, usable] - median[usable]) / scale[usable] if usable.any() else np.zeros((1, 1))
    peak_time, peak_column = np.unravel_index(int(z.argmax()), z.shape)
    peak_roi = int(np.flatnonzero(usable)[peak_column]) if usable.any() else None
    return {
        "raw_nonfinite_count": int((~np.isfinite(raw)).sum()),
        "n_timepoints": int(ts.shape[0]),
        "constant_roi_count": int((std <= 1e-8).sum()),
        "bold_std_median": float(np.median(std)),
        "bold_max_robust_z": float(z.max()),
        "bold_peak_timepoint_1based": int(peak_time + 1) if peak_roi is not None else None,
        "bold_peak_roi_1based": peak_roi + 1 if peak_roi is not None else None,
        "bold_spike_fraction": float((z > 6.0).mean()),
    }


def fc_quality(matrix):
    raw = np.asarray(matrix, dtype=np.float64)
    fc = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).copy()
    np.fill_diagonal(fc, 0.0)
    edge = fc[np.triu_indices(len(fc), k=1)]
    return {
        "fc_nonfinite_count": int((~np.isfinite(raw)).sum()),
        "fc_out_of_range_count": int((np.abs(fc) > 1.00001).sum()),
        "fc_asymmetry_max": float(np.abs(fc - fc.T).max()),
        "fc_abs_mean": float(np.abs(edge).mean()),
        "fc_abs_std": float(np.abs(edge).std()),
        "fc_zero_fraction": float((np.abs(edge) < 1e-8).mean()),
        "nearperfect_edge_fraction": float((np.abs(edge) > 0.9999).mean()),
        "zero_fc_roi_count": int((np.abs(fc).sum(axis=1) < 1e-8).sum()),
        # Rounded only for duplicate screening; original FCs are never changed.
        "fc_fingerprint": hashlib.sha256(np.round(fc, 5).astype("<f4").tobytes()).hexdigest(),
    }


def create_diagnostic_dir(root, seed):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result = Path(root).expanduser().resolve() / f"run_{stamp}_seed{seed}"
    result.mkdir(parents=True, exist_ok=False)
    return result


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def add_quality_flags(rows, threshold=3.5):
    rows = [dict(row) for row in rows]
    flags = [[] for _ in rows]
    for i, row in enumerate(rows):
        for key in ("raw_nonfinite_count", "fc_nonfinite_count", "constant_roi_count",
                    "zero_fc_roi_count", "fc_out_of_range_count"):
            if row.get(key, 0) > 0:
                flags[i].append(key)
        if row.get("load_error"):
            flags[i].append("load_error")
        if row.get("fc_asymmetry_max", 0) > 1e-4:
            flags[i].append("asymmetric_fc")
        if row.get("matches_reference_bold") is False:
            flags[i].append("reference_bold_mismatch")

    duplicates = defaultdict(list)
    for i, row in enumerate(rows):
        if row.get("fc_fingerprint"):
            duplicates[row["fc_fingerprint"]].append(i)
    for indices in duplicates.values():
        if len(indices) < 2:
            continue
        conflict = len({rows[i]["label"] for i in indices}) > 1
        for i in indices:
            rows[i]["duplicate_sample_ids"] = ";".join(str(rows[j]["sample_id"]) for j in indices if i != j)
            flags[i].append("duplicate_fc_conflicting_labels" if conflict else "duplicate_fc")

    # Compare scalar quality features within class; pooled class differences can
    # reflect disease signal. These are exploratory flags, not training filters.
    metrics = ("fc_abs_mean", "fc_abs_std", "fc_zero_fraction", "nearperfect_edge_fraction",
               "n_timepoints", "bold_std_median", "bold_spike_fraction")
    # One unusual value among thousands of BOLD values is common. For zero-MAD
    # fractions, require a substantive absolute deviation rather than flagging
    # any nonzero value as an infinite z-score.
    zero_mad_min_delta = {"bold_spike_fraction": 0.001, "fc_zero_fraction": 0.01,
                          "nearperfect_edge_fraction": 0.01}
    for label in sorted({row["label"] for row in rows}):
        for metric in metrics:
            indices = [i for i, row in enumerate(rows) if row["label"] == label
                       and metric in row and not row.get("load_error")]
            if len(indices) < 5:
                continue
            values = np.array([rows[i][metric] for i in indices], dtype=float)
            center = np.median(values)
            scale = 1.4826 * np.median(np.abs(values - center))
            if scale < 1e-10:
                for i, value in zip(indices, values):
                    rows[i][f"{metric}_robust_z"] = None
                    if abs(value - center) > zero_mad_min_delta.get(metric, 1e-5):
                        flags[i].append(f"deviation_{metric}_zero_mad")
                continue
            else:
                z = np.abs(values - center) / scale
            for i, value in zip(indices, z):
                rows[i][f"{metric}_robust_z"] = float(value)
                if value > threshold:
                    flags[i].append(f"outlier_{metric}")
    for row, reasons in zip(rows, flags):
        if row.get("bold_max_robust_z", 0) > 10:
            reasons.append("extreme_bold_value_z_gt_10")
        row["qc_flag_count"] = len(reasons)
        row["qc_flags"] = ";".join(reasons)
    return rows


def save_quality_report(rows, run_dir, cfg, reference_context=None):
    rows = add_quality_flags(rows, cfg.diagnostic_mad_threshold)
    ranked = sorted(rows, key=lambda row: (-bool(row.get("load_error")), -row["qc_flag_count"],
                                           row["dataset_index_0based"]))
    write_csv(Path(run_dir) / "quality_all.csv", rows)
    write_csv(Path(run_dir) / "quality_review.csv", ranked)
    summary = {
        "sample_count": len(rows),
        "load_error_count": sum(bool(row.get("load_error")) for row in rows),
        "flagged_count": sum(row["qc_flag_count"] > 0 for row in rows),
        "mad_threshold": cfg.diagnostic_mad_threshold,
        "config": asdict(cfg),
        "reference_context": reference_context,
        "note": "Flags are review candidates, not proof of corruption. QC does not auto-exclude samples; "
                "only explicitly configured ADNI IDs are excluded from training. Source files are unchanged.",
        "rules": {
            "extreme_bold_value": "ROI-wise |x - median(x)| / (1.4826 * MAD(x)) > 10",
            "spike_fraction": "Fraction of ROI values with robust z > 6; zero-MAD cohort difference must exceed 0.001",
            "distribution_outlier": "Within-class robust z > diagnostic_mad_threshold; zero-MAD features use absolute-deviation rules",
            "duplicates": "Identical FC values after rounding to 5 decimals; review repeated acquisitions and labels",
        },
    }
    (Path(run_dir) / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f">>> [Sample QC] {summary['flagged_count']}/{len(rows)} samples flagged for review; "
          f"{summary['load_error_count']} load errors. Report: {run_dir}/quality_review.csv")
    for row in ranked[:10]:
        if row["qc_flag_count"]:
            print(f"    Review sample={row['sample_id']} label={row['label']}: {row['qc_flags']}")
    return rows


def adni_reference_context(cfg, rows, original_sample_count=None):
    """Check sample order against a companion MAT, without guessing group semantics."""
    if cfg.data_format not in {"adni_bold", "andi_bold"}:
        return None
    reference_count = len(rows) if original_sample_count is None else original_sample_count
    root = Path(cfg.fc_dir).expanduser().resolve()
    reference = next((folder / f"data{reference_count}.mat" for folder in (root, root.parent)
                      if (folder / f"data{reference_count}.mat").is_file()), None)
    if reference is None:
        return None

    import pandas as pd
    from scipy.io import loadmat
    from brain_diffusion.data import ANDIBOLDDataset

    context = {"reference_mat": str(reference), "reference_sample_count": reference_count}
    try:
        data = loadmat(reference, variable_names=["All_Name", "All_Group", "All_Bold"],
                       simplify_cells=True)
        if "All_Name" not in data or "All_Group" not in data:
            context["note"] = "No All_Name/All_Group fields; group mapping not verified"
            return context
        names = np.atleast_1d(data["All_Name"])
        groups = np.atleast_1d(data["All_Group"])
        if len(names) != reference_count or len(groups) != reference_count:
            context["note"] = "Reference sample count differs; group mapping not verified"
            return context
        source_indices = [int(row["sample_id"]) - 1 for row in rows]
        if len(set(source_indices)) != len(rows) or any(i < 0 or i >= reference_count for i in source_indices):
            context["note"] = "Invalid or duplicate original sample IDs; group mapping not verified"
            return context
        clinical = reference.parent / f"data{reference_count}_clinical.csv"
        lookup = {}
        if clinical.is_file():
            frame = pd.read_csv(clinical, sep=None, engine="python")
            if {"Group", "S_number"}.issubset(frame.columns):
                valid = frame.dropna(subset=["Group", "S_number"])
                lookup = {str(row["S_number"]): str(row["Group"]) for _, row in valid.iterrows()}
                context["clinical_file"] = str(clinical)
                context["complete_clinical_rows"] = len(valid)
        reader = ANDIBOLDDataset.__new__(ANDIBOLDDataset)
        reader.V, reader.orientation = cfg.V, cfg.bold_orientation
        reference_bold = data.get("All_Bold")
        group_labels = defaultdict(lambda: defaultdict(int))
        group_clinical = defaultdict(lambda: defaultdict(int))
        for i, row in zip(source_indices, rows):
            name = str(names[i])
            group = int(groups[i])
            match = re.search(r"S\d+", name)
            clinical_group = lookup.get(match.group(0), "unmatched") if match else "unmatched"
            row["original_scan_name"] = name
            row["original_group"] = group
            row["clinical_group"] = clinical_group
            group_labels[str(group)][str(row["label"])] += 1
            group_clinical[str(group)][clinical_group] += 1
            if reference_bold is not None and not row.get("load_error"):
                raw = reader._load_numeric_csv(row["file_path"])
                ts = reader._as_time_by_roi(raw, row["file_path"])
                original = np.asarray(reference_bold[i], dtype=float)
                if original.ndim == 2 and original.shape[1] != cfg.V and original.shape[0] == cfg.V:
                    original = original.T
                row["reference_timepoints"] = int(original.shape[0])
                crop_start = None
                if (original.ndim == 2 and original.shape[1] == ts.shape[1]
                        and len(original) >= len(ts)):
                    for start in range(len(original) - len(ts) + 1):
                        if np.allclose(ts, original[start:start + len(ts)], rtol=1e-6, atol=1e-5):
                            crop_start = start
                            break
                row["matches_reference_bold"] = crop_start is not None
                row["reference_crop_start_1based"] = crop_start + 1 if crop_start is not None else None
                row["reference_match_kind"] = (
                    "full" if crop_start is not None and ts.shape == original.shape
                    else "contiguous_crop" if crop_start is not None else "mismatch"
                )
        context["original_group_to_binary_label_counts"] = dict(group_labels)
        context["original_group_to_clinical_group_counts"] = dict(group_clinical)
        context["bold_reference_mismatch_count"] = sum(row.get("matches_reference_bold") is False for row in rows)
        context["cropped_reference_sample_ids"] = [row["sample_id"] for row in rows
                                                  if row.get("reference_match_kind") == "contiguous_crop"]
        context["note"] = "Original group semantics must be confirmed; clinical rows are joined by scan number, never by row position"
    except (ValueError, OSError, KeyError, IndexError, NotImplementedError) as error:
        context["reference_error"] = str(error)
    return context


def dataset_quality(dataset):
    raw_records = getattr(dataset, "quality_records", [{} for _ in dataset])
    paths = getattr(dataset, "sample_paths", [""] * len(dataset))
    ids = getattr(dataset, "ids", list(range(1, len(dataset) + 1)))
    return [
        {"dataset_index_0based": i, "sample_id": str(ids[i]), "label": int(dataset.labels[i]),
         "file_path": str(Path(paths[i]).resolve()) if paths[i] else "",
         **fc_quality(dataset[i][0].numpy()), **raw_records[i]}
        for i in range(len(dataset))
    ]


def audit_inputs(cfg):
    """Scan all inputs even when one file fails normal dataset loading."""
    from brain_diffusion.data import ANDIBOLDDataset, FCMatDataset

    if cfg.diagnostic_mad_threshold <= 0:
        raise ValueError("Diagnostic MAD threshold must be positive")

    root = Path(cfg.fc_dir).expanduser().resolve()
    reader = ANDIBOLDDataset.__new__(ANDIBOLDDataset)
    reader.V, reader.orientation = cfg.V, cfg.bold_orientation
    if cfg.data_format in {"adni_bold", "andi_bold"}:
        labels = reader._read_binary_labels(cfg.label_csv)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("ADNI labels must be 0/1")
        ids = list(range(1, len(labels) + 1))
        first = cfg.bold_pattern.format(index=1, id=1)
        if not (root / first).is_file() and (root / "norm_bold_matrix" / first).is_file():
            root /= "norm_bold_matrix"
        paths = [root / cfg.bold_pattern.format(index=i, id=i) for i in ids]
    elif cfg.data_format == "fc":
        table = FCMatDataset._read_label_table(cfg.label_csv)
        labels = table.iloc[:, -1].astype(int).tolist()
        ids = list(range(1, len(labels) + 1)) if table.shape[1] == 1 else [str(int(i)).zfill(7) for i in table.iloc[:, 0]]
        resolver = FCMatDataset.__new__(FCMatDataset)
        resolver.fc_dir, resolver.fc_pattern = str(root), cfg.fc_pattern
        paths = []
        for sid in ids:
            try:
                paths.append(Path(resolver._resolve_fc_path(sid)))
            except FileNotFoundError:
                paths.append(None)
    else:
        raise ValueError("--audit-only supports adni_bold, andi_bold, or fc")

    rows = []
    for i, (sid, label, path) in enumerate(zip(ids, labels, paths)):
        row = {"dataset_index_0based": i, "sample_id": str(sid), "label": int(label),
               "file_path": str(path) if path is not None else "", "load_error": ""}
        try:
            if path is None:
                raise FileNotFoundError(f"No FC file for sample {sid}")
            if cfg.data_format in {"adni_bold", "andi_bold"}:
                raw = reader._load_numeric_csv(str(path))
                ts = reader._as_time_by_roi(raw, str(path))
                row.update(bold_quality(raw, ts))
                with np.errstate(invalid="ignore", divide="ignore"):
                    fc = reader._pearson_fc(ts)
            else:
                fc = FCMatDataset._load_matrix(str(path))
            if fc.shape != (cfg.V, cfg.V):
                raise ValueError(f"FC shape {fc.shape} != ({cfg.V}, {cfg.V})")
            row.update(fc_quality(fc))
        except (ValueError, OSError, IndexError) as error:
            row["load_error"] = str(error)
        rows.append(row)
    run_dir = create_diagnostic_dir(cfg.diagnostic_dir, cfg.seed)
    context = adni_reference_context(cfg, rows)
    save_quality_report(rows, run_dir, cfg, context)
    return run_dir


class SampleDiagnostics:
    def __init__(self, dataset, cfg):
        if cfg.diagnostic_mad_threshold <= 0 or cfg.diagnostic_tail_epochs < 1:
            raise ValueError("Diagnostic MAD threshold and tail epoch count must be positive")
        self.cfg = cfg
        self.run_dir = create_diagnostic_dir(cfg.diagnostic_dir, cfg.seed)
        rows = dataset_quality(dataset)
        context = adni_reference_context(cfg, rows, getattr(dataset, "original_sample_count", None))
        self.quality = save_quality_report(rows, self.run_dir, cfg, context)
        self.predictions = []

    def record_fold(self, fold, train_indices, heldout_indices, train_fc, heldout_fc,
                    train_labels, heldout_labels, details):
        if set(train_indices).intersection(heldout_indices):
            raise ValueError("Diagnostic training and held-out indices overlap")
        train = np.asarray(train_fc, dtype=float)
        heldout = np.asarray(heldout_fc, dtype=float)
        y_train = np.asarray(train_labels)
        for j, index in enumerate(heldout_indices):
            label = int(heldout_labels[j])
            p1 = float(details["final_prob_class1"][j])
            true_prob = p1 if label == 1 else 1 - p1
            same = train[y_train == label]
            opposite = train[y_train != label]
            same_center = same.mean(axis=0)
            same_distances = np.sqrt(np.mean((same - same_center) ** 2, axis=1))
            own_distance = float(np.sqrt(np.mean((heldout[j] - same_center) ** 2)))
            opposite_distance = float(np.sqrt(np.mean((heldout[j] - opposite.mean(axis=0)) ** 2)))
            center = float(np.median(same_distances))
            scale = float(1.4826 * np.median(np.abs(same_distances - center)))
            distance_z = (own_distance - center) / scale if scale > 1e-10 else None
            distances = np.mean((train - heldout[j]) ** 2, axis=1)
            nearest = int(np.argmin(distances))
            row = {
                "dataset_index_0based": int(index), "sample_id": self.quality[index]["sample_id"],
                "label": label, "file_path": self.quality[index]["file_path"],
                "seed": int(self.cfg.seed), "fold": int(fold),
                "prediction_epoch": int(details["final_epoch"]),
                "prob_class1": p1, "true_label_probability": true_prob,
                "predicted_label": int(details["final_predicted_label"][j]),
                "is_error": int(details["final_predicted_label"][j] != label),
                "nll": float(-np.log(max(true_prob, 1e-12))),
                "confident_error": int(true_prob <= 0.2),
                "tail_error_rate": float(details["tail_error_rate"][j]),
                "tail_epochs": int(details["tail_epochs"]),
                "same_class_train_centroid_rmse": own_distance,
                "opposite_class_train_centroid_rmse": opposite_distance,
                "closer_to_opposite_class": int(opposite_distance < own_distance),
                "same_class_distance_robust_z": distance_z,
                "nearest_training_sample_id": self.quality[int(train_indices[nearest])]["sample_id"],
                "nearest_training_label": int(y_train[nearest]),
                "qc_flag_count": self.quality[index]["qc_flag_count"],
                "qc_flags": self.quality[index]["qc_flags"],
            }
            for field in ("original_scan_name", "original_group", "clinical_group", "matches_reference_bold"):
                if field in self.quality[index]:
                    row[field] = self.quality[index][field]
            self.predictions.append(row)
        # Persist every completed fold; interrupted training still has partial data.
        write_csv(self.run_dir / "oof_predictions.csv", self.predictions)
        self._write_review()

    def _write_review(self):
        ranked = sorted(self.predictions, key=lambda row: (-row["qc_flag_count"],
                        -row["confident_error"], -row["tail_error_rate"], -row["nll"]))
        for row in ranked:
            reasons = [row["qc_flags"]] if row["qc_flags"] else []
            if row["confident_error"]:
                reasons.append("confident_heldout_error")
            if row["tail_error_rate"] >= 0.8:
                reasons.append("persistent_error_in_last_epochs")
            if (row["same_class_distance_robust_z"] or 0) > self.cfg.diagnostic_mad_threshold:
                reasons.append("far_from_training_class")
            row["review_reasons"] = ";".join(reasons)
        write_csv(self.run_dir / "sample_review.csv", ranked)

    def finish(self):
        if (len(self.predictions) != len(self.quality)
                or len({row["dataset_index_0based"] for row in self.predictions}) != len(self.quality)):
            raise ValueError("Expected one held-out prediction per sample")
        write_csv(self.run_dir / "oof_predictions.csv", self.predictions)
        summary = {
            "sample_count": len(self.quality),
            "oof_final_epoch_acc": float(np.mean([1 - row["is_error"] for row in self.predictions])),
            "prediction_policy": "Fixed final epoch (80), never best-heldout-ACC epoch",
            "tail_policy": "Last epochs of one model; not independent repeated CV predictions",
            "config": asdict(self.cfg),
        }
        (self.run_dir / "prediction_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f">>> [Sample Diagnostics] Review IDs/files: {self.run_dir}/sample_review.csv")


def merge_diagnostics(run_dirs, output_dir):
    """Aggregate independent seeds using sample IDs/labels and identical model settings."""
    grouped = defaultdict(list)
    reference_config = None
    seen_seeds = set()
    ignore_config = {"seed", "log_path", "diagnostic_dir", "fc_export_dir", "best_network_dir",
                     "save_all_fc", "save_best_networks", "save_sample_diagnostics", "experiment_name"}
    for run in run_dirs:
        run = Path(run).expanduser().resolve()
        summary = json.loads((run / "prediction_summary.json").read_text(encoding="utf-8"))
        config = {k: v for k, v in summary["config"].items() if k not in ignore_config}
        if reference_config is not None and config != reference_config:
            raise ValueError("Runs must use the same data/model settings (only seed/output paths may differ)")
        reference_config = config
        seed = summary["config"]["seed"]
        if seed in seen_seeds:
            raise ValueError(f"Repeated seed {seed}: use independent seeds for aggregation")
        seen_seeds.add(seed)
        with (run / "oof_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                grouped[(row["sample_id"], row["label"], row["file_path"])].append(row)
    rows = []
    for (sid, label, path), records in grouped.items():
        if len(records) != len(run_dirs):
            raise ValueError(f"Sample {sid} is missing or duplicated in one of the runs")
        rows.append({
            "sample_id": sid, "label": label, "file_path": path, "n_runs": len(records),
            "error_rate_across_seeds": float(np.mean([int(r["is_error"]) for r in records])),
            "confident_error_rate": float(np.mean([int(r["confident_error"]) for r in records])),
            "mean_true_label_probability": float(np.mean([float(r["true_label_probability"]) for r in records])),
            "mean_nll": float(np.mean([float(r["nll"]) for r in records])),
            "qc_flag_count": int(records[0]["qc_flag_count"]), "qc_flags": records[0]["qc_flags"],
        })
    rows.sort(key=lambda row: (-row["qc_flag_count"], -row["error_rate_across_seeds"], -row["mean_nll"]))
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "sample_review_across_seeds.csv", rows)
    print(f"Combined {len(run_dirs)} seeds, {len(rows)} samples: {output_dir}/sample_review_across_seeds.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine per-sample diagnostics across independent training seeds")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="runs/sample_diagnostics/merged")
    args = parser.parse_args()
    merge_diagnostics(args.runs, args.output_dir)
