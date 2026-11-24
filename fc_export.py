"""Export complete FC batches, provenance and relocatable plotting manifests.

Merge exports: python -m brain_diffusion.fc_export --runs RUN1 RUN2 --output DIR
"""

import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_array(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, to_numpy(value), allow_pickle=False)
    os.replace(temporary, path)


def array_hash(value):
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str((arr.shape, str(arr.dtype))).encode())
    digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def plot_issues(array, v):
    problems = []
    if array.ndim != 3 or array.shape[1:] != (v, v):
        return [f"Unexpected FC shape: {array.shape}"]
    if len(array) < 2:
        problems.append("Fewer than two samples: distribution plot needs at least two.")
    if not np.isfinite(array).all():
        problems.append("FC contains NaN or infinity; saved unchanged, not replaced by real samples.")
    else:
        if np.max(np.abs(array), initial=0) > 1 + 1e-5:
            problems.append("FC values outside [-1,1].")
        if not np.allclose(array, array.transpose(0, 2, 1), atol=1e-5):
            problems.append("FC is not symmetric.")
    return problems


class FCExporter:
    def __init__(self, cfg, input_fc, model_fc, labels, subject_ids=None):
        self.cfg = cfg
        self.name = cfg.experiment_name.strip()
        if not self.name or self.name == "Real":
            raise ValueError("experiment_name must be nonempty and cannot be 'Real'.")
        self.input_fc = to_numpy(input_fc)
        self.model_fc = to_numpy(model_fc)
        self.labels = np.asarray(labels, dtype=np.int64)
        if self.model_fc.shape != (len(self.labels), cfg.V, cfg.V):
            raise ValueError("Export: dataset FC shape does not match labels or cfg.V.")
        if not np.all(np.isin(self.labels, [0, 1])):
            raise ValueError("Export expects binary labels 0/1, matching this pipeline.")
        self.ids = [str(x) for x in subject_ids] if subject_ids is not None else [f"dataset_index:{i}" for i in range(len(labels))]
        if len(self.ids) != len(labels):
            raise ValueError("Export: subject ID count does not match labels.")
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(cfg.fc_export_dir).expanduser().resolve() / f"run_{stamp}_seed{cfg.seed}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.fingerprint = hashlib.sha256(json.dumps({
            "input_fc": array_hash(self.input_fc), "labels": self.labels.tolist(), "ids": self.ids,
        }, sort_keys=True).encode()).hexdigest()
        self.run_info = {
            "schema_version": 1, "status": "in_progress", "experiment_name": self.name,
            "dataset_fingerprint": self.fingerprint, "config": asdict(cfg),
            "sample_index_base": 0, "generation_stage": "model output before repair/filter/selection",
            "heldout_role": "validation fold exported as real_test; never used as generation condition",
            "density_sweep_supported": not cfg.fc_binarize, "folds": {},
        }
        self.manifests = {c: {"class_id": c, "dataset_fingerprint": self.fingerprint,
                             "density_sweep_supported": not cfg.fc_binarize,
                             "run_complete": False, "units": []} for c in [0, 1]}
        write_json(self.run_dir / "dataset.json", {"dataset_fingerprint": self.fingerprint,
                   "subjects": [{"dataset_index": i, "subject_id": sid, "label": int(self.labels[i])}
                                for i, sid in enumerate(self.ids)]})
        self._flush()

    def _flush(self):
        write_json(self.run_dir / "run_metadata.json", self.run_info)
        for c, manifest in self.manifests.items():
            write_json(self.run_dir / f"manifest_class{c}.json", manifest)

    def begin_fold(self, fold, train_indices, heldout_indices):
        tr = np.asarray(train_indices, dtype=np.int64)
        va = np.asarray(heldout_indices, dtype=np.int64)
        combined = np.concatenate([tr, va])
        if (len(combined) != len(self.labels) or len(np.unique(combined)) != len(self.labels)
                or np.any(combined < 0) or np.any(combined >= len(self.labels))):
            raise ValueError("Train/held-out indices must be a disjoint partition of the dataset.")
        fold_dir = self.run_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(exist_ok=False)
        split = {"train_indices": tr.tolist(), "heldout_indices": va.tolist()}
        split_hash = hashlib.sha256(json.dumps({"dataset": self.fingerprint, **split}, sort_keys=True).encode()).hexdigest()
        write_json(fold_dir / "split.json", {**split, "split_hash": split_hash,
                   "train_subject_ids": [self.ids[i] for i in tr], "heldout_subject_ids": [self.ids[i] for i in va]})
        info = {"split_hash": split_hash, "source": "real_only", "classes": {}}
        for c in [0, 1]:
            directory = fold_dir / f"class_{c}"
            directory.mkdir()
            train = tr[self.labels[tr] == c]
            test = va[self.labels[va] == c]
            for role, indices in [("train", train), ("test", test)]:
                save_array(directory / f"real_{role}.npy", self.model_fc[indices])
                save_array(directory / f"real_{role}_indices.npy", indices)
                if self.cfg.fc_binarize:
                    save_array(directory / f"real_{role}_input.npy", self.input_fc[indices])
            info["classes"][str(c)] = {"real_train_count": len(train), "real_test_count": len(test),
                                       "generated_count": 0}
        self.run_info["folds"][str(fold)] = info
        self._flush()

    def save_generated(self, fc0, fc1, source, *, fold):
        if source not in {"latent_diffusion", "direct_fc_diffusion", "autoencoder_reconstruction"}:
            raise ValueError(f"Unknown generation source: {source}")
        info = self.run_info["folds"][str(fold)]
        if info["source"] != "real_only":
            raise ValueError(f"Generated FCs already exported for fold {fold}.")
        for c, fc in enumerate([fc0, fc1]):
            array = to_numpy(fc)
            directory = self.run_dir / f"fold_{fold:02d}" / f"class_{c}"
            save_array(directory / "generated.npy", array)
            # Rows stay in generation order, not similarity order.
            save_array(directory / "generated_indices.npy", np.arange(len(array), dtype=np.int64))
            real = np.load(directory / "real_test.npy", allow_pickle=False)
            problems = plot_issues(array, self.cfg.V) + plot_issues(real, self.cfg.V)
            details = {**info["classes"][str(c)], "generated_count": len(array),
                       "generated_shape": list(array.shape), "source": source, "plot_issues": problems}
            if source == "autoencoder_reconstruction":
                # Reconstructions map to training subjects; diffusion samples do not.
                indices = np.load(directory / "real_train_indices.npy", allow_pickle=False)
                save_array(directory / "reconstruction_subject_indices.npy", indices)
            write_json(directory / "metadata.json", details)
            info["classes"][str(c)] = details
            self.manifests[c]["units"].append({
                "id": f"seed{self.cfg.seed}_fold{fold:02d}", "seed": self.cfg.seed, "fold": fold,
                "split_hash": info["split_hash"], "source": source, "plot_issues": problems,
                "real": f"fold_{fold:02d}/class_{c}/real_test.npy",
                "methods": {self.name: f"fold_{fold:02d}/class_{c}/generated.npy"},
            })
        info["source"] = source
        self._flush()

    def finish(self):
        self.run_info["status"] = "complete"
        for manifest in self.manifests.values():
            manifest["run_complete"] = True
            if not manifest["units"]:
                manifest["note"] = "No generated FCs: both autoencoder and diffusion are disabled. Real FCs exported only."
        self._flush()


def merge_manifests(run_dirs, output_dir, method_labels=None):
    """Combine methods/seeds only when dataset identity and splits agree."""
    output = Path(output_dir).expanduser().resolve()
    results = {}
    for c in [0, 1]:
        units, fingerprint = {}, None
        for run in run_dirs:
            run = Path(run).expanduser().resolve()
            manifest = json.loads((run / f"manifest_class{c}.json").read_text(encoding="utf-8"))
            if not manifest["run_complete"] or not manifest["units"]:
                raise ValueError(f"Incomplete run or no generated FCs: {run}")
            if not manifest["density_sweep_supported"]:
                raise ValueError(f"Binary FC run is unsuitable for the continuous-FC density plot: {run}")
            if fingerprint is not None and fingerprint != manifest["dataset_fingerprint"]:
                raise ValueError("Cannot merge runs from different datasets, sample orders, or input FCs.")
            fingerprint = manifest["dataset_fingerprint"]
            for item in manifest["units"]:
                if item["plot_issues"]:
                    raise ValueError(f"Invalid plotting input in {run}, {item['id']}: {item['plot_issues']}")
                real_path = (run / item["real"]).resolve()
                real = np.load(real_path, allow_pickle=False)
                if item["id"] not in units:
                    units[item["id"]] = {"id": item["id"], "split_hash": item["split_hash"],
                                         "real": os.path.relpath(real_path, output),
                                         "methods": {}, "sources": {}, "real_hash": array_hash(real)}
                target = units[item["id"]]
                if target["split_hash"] != item["split_hash"] or target["real_hash"] != array_hash(real):
                    raise ValueError(f"Mismatched train/held-out split or real FCs in {item['id']}.")
                for stored_method, path in item["methods"].items():
                    method = (method_labels or {}).get(str(run), stored_method)
                    if method_labels and str(run) in method_labels and len(item["methods"]) != 1:
                        raise ValueError("Label overrides require one method per source run.")
                    if method in target["methods"]:
                        raise ValueError(f"Duplicate method {method!r} in {item['id']}; set distinct --experiment-name values.")
                    generated = (run / path).resolve()
                    if not generated.is_file():
                        raise FileNotFoundError(generated)
                    target["methods"][method] = os.path.relpath(generated, output)
                    target["sources"][method] = item["source"]
        expected = set(next(iter(units.values()))["methods"])
        if len(expected) > 4 or any(set(u["methods"]) != expected for u in units.values()):
            raise ValueError("Every fold/seed must contain the same 1–4 methods; no partial intersections are silently used.")
        results[c] = {"class_id": c, "dataset_fingerprint": fingerprint, "density_sweep_supported": True,
                      "run_complete": True, "units": [units[key] for key in sorted(units)]}
    output.mkdir(parents=True, exist_ok=True)
    for c, manifest in results.items():
        write_json(output / f"manifest_class{c}.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Saved merged manifests: {merge_manifests(args.runs, args.output)}")


if __name__ == "__main__":
    main()
