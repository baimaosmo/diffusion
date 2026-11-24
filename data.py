import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from brain_diffusion.sample_diagnostics import bold_quality, fc_quality


class FCMatDataset(Dataset):
    def __init__(self, fc_dir, label_csv, id_width=7, V=116, fc_pattern=None):
        self.fc_dir = fc_dir
        self.V = V
        self.fc_pattern = fc_pattern

        df = self._read_label_table(label_csv)
        if df.shape[1] == 1:
            df.columns = ["label"]
            df["id"] = [str(i) for i in range(1, len(df) + 1)]
            print("Detected one-column label file. Using sample order as IDs: 1..N")
        else:
            df = df.iloc[:, :2]
            df.columns = ["id", "label"]
            df["id"] = df["id"].astype(str).str.zfill(id_width)

        self.ids = df["id"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.fcs = []
        self.sample_paths = []
        self.quality_records = []

        print("Loading FC matrices...")
        for sid in self.ids:
            fc_path = self._resolve_fc_path(sid)
            mat = self._load_matrix(fc_path)
            if mat.shape != (V, V):
                raise ValueError(f"{fc_path} shape {mat.shape} != ({V},{V})")
            self.quality_records.append(fc_quality(mat))
            self.sample_paths.append(os.path.abspath(fc_path))

            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            mat = np.clip(mat, -1.0, 1.0)
            np.fill_diagonal(mat, 0.0)

            self.fcs.append(torch.from_numpy(mat).float())

        print(f"Loaded {len(self.fcs)} subjects. One FC shape: {self.fcs[0].shape}")

    def __len__(self):
        return len(self.fcs)

    def __getitem__(self, idx):
        return self.fcs[idx], torch.tensor(self.labels[idx], dtype=torch.long)

    def _resolve_fc_path(self, sid):
        raw_id = str(int(sid)) if str(sid).isdigit() else str(sid)
        candidates = []

        if self.fc_pattern:
            candidates.append(self.fc_pattern.format(id=sid, raw_id=raw_id, index=raw_id))

        candidates.extend(
            [
                f"{sid}.csv",
                f"{raw_id}.csv",
                f"sample_{raw_id}.csv",
                f"FC_sample_{raw_id}.csv",
                f"fc_sample_{raw_id}.csv",
                f"FC_{raw_id}.csv",
                f"fc_{raw_id}.csv",
                f"FC_adj_matrix_sample_{raw_id}.csv",
                f"fold_fc_sample_{raw_id}.csv",
                f"Fold_FC_sample_{raw_id}.csv",
                os.path.join("FC", f"FC_adj_matrix_sample_{raw_id}.csv"),
            ]
        )

        for name in candidates:
            path = name if os.path.isabs(name) else os.path.join(self.fc_dir, name)
            if os.path.exists(path):
                return path

        tried = "\n  ".join(candidates[:12])
        raise FileNotFoundError(f"Cannot find FC matrix for sample id={sid}. Tried:\n  {tried}")

    @staticmethod
    def _load_matrix(path):
        try:
            return np.loadtxt(path, dtype=np.float32, delimiter=",")
        except Exception:
            try:
                return np.loadtxt(path, dtype=np.float32)
            except Exception:
                df_fc = pd.read_csv(path, header=None, dtype=str)
                rows = df_fc[0].apply(lambda x: x.strip().split())
                return np.array(rows.tolist(), dtype=np.float32)

    @staticmethod
    def _read_label_table(path):
        errors = []
        for separator in (",", r"\s+"):
            try:
                frame = pd.read_csv(path, header=None, sep=separator, engine="python")
                numeric = frame.apply(pd.to_numeric, errors="coerce")
                if frame.shape[1] in (1, 2) and not numeric.isna().any().any():
                    return numeric
                errors.append(f"separator={separator!r}, shape={frame.shape}")
            except Exception as exc:
                errors.append(str(exc))
        raise ValueError(f"Cannot parse numeric label table {path}: {'; '.join(errors)}")


class ANDIBOLDDataset(Dataset):
    """Convert one raw normalized BOLD CSV per subject into a Pearson FC matrix."""

    def __init__(
        self,
        bold_dir,
        label_file,
        V=116,
        bold_pattern="norm_bold_matrix_sample_{index}.csv",
        orientation="auto",
        fc_cache_dir=None,
        excluded_sample_ids=(),
    ):
        requested_dir = os.path.abspath(os.path.expanduser(bold_dir))
        self.bold_dir = requested_dir
        self.V = V
        self.bold_pattern = bold_pattern
        self.orientation = orientation
        if orientation not in {"auto", "time_by_roi", "roi_by_time"}:
            raise ValueError(
                "orientation must be auto, time_by_roi, or roi_by_time"
            )
        if not os.path.isdir(self.bold_dir):
            raise FileNotFoundError(f"BOLD directory not found: {self.bold_dir}")

        first_filename = bold_pattern.format(index=1, id=1)
        direct_first = os.path.join(self.bold_dir, first_filename)
        nested_dir = os.path.join(self.bold_dir, "norm_bold_matrix")
        nested_first = os.path.join(nested_dir, first_filename)
        if not os.path.isfile(direct_first) and os.path.isfile(nested_first):
            self.bold_dir = nested_dir
            print(f"Detected nested BOLD directory: {self.bold_dir}")

        if not label_file or not os.path.isfile(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        labels = self._read_binary_labels(label_file)
        if not np.all(np.isin(labels, [0.0, 1.0])):
            raise ValueError(
                f"ANDI labels must be binary 0/1; found {sorted(np.unique(labels).tolist())}"
            )
        self.original_sample_count = len(labels)
        if any(isinstance(sid, bool) or not isinstance(sid, (int, np.integer))
               for sid in excluded_sample_ids):
            raise ValueError("Excluded ADNI sample IDs must be integers")
        self.excluded_sample_ids = tuple(sorted(set(excluded_sample_ids)))
        invalid_ids = [sid for sid in self.excluded_sample_ids
                       if not 1 <= sid <= self.original_sample_count]
        if invalid_ids:
            raise ValueError(f"Excluded ADNI sample IDs outside 1..{self.original_sample_count}: {invalid_ids}")
        self.ids = [sid for sid in range(1, self.original_sample_count + 1)
                    if sid not in self.excluded_sample_ids]
        if not self.ids:
            raise ValueError("Cannot exclude every ADNI sample")
        # Select labels by ORIGINAL sample ID; never renumber after exclusions.
        self.labels = [int(labels[sid - 1]) for sid in self.ids]
        self.fcs = []
        self.sample_paths = []
        self.quality_records = []

        cache_dir = None
        if fc_cache_dir:
            cache_dir = os.path.abspath(os.path.expanduser(fc_cache_dir))
            os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading ANDI raw BOLD matrices from {self.bold_dir}...")
        if self.excluded_sample_ids:
            print(f"ADNI excluded sample IDs: {list(self.excluded_sample_ids)}; "
                  f"retaining {len(self.ids)}/{self.original_sample_count}. "
                  "Original files and label file are unchanged.")
        shapes = set()
        for index in self.ids:
            filename = bold_pattern.format(index=index, id=index)
            path = filename if os.path.isabs(filename) else os.path.join(self.bold_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing BOLD sample {index}: {path}. "
                    "Labels and sample files must have a one-to-one order mapping."
                )
            raw = self._load_numeric_csv(path)
            time_by_roi = self._as_time_by_roi(raw, path)
            shapes.add(tuple(time_by_roi.shape))
            fc = self._pearson_fc(time_by_roi)
            self.sample_paths.append(os.path.abspath(path))
            self.quality_records.append(bold_quality(raw, time_by_roi))
            self.fcs.append(torch.from_numpy(fc).float())
            if cache_dir:
                np.savetxt(
                    os.path.join(cache_dir, f"FC_sample_{index}.csv"),
                    fc,
                    delimiter=",",
                    fmt="%.8f",
                )

        next_filename = bold_pattern.format(index=self.original_sample_count + 1,
                                            id=self.original_sample_count + 1)
        next_path = (
            next_filename
            if os.path.isabs(next_filename)
            else os.path.join(self.bold_dir, next_filename)
        )
        if os.path.isfile(next_path):
            raise ValueError(
                f"Found extra BOLD file {next_path}, but the label file contains only "
                f"{self.original_sample_count} original rows. Refusing to risk label misalignment."
            )

        counts = np.bincount(np.asarray(self.labels), minlength=2)
        print(
            f"Loaded {len(self.fcs)} ANDI subjects "
            f"(class0={counts[0]}, class1={counts[1]})."
        )
        print(f"Observed time-by-ROI shapes: {sorted(shapes)}")
        print(f"One Pearson FC shape: {tuple(self.fcs[0].shape)}")
        if cache_dir:
            np.savetxt(
                os.path.join(cache_dir, "labels.csv"),
                np.asarray(self.labels, dtype=np.int64),
                delimiter=",",
                fmt="%d",
            )
            # Use this two-column table when reloading cached FCs with noncontiguous IDs.
            np.savetxt(
                os.path.join(cache_dir, "labels_with_ids.csv"),
                np.column_stack([self.ids, self.labels]), delimiter=",", fmt="%d",
            )
            print(f"Saved Pearson FC matrices to: {cache_dir}")

    def __len__(self):
        return len(self.fcs)

    def __getitem__(self, idx):
        return self.fcs[idx], torch.tensor(self.labels[idx], dtype=torch.long)

    @staticmethod
    def _read_binary_labels(path):
        errors = []
        for delimiter in (",", None):
            try:
                values = np.loadtxt(path, dtype=np.float64, delimiter=delimiter)
                if values.ndim == 0:
                    values = values.reshape(1)
                if values.ndim != 1:
                    raise ValueError(f"expected one column, got shape {values.shape}")
                return values
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(
            f"Could not read one-column numeric label file {path}: {'; '.join(errors)}"
        )

    @staticmethod
    def _load_numeric_csv(path):
        # pandas handles optional headers/index columns more safely than loadtxt.
        frame = pd.read_csv(path, header=None)
        frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        # Remove a textual header row or index column only when it is entirely nonnumeric.
        if len(numeric) and numeric.iloc[0].isna().all():
            numeric = numeric.iloc[1:]
        if numeric.shape[1] and numeric.iloc[:, 0].isna().all():
            numeric = numeric.iloc[:, 1:]
        if numeric.isna().any().any():
            bad = np.argwhere(numeric.isna().to_numpy())
            row, col = bad[0]
            raise ValueError(
                f"{path} contains a non-numeric or missing value near "
                f"row={row + 1}, column={col + 1}"
            )
        matrix = numeric.to_numpy(dtype=np.float32)
        if matrix.ndim != 2 or min(matrix.shape) < 2:
            raise ValueError(f"{path} has invalid BOLD shape {matrix.shape}")
        return matrix

    def _as_time_by_roi(self, matrix, path):
        if self.orientation == "time_by_roi":
            result = matrix
        elif self.orientation == "roi_by_time":
            result = matrix.T
        else:
            row_is_roi = matrix.shape[0] == self.V
            col_is_roi = matrix.shape[1] == self.V
            if row_is_roi == col_is_roi:
                raise ValueError(
                    f"Cannot infer BOLD orientation for {path}, shape={matrix.shape}, V={self.V}. "
                    "Use --bold-orientation time_by_roi or roi_by_time."
                )
            result = matrix.T if row_is_roi else matrix
        if result.shape[1] != self.V:
            raise ValueError(
                f"{path} becomes shape {result.shape}; expected timepoints x {self.V} ROIs"
            )
        if result.shape[0] < 3:
            raise ValueError(f"{path} has too few timepoints: {result.shape[0]}")
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _pearson_fc(time_by_roi):
        fc = np.corrcoef(time_by_roi, rowvar=False).astype(np.float32)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        fc = np.clip(fc, -1.0, 1.0)
        fc = (fc + fc.T) / 2.0
        np.fill_diagonal(fc, 0.0)
        return fc


class ADHD200AALDataset(Dataset):
    def __init__(
        self,
        root_dir,
        site="NYU",
        phenotypic_csv=None,
        V=116,
        tc_prefix="sfnwmrda",
        rest="first",
        diagnosis_col="DX",
        id_col="ScanDir ID",
    ):
        self.root_dir = root_dir
        self.site = site
        self.site_dir = os.path.join(root_dir, site)
        self.V = V
        self.tc_prefix = tc_prefix
        self.rest = rest

        phenotypic_csv = phenotypic_csv or os.path.join(self.site_dir, f"{site}_phenotypic.csv")
        if not os.path.exists(phenotypic_csv):
            raise FileNotFoundError(f"{phenotypic_csv} not found")

        df = pd.read_csv(phenotypic_csv, engine="python")
        labels = self._build_label_map(df, id_col=id_col, diagnosis_col=diagnosis_col)

        self.ids = []
        self.labels = []
        self.fcs = []
        self.sample_paths = []
        self.quality_records = []
        skipped = 0

        print(f"Loading ADHD200 AAL time courses from site={site}...")
        subject_dirs = sorted(path for path in glob(os.path.join(self.site_dir, "*")) if os.path.isdir(path))
        for subject_dir in subject_dirs:
            sid = os.path.basename(subject_dir)
            label = labels.get(sid)
            if label is None:
                label = labels.get(self._raw_id(sid))
            if label is None:
                skipped += 1
                continue

            tc_path = self._resolve_timecourse_path(subject_dir, sid)
            if tc_path is None:
                skipped += 1
                continue

            try:
                tc = self._load_timecourse(tc_path, V=V)
                fc = self._timecourse_to_fc(tc)
            except Exception as exc:
                print(f"[Warning] Skip {sid}: {exc}")
                skipped += 1
                continue

            self.ids.append(sid)
            self.labels.append(label)
            self.fcs.append(torch.from_numpy(fc).float())
            self.sample_paths.append(os.path.abspath(tc_path))
            self.quality_records.append({})

        if not self.fcs:
            raise RuntimeError(f"No usable ADHD200 AAL subjects found in {self.site_dir}")

        counts = {label: self.labels.count(label) for label in sorted(set(self.labels))}
        print(f"Loaded {len(self.fcs)} {site} subjects. Label counts: {counts}. Skipped: {skipped}.")
        print(f"One FC shape: {self.fcs[0].shape}")

    def __len__(self):
        return len(self.fcs)

    def __getitem__(self, idx):
        return self.fcs[idx], torch.tensor(self.labels[idx], dtype=torch.long)

    def _resolve_timecourse_path(self, subject_dir, sid):
        prefix = self.tc_prefix or "*"
        rest_part = "rest_*" if self.rest == "first" else f"rest_{self.rest}"
        pattern = os.path.join(subject_dir, f"{prefix}*{rest_part}_aal_TCs.1D")
        candidates = sorted(glob(pattern))
        if candidates:
            return candidates[0]

        fallback = sorted(glob(os.path.join(subject_dir, f"*{rest_part}_aal_TCs.1D")))
        return fallback[0] if fallback else None

    @staticmethod
    def _build_label_map(df, id_col, diagnosis_col):
        labels = {}
        for _, row in df.iterrows():
            try:
                sid_raw = str(int(row[id_col]))
                dx = int(row[diagnosis_col])
            except Exception:
                continue
            if dx < 0:
                continue
            label = 0 if dx == 0 else 1
            labels[sid_raw] = label
            labels[sid_raw.zfill(7)] = label
        return labels

    @staticmethod
    def _raw_id(sid):
        return str(int(sid)) if str(sid).isdigit() else str(sid)

    @staticmethod
    def _load_timecourse(path, V=116):
        tc = np.loadtxt(path, dtype=np.float32, skiprows=1, usecols=range(2, V + 2))
        if tc.ndim != 2 or tc.shape[1] != V:
            raise ValueError(f"{path} time course shape {tc.shape} does not have {V} AAL columns")
        return np.nan_to_num(tc, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _timecourse_to_fc(tc):
        std = tc.std(axis=0)
        keep = std > 1e-8
        if keep.sum() < 2:
            raise ValueError("Too few non-constant AAL time courses")

        tc_z = tc.copy()
        tc_z[:, ~keep] = 0.0
        fc = np.corrcoef(tc_z, rowvar=False).astype(np.float32)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        fc = np.clip(fc, -1.0, 1.0)
        np.fill_diagonal(fc, 0.0)
        return fc
