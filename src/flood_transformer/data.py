from __future__ import annotations

import os
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_names: List[str]
    flood_threshold: float
    meta_df: pd.DataFrame
    data_source: str


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _find_col(df: pd.DataFrame, candidates: List[str], allowed_cols: Optional[List[str]] = None) -> Optional[str]:
    cols = allowed_cols if allowed_cols is not None else list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    norm_map = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in cols}
    for cand in candidates:
        k = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if k in norm_map:
            return norm_map[k]

    # 仅对长度>=3的候选词做包含匹配，避免 "T" 命中 "date_utc" 这种误匹配
    for c in cols:
        low = c.lower()
        tokens = [x for x in re.split(r"[^a-z0-9]+", low) if x]
        if any((len(cand) >= 3) and (cand.lower() in low or cand.lower() in tokens) for cand in candidates):
            return c
    return None


def _maybe_parse_date(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc:
            out = df.copy()
            out[c] = pd.to_datetime(out[c], errors="coerce")
            if out[c].notna().mean() > 0.8:
                return out.sort_values(c).reset_index(drop=True)
    return df.reset_index(drop=True)


def _load_station_data(config: ExperimentConfig) -> Tuple[pd.DataFrame, str, List[str], str]:
    forcing_files = sorted([f for f in os.listdir(config.forcing_dir) if f.endswith(".csv")])
    streamflow_files = sorted([f for f in os.listdir(config.streamflow_dir) if f.endswith(".csv")]) if os.path.exists(config.streamflow_dir) else []

    forcing_ids = {os.path.splitext(f)[0] for f in forcing_files}
    streamflow_ids = {os.path.splitext(f)[0] for f in streamflow_files}

    station_id = config.station_id
    if station_id is None:
        common = sorted(forcing_ids.intersection(streamflow_ids))
        if common:
            station_id = common[0]
        elif forcing_ids:
            station_id = sorted(forcing_ids)[0]
        else:
            raise FileNotFoundError("Forcing 目录下未找到 CSV 数据")

    forcing_path = os.path.join(config.forcing_dir, f"{station_id}.csv")
    if not os.path.exists(forcing_path):
        raise FileNotFoundError(f"找不到 forcing 文件: {forcing_path}")

    forcing_df = _maybe_parse_date(pd.read_csv(forcing_path))

    stream_path = os.path.join(config.streamflow_dir, f"{station_id}.csv")
    if os.path.exists(stream_path):
        stream_df = _maybe_parse_date(pd.read_csv(stream_path))
        q_stream_col = _find_col(stream_df, config.streamflow_col_candidates)
        if q_stream_col is None:
            numeric_stream_cols = [c for c in stream_df.columns if pd.api.types.is_numeric_dtype(stream_df[c])]
            if len(numeric_stream_cols) == 0:
                raise ValueError(f"streamflow 文件缺少数值列: {stream_path}")
            q_stream_col = numeric_stream_cols[0]

        target_col = "target_q"
        stream_df = stream_df.rename(columns={q_stream_col: target_col})

        date_f = _find_col(forcing_df, ["date", "time"])
        date_s = _find_col(stream_df, ["date", "time"])
        if date_f is not None and date_s is not None:
            merged = forcing_df.merge(stream_df, left_on=date_f, right_on=date_s, how="inner", suffixes=("", "_q"))
        else:
            min_len = min(len(forcing_df), len(stream_df))
            merged = pd.concat([forcing_df.iloc[:min_len].reset_index(drop=True), stream_df.iloc[:min_len].reset_index(drop=True)], axis=1)
    else:
        raise FileNotFoundError(f"找不到 streamflow 文件: {stream_path}")

    return merged, station_id, list(forcing_df.columns), target_col


def _build_feature_target(
    df: pd.DataFrame,
    config: ExperimentConfig,
    forcing_cols: Optional[List[str]] = None,
    target_col: str = "target_q",
):
    forcing_cols = forcing_cols or list(df.columns)
    forcing_numeric_cols = [c for c in forcing_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if target_col not in df.columns:
        raise ValueError(f"目标流量列不存在: {target_col}")

    used_cols = list(dict.fromkeys(forcing_numeric_cols))
    if len(used_cols) == 0:
        raise ValueError("Forcing 中没有可用数值特征列")

    if config.include_streamflow_history_input and target_col not in used_cols:
        used_cols.append(target_col)

    selected_cols = list(dict.fromkeys(used_cols + [target_col]))
    out = df[selected_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feat = out[used_cols].to_numpy(dtype=np.float32)
    q = out[target_col].to_numpy(dtype=np.float32)

    q_mean, q_std = q.mean(), q.std() + 1e-6
    x_mean, x_std = feat.mean(axis=0), feat.std(axis=0) + 1e-6

    feat_norm = (feat - x_mean) / x_std
    q_norm = (q - q_mean) / q_std

    meta = {
        "target_col": target_col,
        "feature_cols": used_cols,
        "q_mean": float(q_mean),
        "q_std": float(q_std),
        "x_mean": x_mean,
        "x_std": x_std,
    }
    return feat_norm, q_norm, q, meta, out


def _make_windows(
    features: np.ndarray,
    target_norm: np.ndarray,
    seq_len: int,
    pred_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list, y_list = [], []
    end = len(features) - seq_len - pred_horizon + 1
    for i in range(max(0, end)):
        x_list.append(features[i : i + seq_len])
        y_list.append(target_norm[i + seq_len + pred_horizon - 1])

    if not x_list:
        raise ValueError("样本太短，无法构建时序窗口，请减小 seq_len 或检查数据")

    return np.stack(x_list), np.array(y_list, dtype=np.float32)


def _synthetic_data(config: ExperimentConfig):
    n = 3000
    t = np.arange(n)
    rain = np.maximum(0, np.random.randn(n) * 3 + (np.sin(t / 15.0) + 1.2) * 2)
    temp = 15 + 10 * np.sin(t / 365 * 2 * np.pi) + np.random.randn(n)
    pet = np.maximum(0.1, 2 + 0.2 * temp + np.random.randn(n) * 0.3)
    soil = np.zeros(n)
    q = np.zeros(n)
    for i in range(1, n):
        soil[i] = 0.92 * soil[i - 1] + 0.08 * rain[i] - 0.05 * pet[i]
        q[i] = 0.85 * q[i - 1] + 0.25 * rain[i] + 0.1 * max(soil[i], 0)

    df = pd.DataFrame(
        {
            "P": rain,
            "T": temp,
            "PET": pet,
            "Soil": soil,
            "target_q": q,
        }
    )
    feat_norm, q_norm, q_raw, meta, clean_df = _build_feature_target(df, config, forcing_cols=["P", "T", "PET", "Soil"], target_col="target_q")
    return feat_norm, q_norm, q_raw, meta, clean_df, "synthetic"


def load_data(config: ExperimentConfig) -> DataBundle:
    if not os.path.exists(config.forcing_dir):
        raise FileNotFoundError("forcing 目录不存在")

    merged_df, station_id, forcing_cols, target_col = _load_station_data(config)
    feat_norm, q_norm, q_raw, meta, clean_df = _build_feature_target(
        merged_df,
        config,
        forcing_cols=forcing_cols,
        target_col=target_col,
    )
    data_source = "observed"

    x, y = _make_windows(feat_norm, q_norm, config.seq_len, config.pred_horizon)

    n = len(x)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    x_test, y_test = x[n_train + n_val :], y[n_train + n_val :]

    train_loader = DataLoader(SeqDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_val, y_val), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(x_test, y_test), batch_size=config.batch_size, shuffle=False)

    flood_threshold = float(np.quantile(q_raw, config.flood_quantile_threshold))
    meta_df = clean_df.copy()
    meta_df["station_id"] = station_id

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_names=meta["feature_cols"],
        flood_threshold=flood_threshold,
        meta_df=meta_df,
        data_source=data_source,
    )
