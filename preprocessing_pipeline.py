from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller


# =========================
# CONFIGURATION
# =========================
OUTPUT_DIR = Path(r"C:\Users\ARYA\Gemastik div III\Dataset\processed")
KONSUMEN_PATH = OUTPUT_DIR / "konsumen_raw.csv"
PRODUSEN_PATH = OUTPUT_DIR / "produsen_raw.csv"

TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

LOOKBACK = 30
HORIZON = 14
RANDOM_STATE = 42

MAX_WEEKEND_INTERP = 2
MAX_HOLIDAY_FFILL = 7
MAX_INTERP_GAP = 30
OUTLIER_IQR_MULT = 3.0
RET_CLIP_ABS = 50.0
ISOF_CONTAMINATION = 0.01

COMMODITIES = ["beras", "bawang", "cabai"]
PROVINCES_CONSUMER = ["dki", "jabar", "jateng", "jatim", "sulsel", "sumut"]
PROVINCES_PRODUCER = ["jabar", "jateng", "sulsel", "sumut"]  # no jatim for modeling
PROVINCES_COMMON = ["jabar", "jateng", "sulsel", "sumut"]  # for spread

# (province, start, end, side)
VOID_MASKS = {
    "sulsel_consumer_2024": ("sulsel", "2024-01-02", "2024-12-31", "consumer"),
    "jatim_producer_2022": ("jatim", "2022-01-01", "2022-12-31", "producer"),
    "jatim_producer_2024": ("jatim", "2024-01-02", "2024-12-31", "producer"),
}

LOG_PATH = OUTPUT_DIR / "preprocessing_log.txt"


def log(msg: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def reset_log() -> None:
    LOG_PATH.write_text("", encoding="utf-8")


def _run_lengths(mask: pd.Series) -> pd.Series:
    grp = (mask != mask.shift()).cumsum()
    return mask.groupby(grp).transform("sum") * mask


def load_and_merge() -> pd.DataFrame:
    log("STEP 1 - Loading raw CSV files.")
    kons = pd.read_csv(KONSUMEN_PATH)
    prod = pd.read_csv(PRODUSEN_PATH)

    kons["date"] = pd.to_datetime(kons["date"])
    prod["date"] = pd.to_datetime(prod["date"])

    for c in kons.columns:
        if c != "date":
            kons[c] = pd.to_numeric(kons[c], errors="coerce")
    for c in prod.columns:
        if c != "date":
            prod[c] = pd.to_numeric(prod[c], errors="coerce")

    kons = kons.rename(columns={c: f"{c}_cons" for c in kons.columns if c != "date"})
    prod = prod.rename(columns={c: f"{c}_prod" for c in prod.columns if c != "date"})

    df = pd.merge(kons, prod, on="date", how="outer").sort_values("date").set_index("date")
    full_index = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    df = df.reindex(full_index)
    df.index.name = "date"
    log(f"Merged shape: {df.shape}")
    return df


def apply_hard_void_mask(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("STEP 2 - Applying hard void masks.")
    out = df.copy()
    void_mask = pd.DataFrame(False, index=out.index, columns=out.columns)

    for key, (prov, start, end, side) in VOID_MASKS.items():
        candidate_cols = [c for c in out.columns if c.endswith("_cons" if side == "consumer" else "_prod")]
        cols = [c for c in candidate_cols if f"_{prov}_" in c]
        idx = (out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))
        if not cols:
            log(f"Void mask '{key}' skipped (no matching columns).")
            continue
        out.loc[idx, cols] = np.nan
        void_mask.loc[idx, cols] = True
        log(f"Applied void mask '{key}' to {len(cols)} columns, range {start}..{end}.")

    valid_mask = ~out.isna()
    valid_mask.to_csv(OUTPUT_DIR / "valid_mask.csv")
    void_mask.to_csv(OUTPUT_DIR / "void_mask.csv")
    log("Saved valid_mask.csv and void_mask.csv.")
    return out, void_mask


def fill_missing_by_type(df: pd.DataFrame, void_mask: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("STEP 3 - Missing value handling by type (A/B/C).")
    out = df.copy()
    long_gap_flags = pd.DataFrame(False, index=out.index, columns=[f"flag_{c}_gap_long" for c in out.columns])
    weekend = out.index.dayofweek >= 5

    for col in out.columns:
        s = out[col].copy()
        na_mask = s.isna()
        non_void_na = na_mask & (~void_mask[col])
        if not non_void_na.any():
            continue

        run_len = _run_lengths(non_void_na.astype(int))

        # Type A: weekend
        type_a = non_void_na & weekend
        type_a = type_a & (run_len <= MAX_WEEKEND_INTERP)
        if type_a.any():
            temp = s.copy()
            temp[type_a] = np.nan
            s = temp.interpolate(method="linear", limit=MAX_WEEKEND_INTERP, limit_area="inside")

        # Recompute after Type A
        non_void_na = s.isna() & (~void_mask[col])
        run_len = _run_lengths(non_void_na.astype(int))

        # Type B: weekday short holiday gaps (1-7)
        weekday = out.index.dayofweek < 5
        type_b = non_void_na & weekday & (run_len >= 1) & (run_len <= MAX_HOLIDAY_FFILL)
        if type_b.any():
            ff = s.ffill(limit=MAX_HOLIDAY_FFILL)
            s[type_b] = ff[type_b]

        # Recompute after Type B
        non_void_na = s.isna() & (~void_mask[col])
        run_len = _run_lengths(non_void_na.astype(int))

        # Type C: long structural gaps
        type_c = non_void_na & (run_len > MAX_HOLIDAY_FFILL)
        if type_c.any():
            long_gap_flags.loc[type_c, f"flag_{col}_gap_long"] = True
            # never synthesize too long by interpolation. Use ffill then bfill.
            s = s.ffill(limit=MAX_INTERP_GAP).bfill(limit=MAX_INTERP_GAP)

        # Final cleanup for remaining non-void NaN
        remaining = s.isna() & (~void_mask[col])
        if remaining.any():
            s = s.ffill().bfill()
            remaining_after = s.isna() & (~void_mask[col])
            if remaining_after.any():
                s[remaining_after] = s.median()
                log(f"{col}: fallback median fill for {remaining_after.sum()} values.")

        out[col] = s

    # Void values remain NaN by design
    non_void_nan_total = int((out.isna() & (~void_mask)).sum().sum())
    if non_void_nan_total != 0:
        raise AssertionError(f"Non-void NaN remains after filling: {non_void_nan_total}")

    log("Missing handling complete with no non-void NaN.")
    return out, long_gap_flags


def winsorize_iqr_per_year(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("STEP 4.1 - IQR winsorization by year.")
    out = df.copy()
    records: List[dict] = []

    for col in cols:
        for year, idx in out.groupby(out.index.year).groups.items():
            vals = out.loc[idx, col]
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - OUTLIER_IQR_MULT * iqr
            upper = q3 + OUTLIER_IQR_MULT * iqr
            clipped = vals.clip(lower=lower, upper=upper)
            changed = vals != clipped
            if changed.any():
                for dt in vals.index[changed]:
                    records.append(
                        {
                            "date": dt,
                            "column": col,
                            "year": year,
                            "original": float(vals.loc[dt]),
                            "clipped": float(clipped.loc[dt]),
                            "lower": float(lower),
                            "upper": float(upper),
                        }
                    )
                out.loc[idx, col] = clipped

    log(f"Winsorized points: {len(records)}")
    return out, pd.DataFrame(records)


def add_isolation_forest_flags(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    log("STEP 4.2 - Isolation Forest multivariate flags on training period.")
    out = df.copy()
    train_df = out.loc[:TRAIN_END, base_cols]
    iso = IsolationForest(contamination=ISOF_CONTAMINATION, random_state=RANDOM_STATE)
    iso.fit(train_df.values)

    preds = iso.predict(out[base_cols].values)  # -1 outlier, 1 inlier
    row_outlier = preds == -1
    log(f"IsolationForest row outliers: {int(row_outlier.sum())}")

    for col in base_cols:
        fcol = f"outlier_flag_{col}"
        out[fcol] = row_outlier.astype(int)
    return out


def add_rolling_zscore_flags(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    log("STEP 4.3 - Rolling Z-score features.")
    out = df.copy()
    for col in cols:
        rm = out[col].rolling(30, min_periods=15).mean()
        rs = out[col].rolling(30, min_periods=15).std()
        z = (out[col] - rm) / rs.replace(0, np.nan)
        out[f"zscore_{col}"] = z
        out[f"spike_raw_{col}"] = (z.abs() > 3).astype(int)
    return out


def compute_spread_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    log("STEP 5 - Computing spread features.")
    out = df.copy()
    spread_cols: List[str] = []
    spread_pct_cols: List[str] = []
    spread_vol_cols: List[str] = []

    for com in COMMODITIES:
        for prov in PROVINCES_COMMON:
            c_cons = f"{com}_{prov}_cons"
            c_prod = f"{com}_{prov}_prod"
            if c_cons not in out.columns or c_prod not in out.columns:
                continue
            s_col = f"spread_{com}_{prov}"
            out[s_col] = out[c_cons] - out[c_prod]
            neg = out[s_col] < 0
            if neg.any():
                log(f"Negative spread found in {s_col}: {int(neg.sum())}; set NaN then interpolate.")
                out.loc[neg, s_col] = np.nan
                out[s_col] = out[s_col].interpolate(limit=7, limit_direction="both").ffill().bfill()

            s_pct = f"spread_pct_{com}_{prov}"
            out[s_pct] = np.where(out[c_prod] != 0, (out[s_col] / out[c_prod]) * 100.0, np.nan)
            out[s_pct] = out[s_pct].replace([np.inf, -np.inf], np.nan).ffill().bfill()

            s_vol = f"spread_vol_{com}_{prov}"
            out[s_vol] = out[s_col].rolling(30, min_periods=10).std().ffill().bfill()

            spread_cols.append(s_col)
            spread_pct_cols.append(s_pct)
            spread_vol_cols.append(s_vol)

    return out, spread_cols, spread_pct_cols, spread_vol_cols


def compute_log_returns(df_level: pd.DataFrame, price_and_spread_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    log("STEP 6 - Computing log returns.")
    out = df_level.copy()
    return_cols: List[str] = []
    for col in price_and_spread_cols:
        rcol = f"log_return_{col}"
        out[rcol] = np.log(out[col] / out[col].shift(1)) * 100.0
        out[rcol] = out[rcol].replace([np.inf, -np.inf], np.nan)
        out[rcol] = out[rcol].clip(-RET_CLIP_ABS, RET_CLIP_ABS)
        return_cols.append(rcol)

    out = out.iloc[1:].copy()  # drop first row from differencing
    return out, return_cols


def add_engineered_features(df: pd.DataFrame, return_cols: List[str]) -> pd.DataFrame:
    log("STEP 7 - Feature engineering.")
    out = df.copy()
    windows = [7, 14, 30]

    for col in return_cols:
        for w in windows:
            out[f"rolling_mean_{col}_{w}d"] = out[col].rolling(w, min_periods=max(3, w // 2)).mean()
            out[f"rolling_std_{col}_{w}d"] = out[col].rolling(w, min_periods=max(3, w // 2)).std()
            out[f"rolling_max_{col}_{w}d"] = out[col].rolling(w, min_periods=max(3, w // 2)).max()
            out[f"rolling_min_{col}_{w}d"] = out[col].rolling(w, min_periods=max(3, w // 2)).min()

        out[f"momentum_{col}_7d"] = out[col].rolling(7, min_periods=4).sum()
        out[f"momentum_{col}_14d"] = out[col].rolling(14, min_periods=7).sum()
        out[f"momentum_{col}_30d"] = out[col].rolling(30, min_periods=15).sum()
        out[f"ewm_vol_{col}"] = out[col].ewm(span=14, min_periods=5).std()

    # Cross-commodity spread return within same province and side
    sides = ["cons", "prod"]
    for side in sides:
        provinces = PROVINCES_CONSUMER if side == "cons" else PROVINCES_PRODUCER
        for p in provinces:
            c_b = f"log_return_beras_{p}_{side}"
            c_c = f"log_return_cabai_{p}_{side}"
            c_bw = f"log_return_bawang_{p}_{side}"
            if c_b in out.columns and c_c in out.columns:
                out[f"spread_return_cabai_beras_{p}_{side}"] = out[c_c] - out[c_b]
            if c_b in out.columns and c_bw in out.columns:
                out[f"spread_return_bawang_beras_{p}_{side}"] = out[c_bw] - out[c_b]

    # Calendar cyclical
    dow = out.index.dayofweek
    month = out.index.month
    woy = out.index.isocalendar().week.astype(int)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["woy_sin"] = np.sin(2 * np.pi * woy / 52)
    out["woy_cos"] = np.cos(2 * np.pi * woy / 52)

    # Events
    out["event_bbm_2022"] = ((out.index >= "2022-09-03") & (out.index <= "2022-09-04")).astype(int)
    out["event_elnino_2023"] = ((out.index >= "2023-06-01") & (out.index <= "2023-12-31")).astype(int)

    # Approximate Ramadan/Lebaran ranges (2021-2025) without extra dependency.
    ramadan_ranges = [
        ("2021-04-13", "2021-05-12"),
        ("2022-04-02", "2022-05-01"),
        ("2023-03-23", "2023-04-21"),
        ("2024-03-11", "2024-04-09"),
        ("2025-03-01", "2025-03-30"),
    ]
    lebaran_days = ["2021-05-13", "2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31"]
    ramadan_flag = pd.Series(0, index=out.index)
    for s, e in ramadan_ranges:
        ramadan_flag.loc[(out.index >= s) & (out.index <= e)] = 1
    out["event_ramadan"] = ramadan_flag.astype(int)

    lebaran_flag = pd.Series(0, index=out.index)
    for d in lebaran_days:
        ds = pd.Timestamp(d) - pd.Timedelta(days=7)
        de = pd.Timestamp(d) + pd.Timedelta(days=7)
        lebaran_flag.loc[(out.index >= ds) & (out.index <= de)] = 1
    out["event_lebaran"] = lebaran_flag.astype(int)
    log("Event flags added (Ramadan/Lebaran via fixed date ranges).")

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return out


def generate_spike_label(return_series: pd.Series, rolling_std: pd.Series, horizon: int = HORIZON, min_duration: int = 2) -> pd.Series:
    threshold = 2.0 * rolling_std
    is_above = (return_series > threshold).astype(int)
    run_length = is_above.groupby((is_above != is_above.shift()).cumsum()).transform("count") * is_above
    sustained = (run_length >= min_duration).astype(int)
    label = sustained.shift(-horizon).fillna(0).astype(int)
    return label


def build_spike_labels(df: pd.DataFrame, target_return_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("STEP 8 - Generating spike labels.")
    labels = pd.DataFrame(index=df.index)
    report_rows: List[dict] = []
    for col in target_return_cols:
        rolling_std = df[col].rolling(30, min_periods=15).std().replace(0, np.nan).bfill().ffill()

        # adaptive threshold to target prevalence in [2, 20]
        thr_mult = 2.0
        label = pd.Series(0, index=df.index, dtype=int)
        prevalence = 0.0
        for _ in range(20):
            threshold = thr_mult * rolling_std
            is_above = (df[col] > threshold).astype(int)
            run_length = is_above.groupby((is_above != is_above.shift()).cumsum()).transform("count") * is_above
            sustained = (run_length >= 2).astype(int)
            label = sustained.shift(-HORIZON).fillna(0).astype(int)
            prevalence = float(label.mean() * 100)
            if 2.0 <= prevalence <= 20.0:
                break
            if prevalence < 2.0:
                thr_mult = max(0.2, thr_mult - 0.1)
            else:
                thr_mult = min(3.0, thr_mult + 0.1)

        lcol = f"spike_{col.replace('log_return_', '')}"
        labels[lcol] = label
        report_rows.append({"column": lcol, "prevalence_pct": prevalence, "threshold_mult": thr_mult})

    report = pd.DataFrame(report_rows).sort_values("column")
    report.to_csv(OUTPUT_DIR / "spike_label_report.csv", index=False)
    log("Saved spike_label_report.csv.")
    return labels, report


def build_geo_sim_matrix(provinces: List[str], distances_km: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    n = len(provinces)
    m = np.zeros((n, n), dtype=float)
    for i, a in enumerate(provinces):
        for j, b in enumerate(provinces):
            if i == j:
                m[i, j] = 1.0
                continue
            key = (a, b) if (a, b) in distances_km else (b, a)
            d = distances_km.get(key, 3000.0)
            m[i, j] = 1.0 / (1.0 + d)
    # normalize to [0,1]
    mn, mx = m.min(), m.max()
    if mx > mn:
        m = (m - mn) / (mx - mn)
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=provinces, columns=provinces)


def build_adjacency_matrix(df_feat: pd.DataFrame, provinces: List[str], alpha: float = 0.5) -> pd.DataFrame:
    log("STEP 10 - Building adjacency matrix 6x6.")
    distances_km = {
        ("dki", "jabar"): 150,
        ("dki", "jateng"): 450,
        ("dki", "jatim"): 800,
        ("dki", "sulsel"): 2100,
        ("dki", "sumut"): 1900,
        ("jabar", "jateng"): 300,
        ("jabar", "jatim"): 650,
        ("jabar", "sulsel"): 2200,
        ("jabar", "sumut"): 2000,
        ("jateng", "jatim"): 350,
        ("jateng", "sulsel"): 1800,
        ("jateng", "sumut"): 2100,
        ("jatim", "sulsel"): 1500,
        ("jatim", "sumut"): 2400,
        ("sulsel", "sumut"): 2900,
    }
    geo_sim = build_geo_sim_matrix(provinces, distances_km)

    beras_cols = [f"log_return_beras_{p}_cons" for p in provinces if f"log_return_beras_{p}_cons" in df_feat.columns]
    corr = df_feat.loc[:TRAIN_END, beras_cols].corr().clip(lower=0)
    if corr.max().max() > corr.min().min():
        corr = (corr - corr.min().min()) / (corr.max().max() - corr.min().min())
    corr = corr.reindex(index=[f"log_return_beras_{p}_cons" for p in provinces], columns=[f"log_return_beras_{p}_cons" for p in provinces]).fillna(0.0)
    corr.index = provinces
    corr.columns = provinces

    A = alpha * geo_sim.values + (1.0 - alpha) * corr.values
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    adj = pd.DataFrame(A, index=provinces, columns=provinces)
    adj.to_csv(OUTPUT_DIR / "adjacency_matrix_6x6.csv")
    log("Saved adjacency_matrix_6x6.csv.")
    return adj


def temporal_split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    log("STEP 11 - Applying temporal split.")
    splits = {
        "train": df.loc["2021-01-02":"2022-12-31"].copy(),
        "val": df.loc[VAL_START:VAL_END].copy(),
        "test": df.loc[TEST_START:TEST_END].copy(),
    }
    assert splits["train"].index.max() < splits["val"].index.min()
    assert splits["val"].index.max() < splits["test"].index.min()
    return splits


def normalize_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    spread_related_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    log("STEP 9 - Normalization with train-only fit.")
    train_n = train.copy()
    val_n = val.copy()
    test_n = test.copy()
    scaler_dict: Dict[str, object] = {}

    spread_related_set = set(spread_related_cols)

    for col in feature_cols:
        if col in spread_related_set:
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        train_n[col] = scaler.fit_transform(train[[col]]).ravel()
        val_n[col] = scaler.transform(val[[col]]).ravel()
        test_n[col] = scaler.transform(test[[col]]).ravel()
        scaler_dict[col] = scaler

    joblib.dump(scaler_dict, OUTPUT_DIR / "scalers.pkl")
    log("Saved scalers.pkl.")
    return train_n, val_n, test_n, scaler_dict


class FoodPriceDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for iTransformer.
    """

    def __init__(
        self,
        df_features: pd.DataFrame,
        df_labels: pd.DataFrame,
        df_mask: pd.DataFrame,
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
        variate_cols: List[str] | None = None,
    ) -> None:
        if variate_cols is None:
            raise ValueError("variate_cols must be provided.")

        self.X = torch.tensor(df_features[variate_cols].values, dtype=torch.float32)
        self.y_return = torch.tensor(df_features[variate_cols].values, dtype=torch.float32)
        self.spike_label = torch.tensor(df_labels.values, dtype=torch.float32)
        self.mask = torch.tensor(df_mask[variate_cols].values, dtype=torch.bool)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.X) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_window = self.X[idx : idx + self.lookback]
        y_window = self.y_return[idx + self.lookback : idx + self.lookback + self.horizon]
        spike_window = self.spike_label[idx + self.lookback : idx + self.lookback + self.horizon]
        mask_window = self.mask[idx : idx + self.lookback]
        return {
            "x": x_window,
            "y": y_window,
            "spike_label": spike_window,
            "mask": mask_window,
        }


def validate_pipeline(
    train_feat: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    non_void_cols: List[str],
    return_target_cols: List[str],
    spike_report: pd.DataFrame,
    level_df: pd.DataFrame,
    spread_cols: List[str],
    adj: pd.DataFrame,
) -> None:
    log("STEP 14 - Running validation checklist.")

    assert train_feat[non_void_cols].isna().sum().sum() == 0, "NaN found in train non-void feature columns."
    assert train_df.index.max() < val_df.index.min(), "Train/Val leakage."
    assert val_df.index.max() < test_df.index.min(), "Val/Test leakage."

    for col in return_target_cols:
        s = train_df[col].dropna()
        if len(s) < 50:
            continue
        p_val = adfuller(s)[1]
        if p_val >= 0.05:
            raise AssertionError(f"{col} still non-stationary after transform (p={p_val:.4f})")

    for _, row in spike_report.iterrows():
        p = float(row["prevalence_pct"])
        if not (2.0 <= p <= 20.0):
            log(f"Warning: {row['column']} spike prevalence out of target [2,20]: {p:.2f}%")

    for col in spread_cols:
        if level_df[col].min() < -100:
            raise AssertionError(f"Suspicious negative spread in {col}")

    assert np.allclose(adj.values, adj.values.T), "Adjacency not symmetric."
    assert np.allclose(np.diag(adj.values), 1.0), "Adjacency diagonal not 1."
    log("Validation passed ✅")


def main() -> None:
    reset_log()
    log("Preprocessing pipeline started.")

    # STEP 1
    df = load_and_merge()

    # STEP 2
    df_masked, void_mask = apply_hard_void_mask(df)

    # STEP 3
    df_filled, long_gap_flags = fill_missing_by_type(df_masked, void_mask)

    # Base columns
    consumer_cols = [c for c in df_filled.columns if c.endswith("_cons")]
    producer_cols = [c for c in df_filled.columns if c.endswith("_prod")]
    base_price_cols = consumer_cols + producer_cols

    # STEP 4.1
    df_wins, wins_log = winsorize_iqr_per_year(df_filled, base_price_cols)
    wins_log.to_csv(OUTPUT_DIR / "winsorized_log.csv", index=False)
    log("Saved winsorized_log.csv.")

    # STEP 4.2
    df_if = add_isolation_forest_flags(df_wins, base_price_cols)

    # STEP 4.3
    df_z = add_rolling_zscore_flags(df_if, base_price_cols)

    # Attach long gap flags
    df_level = pd.concat([df_z, long_gap_flags], axis=1)

    # STEP 5
    df_level, spread_cols, spread_pct_cols, spread_vol_cols = compute_spread_features(df_level)

    # Keep level prices for inverse transform
    df_level.to_csv(OUTPUT_DIR / "df_level_full.csv")
    log("Saved df_level_full.csv.")

    # STEP 6
    return_base_cols = base_price_cols + spread_cols
    df_ret, return_cols = compute_log_returns(df_level, return_base_cols)

    # STEP 7
    df_feat = add_engineered_features(df_ret, return_cols)

    # STEP 8
    # target variates: 30 columns (18 consumer + 12 producer excluding jatim producer)
    target_price_cols = consumer_cols + [c for c in producer_cols if "_jatim_" not in c]
    target_return_cols = [f"log_return_{c}" for c in target_price_cols if f"log_return_{c}" in df_feat.columns]
    labels, spike_report = build_spike_labels(df_feat, target_return_cols)

    # STEP 10 adjacency (uses feature df with returns)
    adj = build_adjacency_matrix(df_feat, PROVINCES_CONSUMER, alpha=0.5)

    # STEP 11 split
    feature_splits = temporal_split(df_feat)
    label_splits = temporal_split(labels)
    mask_after_ret = (~void_mask).iloc[1:].copy()  # align with return drop first row
    mask_splits = temporal_split(mask_after_ret.astype(bool))

    # Feature selection for model input:
    # - target return variates (30)
    # - spread returns + spread pct + spread vol + event/calendar + outlier/long-gap flags
    spread_return_cols = [c for c in df_feat.columns if c.startswith("log_return_spread_")]
    extra_cols = [
        c
        for c in df_feat.columns
        if (
            c.startswith("rolling_")
            or c.startswith("momentum_")
            or c.startswith("ewm_vol_")
            or c.startswith("spread_return_")
            or c.startswith("event_")
            or c in ["dow_sin", "dow_cos", "month_sin", "month_cos", "woy_sin", "woy_cos"]
            or c.startswith("outlier_flag_")
            or c.startswith("flag_")
            or c.startswith("spike_raw_")
            or c.startswith("zscore_")
            or c.startswith("spread_pct_")
            or c.startswith("spread_vol_")
        )
    ]
    feature_cols = sorted(set(target_return_cols + spread_return_cols + spread_pct_cols + spread_vol_cols + extra_cols))
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    # STEP 9 normalization
    spread_related_cols = [c for c in feature_cols if ("spread_" in c and not c.startswith("log_return_"))]
    train_n, val_n, test_n, _ = normalize_splits(
        feature_splits["train"], feature_splits["val"], feature_splits["test"], feature_cols, spread_related_cols
    )

    # Persist outputs
    train_n.to_csv(OUTPUT_DIR / "df_train_features.csv")
    val_n.to_csv(OUTPUT_DIR / "df_val_features.csv")
    test_n.to_csv(OUTPUT_DIR / "df_test_features.csv")
    label_splits["train"].to_csv(OUTPUT_DIR / "df_train_labels.csv")
    label_splits["val"].to_csv(OUTPUT_DIR / "df_val_labels.csv")
    label_splits["test"].to_csv(OUTPUT_DIR / "df_test_labels.csv")
    log("Saved train/val/test feature and label CSV files.")

    # Build example datasets (sanity only)
    mask_for_dataset = pd.DataFrame(index=mask_splits["train"].index)
    for vcol in target_return_cols:
        level_col = vcol.replace("log_return_", "")
        if level_col in mask_splits["train"].columns:
            mask_for_dataset[vcol] = mask_splits["train"][level_col].astype(bool)
        else:
            mask_for_dataset[vcol] = True

    _ = FoodPriceDataset(
        df_features=train_n[feature_cols],
        df_labels=label_splits["train"],
        df_mask=mask_for_dataset,
        lookback=LOOKBACK,
        horizon=HORIZON,
        variate_cols=target_return_cols,
    )
    log("FoodPriceDataset sanity initialization passed.")

    # Validation checklist
    validate_pipeline(
        train_feat=train_n,
        train_df=feature_splits["train"],
        val_df=feature_splits["val"],
        test_df=feature_splits["test"],
        non_void_cols=feature_cols,
        return_target_cols=target_return_cols,
        spike_report=spike_report,
        level_df=df_level,
        spread_cols=spread_cols,
        adj=adj,
    )

    print("All preprocessing validations passed.")
    log("Preprocessing pipeline finished successfully.")


if __name__ == "__main__":
    main()
